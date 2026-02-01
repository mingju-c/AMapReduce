#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses
import json
import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from FlashOAgents import OpenAIServerModel
from FlashOAgents.utils import make_json_serializable
from base_agent import ManageAgent
from utils import read_jsonl

from src.evaluation.data_loader import WideSearchDataLoader, WideSearchResponse
from src.evaluation.evaluation import evaluate_single_query

from loguru import logger as loguru_logger
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


def _configure_loguru() -> None:
    """Reduce external debug logs."""
    try:
        loguru_logger.remove()
    except Exception:
        pass
    loguru_logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "INFO"))


_configure_loguru()

_LOADER_CACHE: Dict[Tuple[str, str], WideSearchDataLoader] = {}

# Keep Expmemory completions short and inexpensive.
DEFAULT_MEMORY_MAX_OUTPUT_TOKENS = 1024


class MemoryLLMWrapper:
    """Adapter so OpenAIServerModel can satisfy Expmemory's LLMCallable protocol."""

    def __init__(self, model: OpenAIServerModel) -> None:
        self.model = model

    def __call__(
        self,
        messages: List[Any],
        temperature: float = 0.1,
        max_tokens: int = DEFAULT_MEMORY_MAX_OUTPUT_TOKENS,
        stop_sequences: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> str:
        # num_comps is managed upstream by Expmemory; ignore here.
        del num_comps

        payload = [{"role": msg.role, "content": msg.content} for msg in messages]
        try:
            response = self.model(
                payload,
                stop_sequences=stop_sequences,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            response = self.model(
                payload,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        content = getattr(response, "content", None)
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        if content is None and hasattr(response, "raw"):
            raw = response.raw
            if isinstance(raw, dict):
                content = raw.get("content")

        return content or ""


def run_eval(
    args: argparse.Namespace,
    instance_id: Optional[str],
    agent_result: Any,
    trial_idx: int,
    instance_dir: Path,
    *,
    force: bool = False,
) -> bool:
    """Run WideSearch evaluation for the given instance-trial pair."""
    if not args.enable_eval or not instance_id:
        return False

    result_root = Path(args.eval_result_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    csv_path = result_root / f"{instance_id}_trial_{trial_idx}_eval_result.csv"
    json_path = csv_path.with_suffix(".json")
    if json_path.exists() and not force:
        return False

    if not agent_result:
        instance_path = instance_dir / f"{instance_id}_trial_{trial_idx}.jsonl"
        try:
            raw_line = instance_path.read_text(encoding="utf-8").strip()
            if raw_line:
                agent_result = json.loads(raw_line).get("agent_result")
        except FileNotFoundError:
            logger.warning(
                "Instance file missing for %s (trial %d); skip evaluation.",
                instance_id,
                trial_idx,
            )
            return False

    data_path = args.eval_data_path or args.infile
    cache_key = (
        str(Path(data_path).resolve()),
        str(Path(args.eval_answer_root).resolve()),
    )
    loader = _LOADER_CACHE.get(cache_key)
    if loader is None:
        loader = WideSearchDataLoader(data_path, args.eval_answer_root)
        _LOADER_CACHE[cache_key] = loader

    response_text = _ensure_response_text(agent_result)
    if not response_text:
        logger.warning("Skip evaluation for %s: empty response", instance_id)
        return False

    query = loader.load_query_by_instance_id(instance_id)
    response_obj = WideSearchResponse(instance_id=instance_id, response=response_text)
    eval_result = evaluate_single_query(
        query,
        response_obj,
        str(csv_path),
        args.eval_model_config_name,
    )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataclasses.asdict(eval_result), handle, ensure_ascii=False, indent=2)
    logger.info("Evaluation completed for %s (trial %d)", instance_id, trial_idx)
    return True


def _ensure_response_text(agent_result: Any) -> Optional[str]:
    """Convert an agent result payload into plain text for evaluation."""
    if agent_result is None:
        return None
    if isinstance(agent_result, str):
        return agent_result
    if isinstance(agent_result, dict):
        for key in ("content", "agent_result", "final_answer", "output"):
            value = agent_result.get(key)
            if isinstance(value, str):
                return value
        try:
            return json.dumps(agent_result, ensure_ascii=False)
        except TypeError:
            return str(agent_result)
    return str(agent_result)


def _count_tool_calls(result: Dict[str, Any]) -> int:
    def _accumulate_from_steps(steps: List[Dict[str, Any]]) -> int:
        subtotal = 0
        if not isinstance(steps, list):
            return subtotal
        for step in steps:
            if isinstance(step, dict):
                tool_calls = step.get("tool_calls")
                if isinstance(tool_calls, list):
                    subtotal += len(tool_calls)
                elif isinstance(tool_calls, dict):
                    subtotal += 1
        return subtotal

    total = _accumulate_from_steps(result.get("agent_trajectory"))

    mapreduce_records = result.get("mapreduce_task_records") or []
    if isinstance(mapreduce_records, list):
        for record in mapreduce_records:
            if not isinstance(record, dict):
                continue
            batches = record.get("batches") or []
            if not isinstance(batches, list):
                continue
            for batch in batches:
                if not isinstance(batch, dict):
                    continue
                attempts = batch.get("attempts") or []
                if not isinstance(attempts, list):
                    continue
                for attempt in attempts:
                    if not isinstance(attempt, dict):
                        continue
                    agent_trace = attempt.get("agent_trace") or {}
                    if not isinstance(agent_trace, dict):
                        continue
                    steps = agent_trace.get("steps")
                    total += _accumulate_from_steps(steps)
    return total


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _truncate_trajectory_until_mapreduce(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep trajectory only up to the first mapreducetool call."""
    truncated: list[Dict[str, Any]] = []
    for step in trajectory or []:
        truncated.append(step)
        tool_calls = step.get("tool_calls") if isinstance(step, dict) else None
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("name") == "mapreducetool":
                    return truncated
    return truncated


def _prune_result_for_storage(result: Dict[str, Any], keep_until_mapreduce: bool = False) -> Dict[str, Any]:
    """Prune result for storage; optionally keep trajectory until first mapreducetool."""
    pruned = dict(result)
    if keep_until_mapreduce and "agent_trajectory" in pruned:
        pruned["agent_trajectory"] = _truncate_trajectory_until_mapreduce(pruned.get("agent_trajectory", []))
    else:
        pruned.pop("agent_trajectory", None)
    pruned.pop("agent_trajectory_full", None)
    pruned.pop("mapreduce_task_records", None)
    pruned.pop("mapreduce_last_run_trace", None)
    # Keep the last assistant message as compact response context
    prediction = _ensure_response_text(pruned.get("agent_result")) or ""
    pruned["messages"] = [{"role": "assistant", "content": prediction}] if prediction else []
    return pruned


def _capture_full_steps(manage_agent: ManageAgent) -> Optional[List[Dict[str, Any]]]:
    try:
        agent_fn = getattr(manage_agent, "agent_fn", None)
        memory = getattr(agent_fn, "memory", None)
        if memory is None:
            return None
        if hasattr(memory, "get_full_steps"):
            return memory.get_full_steps()
        steps = getattr(memory, "steps", None)
        if steps is None:
            return None
        return [step.dict() for step in steps]
    except Exception:
        return None


def _attach_mapreduce_records(result: Dict[str, Any], manage_agent: ManageAgent) -> None:
    tool = getattr(manage_agent, "mapreduce_tool", None)
    if tool is None:
        return
    try:
        result["mapreduce_task_records"] = list(getattr(tool, "task_records", []) or [])
        result["mapreduce_last_run_trace"] = getattr(tool, "last_run_trace", None)
    except Exception:
        return


def _parse_instance_filename(name: str) -> Optional[Tuple[str, int]]:
    """Extract (instance_id, trial_idx) from instance file name."""
    if not name.endswith(".jsonl"):
        return None
    stem = name[:-6]
    if "_trial_" not in stem:
        return stem, 0
    base, trial_str = stem.rsplit("_trial_", 1)
    try:
        return base, int(trial_str)
    except ValueError:
        return None


def _filter_selected_tasks(
    data: Iterable[Dict[str, Any]],
    selected: Optional[Iterable[str]],
) -> list[Dict[str, Any]]:
    if not selected:
        return list(data)
    indices: list[int] = []
    for item in selected:
        try:
            indices.append(int(item))
        except ValueError:
            continue
    return [entry for idx, entry in enumerate(data) if idx in indices]


def process_trial(
    item: Dict[str, Any],
    model_kwargs: Dict[str, Any],
    summary_interval: int,
    prompts_type: str,
    max_steps: int,
    manage_kwargs: Dict[str, Any],
    trial_idx: int,
    *,
    capture_full_trace: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute inference for a single query."""
    model = OpenAIServerModel(**model_kwargs)
    if hasattr(model, "reset_usage"):
        model.reset_usage()
    
    manage_agent = ManageAgent(
        model,
        summary_interval=summary_interval,
        prompts_type=prompts_type,
        max_steps=max_steps,
        **manage_kwargs,
    )
    question = item["query"]
    instance_id = item.get("instance_id")
    reasoning_start_time = datetime.utcnow().isoformat() + "Z"
    try:
        agent_output = manage_agent(question)
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.error("Exception occurred while calling manage_agent: %s", exc)
        return None
    reasoning_end_time = datetime.utcnow().isoformat() + "Z"

    usage = {}
    if hasattr(model, "get_usage"):
        usage = model.get_usage(reset=True)
        usage.setdefault("prompt_tokens", 0)
        usage.setdefault("completion_tokens", 0)
        usage.setdefault("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"])
        usage.setdefault("request_count", 0)

    result = {
        "instance_id": instance_id,
        "trial_idx": trial_idx,
        "question": question,
        "reasoning_start_time": reasoning_start_time,
        "reasoning_end_time": reasoning_end_time,
        **agent_output,
    }
    if capture_full_trace:
        full_steps = _capture_full_steps(manage_agent)
        if full_steps is not None:
            result["agent_trajectory_full"] = full_steps
    _attach_mapreduce_records(result, manage_agent)
    result["token_usage"] = usage
    result["tool_call_count"] = _count_tool_calls(result)
    return result


def calc_summary_results(
    eval_dir: Path,
    trial_num: int,
    summary_result_path: Path,
    *,
    allowed_instances: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate evaluation metrics across trials and instances."""
    if not eval_dir.exists():
        logger.warning("Evaluation directory %s does not exist. Skip summary.", eval_dir)
        return {}

    allowed_set: Optional[set[str]] = None
    if allowed_instances:
        allowed_set = {iid for iid in allowed_instances if iid}
        if allowed_set is not None and not allowed_set:
            allowed_set = None

    metrics = [
        "score",
        "precision_by_row",
        "recall_by_row",
        "f1_by_row",
        "precision_by_item",
        "recall_by_item",
        "f1_by_item",
    ]

    instance_metrics: Dict[str, Dict[str, Dict[int, float]]] = {}
    for json_file in eval_dir.glob("*_eval_result.json"):
        stem = json_file.stem
        if not stem.endswith("_eval_result"):
            continue
        base = stem[: -len("_eval_result")]
        if "_trial_" not in base:
            continue
        instance_id, trial_str = base.rsplit("_trial_", 1)
        try:
            trial_idx = int(trial_str)
        except ValueError:
            continue
        if allowed_set is not None and instance_id not in allowed_set:
            continue
        try:
            result = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read evaluation file %s: %s", json_file, exc)
            continue
        metric_map = instance_metrics.setdefault(instance_id, {m: {} for m in metrics})
        for metric in metrics:
            value = result.get(metric)
            if isinstance(value, (int, float)):
                metric_map[metric][trial_idx] = float(value)

    aggregated: Dict[str, list[Dict[str, float]]] = {m: [] for m in metrics}
    for instance_id, metric_map in instance_metrics.items():
        for metric, values_dict in metric_map.items():
            if len(values_dict) < trial_num:
                continue
            ordered = [
                value
                for _, value in sorted(values_dict.items(), key=lambda kv: kv[0])
            ][:trial_num]
            aggregated[metric].append(
                {
                    "avg_n": float(sum(ordered) / len(ordered)),
                    "max_n": float(max(ordered)),
                    "min_n": float(min(ordered)),
                }
            )

    summary: Dict[str, Dict[str, float]] = {}
    for metric, records in aggregated.items():
        if not records:
            continue
        summary[metric] = {
            "avg_n": float(sum(r["avg_n"] for r in records) / len(records)),
            "max_n": float(sum(r["max_n"] for r in records) / len(records)),
            "min_n": float(sum(r["min_n"] for r in records) / len(records)),
        }

    summary_result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary metrics written to %s", summary_result_path)
    return summary


def main(args: argparse.Namespace) -> None:
    args.enable_eval = _coerce_bool(args.enable_eval)
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    default_model = os.environ.get("DEFAULT_MODEL")
    if not default_model:
        raise ValueError("DEFAULT_MODEL environment variable is required.")

    model_kwargs = {
        "model_id": default_model,
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 32768,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
        "temperature": 0.0,
        "top_p": 1.0,
    }

    expmemory = None
    if args.enable_expmemory:
        try:
            os.environ.setdefault("OPENAI_API_BASE", os.environ.get("OPENAI_API_BASE") or "")
            os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "")
            from mas.utils import EmbeddingFunc  # lazy import to avoid env issues
            from mas.memory.mas_memory.Expmemory import Expmemory
            from mas.memory.common import MASMessage
            from mas.llm import Message as MASMessageType, LLMCallable

            memory_model = OpenAIServerModel(**model_kwargs)
            embedding_func = EmbeddingFunc()

            expmemory = Expmemory(
                namespace=args.expmemory_namespace,
                global_config={
                    "working_dir": args.expmemory_workdir,
                    "hop": args.expmemory_hop,
                    "start_insights_threshold": args.expmemory_start_insights_threshold,
                    "rounds_per_insights": args.expmemory_rounds_per_insights,
                    "insights_point_num": args.expmemory_insights_point_num,
                    "merge_insights_interval": args.expmemory_merge_insights_interval,
                },
                llm_model=MemoryLLMWrapper(memory_model),  # type: ignore[arg-type]
                embedding_func=embedding_func,
            )
        except Exception as exc:
            logger.warning("Failed to initialize Expmemory, continuing without it: %s", exc)
            expmemory = None

    if args.infile.lower().endswith(".json"):
        data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    else:
        data = read_jsonl(args.infile)

    data = _filter_selected_tasks(data, args.selected_tasks)
    selected_instance_ids = {
        entry.get("instance_id")
        for entry in data
        if entry.get("instance_id")
    }

    instance_output_dir = Path(args.outfile).resolve()
    instance_output_dir.mkdir(parents=True, exist_ok=True)
    full_response_dir = Path(args.full_response_dir).resolve() if args.full_response_dir else None
    if full_response_dir and full_response_dir == instance_output_dir:
        full_response_dir = instance_output_dir / "full_responses"
        logger.warning(
            "full_response_dir matches outfile; writing full responses under %s",
            full_response_dir,
        )
    full_response_lock = threading.Lock() if full_response_dir else None
    store_full_trace = _coerce_bool(args.store_full_trace)
    trace_until_mapreduce = args.trace_until_mapreduce

    usage_path = instance_output_dir / "usage.json"
    usage_lock = threading.Lock()
    def write_instance_file(record: Dict[str, Any]) -> None:
        instance_id = record.get("instance_id")
        trial_idx = record.get("trial_idx", 0)
        if not instance_id:
            return
        instance_path = instance_output_dir / f"{instance_id}_trial_{trial_idx}.jsonl"
        with instance_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_full_response(record: Dict[str, Any]) -> None:
        if not full_response_dir:
            return
        instance_id = record.get("instance_id")
        trial_idx = record.get("trial_idx", 0)
        if not instance_id:
            return
        full_response_dir.mkdir(parents=True, exist_ok=True)
        full_path = full_response_dir / f"{instance_id}_trial_{trial_idx}_full.jsonl"
        file_lock = full_response_lock or threading.Lock()
        payload = make_json_serializable(record)
        with file_lock:
            with full_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_usage_record(record: Dict[str, Any]) -> None:
        usage = record.get("token_usage") or {}
        entry = {
            "instance_id": record.get("instance_id"),
            "trial_idx": record.get("trial_idx", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get(
                "total_tokens",
                usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            ),
            "request_count": usage.get("request_count", 0),
            "prompt_cost_usd": usage.get("prompt_cost_usd", 0.0),
            "completion_cost_usd": usage.get("completion_cost_usd", 0.0),
            "total_cost_usd": usage.get(
                "total_cost_usd",
                usage.get("prompt_cost_usd", 0.0) + usage.get("completion_cost_usd", 0.0),
            ),
            "tool_calls": record.get("tool_call_count", 0),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with usage_lock:
            if usage_path.exists():
                try:
                    existing = json.loads(usage_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = []
            else:
                existing = []
            existing.append(entry)
            usage_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    existing_pairs: set[Tuple[str, int]] = set()
    for path in instance_output_dir.glob("*.jsonl"):
        parsed = _parse_instance_filename(path.name)
        if parsed:
            existing_pairs.add(parsed)

    if args.enable_eval:
        eval_runs = 0
        for instance_id, trial_idx in existing_pairs:
            agent_result = None
            instance_path = instance_output_dir / f"{instance_id}_trial_{trial_idx}.jsonl"
            try:
                raw_line = instance_path.read_text(encoding="utf-8").strip()
                if raw_line:
                    agent_result = json.loads(raw_line).get("agent_result")
            except Exception as exc:
                logger.warning("Failed to read %s: %s", instance_path, exc)

            if run_eval(args, instance_id, agent_result, trial_idx, instance_output_dir):
                eval_runs += 1
        if eval_runs:
            logger.info("Evaluated %d existing records.", eval_runs)

    tasks_to_run: list[Tuple[Dict[str, Any], int]] = []
    for entry in data:
        instance_id = entry.get("instance_id")
        if not instance_id:
            continue
        for trial_idx in range(args.trial_num):
            key = (instance_id, trial_idx)
            if key in existing_pairs:
                continue
            tasks_to_run.append((entry, trial_idx))

    logger.info(
        "Total tasks: %d, Trials each: %d, Completed combos: %d, Remaining combos: %d",
        len(data),
        args.trial_num,
        len(existing_pairs),
        len(tasks_to_run),
    )

    if not tasks_to_run:
        logger.info("No pending combinations. All done.")

    manage_kwargs = {
        "mapreduce_batch_size": args.mapreduce_batch_size,
        "mapreduce_max_retries": args.mapreduce_max_retries,
        "mapreduce_workers": args.mapreduce_workers,
        "mapreduce_max_steps": args.mapreduce_max_steps or args.max_steps,
        "mapreduce_insight_topk": args.mapreduce_insight_topk,
        "inspector_text_limit": args.inspector_text_limit,
        "inspector_audio_limit": args.inspector_audio_limit,
        "mapreduce_plan_mode": args.mapreduce_plan_mode,
        "expmemory": expmemory,
    }

    results = []
    file_lock = threading.Lock()

    def safe_write(result: Dict[str, Any]) -> None:
        payload = dict(result)
        payload.pop("eval_result", None)
        payload.pop("agent_trajectory_full", None)
        if not store_full_trace:
            payload = _prune_result_for_storage(payload, keep_until_mapreduce=trace_until_mapreduce)
        with file_lock:
            write_instance_file(payload)
            append_usage_record(payload)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        lower = max(1, args.summary_interval - 1)
        upper = max(lower, args.summary_interval + 1)
        summary_interval = random.randint(lower, upper)

        futures = [
            executor.submit(
                process_trial,
                item,
                model_kwargs,
                summary_interval,
                args.prompts_type,
                args.max_steps,
                manage_kwargs,
                trial_idx,
                capture_full_trace=bool(full_response_dir),
            )
            for item, trial_idx in tasks_to_run
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if not result:
                continue
            instance_id = result.get("instance_id")
            trial_idx = result.get("trial_idx", 0)
            run_eval(
                args,
                instance_id,
                result.get("agent_result"),
                trial_idx=trial_idx,
                instance_dir=instance_output_dir,
            )
            eval_payload = None
            eval_json_path = Path(args.eval_result_dir) / f"{instance_id}_trial_{trial_idx}_eval_result.json"
            if eval_json_path.exists():
                try:
                    eval_payload = json.loads(eval_json_path.read_text(encoding="utf-8"))
                except Exception:
                    eval_payload = None

            def _compute_label(result_dict: Dict[str, Any], eval_payload: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
                base_label = bool(result_dict.get("agent_result"))
                metrics: Dict[str, Any] = {}
                score = eval_payload.get("score") if isinstance(eval_payload, dict) else None
                f1_by_item = eval_payload.get("f1_by_item") if isinstance(eval_payload, dict) else None
                f1_by_row = eval_payload.get("f1_by_row") if isinstance(eval_payload, dict) else None
                precision_by_row = eval_payload.get("precision_by_row") if isinstance(eval_payload, dict) else None
                if score is not None:
                    metrics["eval_score"] = score
                if f1_by_item is not None:
                    metrics["f1_by_item"] = f1_by_item
                if f1_by_row is not None:
                    metrics["f1_by_row"] = f1_by_row
                if precision_by_row is not None:
                    metrics["precision_by_row"] = precision_by_row
                confidence_score = None
                if score is not None or f1_by_row is not None or f1_by_item is not None:
                    confidence_score = (
                        (score if score is not None else 0.0) * 0.6
                        + (f1_by_row if f1_by_row is not None else 0.0) * 0.3
                        + (f1_by_item if f1_by_item is not None else 0.0) * 0.1
                    )
                    metrics["confidence_score"] = confidence_score
                score_ok = (score is not None) and (score >= args.score_threshold)
                f1_row_ok = (f1_by_row is not None) and (f1_by_row >= 0.7)
                f1_combined_ok = (
                    (f1_by_row is not None)
                    and (f1_by_item is not None)
                    and (f1_by_row >= 0.45)
                    and (f1_by_item >= 0.7)
                )
                passed = bool(score_ok or f1_row_ok or f1_combined_ok)
                return passed, metrics

            label, metrics = _compute_label(result, eval_payload)
            if isinstance(result, dict):
                result["label"] = label
                if metrics:
                    result["eval_metrics"] = metrics

            if expmemory is not None and instance_id:
                try:
                    extra_fields: Dict[str, Any] = {}
                    exec_inputs = result.get("mapreduce_execute_inputs") if isinstance(result, dict) else None
                    if isinstance(exec_inputs, dict):
                        extra_fields["mapreduce_execute_inputs"] = exec_inputs
                        if exec_inputs.get("pattern_info") is not None:
                            extra_fields["pattern_info"] = exec_inputs.get("pattern_info")
                    if metrics:
                        extra_fields.update(metrics)
                    mas_message = MASMessage(
                        task_main=result.get("question") or instance_id,
                        task_description=result.get("question"),
                        label=label,
                    )
                    extra_fields["instance_id"] = instance_id
                    extra_fields["trial_idx"] = trial_idx
                    extra_fields["mapreduce_call_stage"] = "execute_first"
                    mas_message.extra_fields.update(extra_fields)
                    expmemory.add_memory(mas_message)
                except Exception as exc:
                    logger.warning("Failed to add memory for %s: %s", instance_id, exc)
            results.append(result)
            existing_pairs.add((instance_id, trial_idx))
            write_full_response(result)
            safe_write(result)

    logger.info(
        "Processing completed. Newly added: %d, Total completed combos: %d",
        len(results),
        len(existing_pairs),
    )

    
    if expmemory is not None:
        try:
            expmemory.insights_layer.clear_insights()
            expmemory.insights_layer._index_done()
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Failed to cleanup insights after all tasks: %s", exc)

    if args.enable_eval:
        eval_dir = Path(args.eval_result_dir).resolve()
        summary_path = eval_dir / "summary.json"
        calc_summary_results(
            eval_dir,
            args.trial_num,
            summary_path,
            allowed_instances=selected_instance_ids or None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data generation script")
    parser.add_argument("--infile", type=str, default="./data/Widesearch/widesearch.jsonl", help="input path")
    parser.add_argument("--outfile", type=str, default="./output_widesearch/instances", help="output directory for per-trial records")
    parser.add_argument("--outdir", dest="outfile", type=str, help="Alias for --outfile")
    parser.add_argument("--summary_interval", type=int, default=8, help="Summary interval")
    parser.add_argument("--prompts_type", type=str, default="default", help="Type of prompts to use")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrency")
    parser.add_argument("--max_steps", type=int, default=40, help="Maximum number of steps")
    parser.add_argument("--mapreduce_batch_size", type=int, default=None, help="Rows per batch when calling mapreducetool (omit for auto)")
    parser.add_argument("--mapreduce_max_retries", type=int, default=2, help="Retries per batch for mapreduce execution")
    parser.add_argument("--mapreduce_workers", type=int, default=8, help="Thread pool size for mapreduce batches")
    parser.add_argument("--mapreduce_max_steps", type=int, default=40, help="Max steps for sub-agents spawned by mapreducetool")
    parser.add_argument("--mapreduce_insight_topk", type=int, default=3, help="Number of memory hints to include in mapreduce plan stage")
    parser.add_argument("--inspector_text_limit", type=int, default=100000, help="Token/char limit for text inspector tool")
    parser.add_argument("--inspector_audio_limit", type=int, default=100000, help="Token/char limit for audio inspector tool")
    parser.add_argument("--store_full_trace", default="True", help="Store full agent trajectory/messages in primary output")
    parser.add_argument("--trace_until_mapreduce", action="store_true", help="When pruning, keep trajectory up to first mapreducetool call")
    parser.add_argument("--selected_tasks", nargs="*", default=[10], help="Optional list of dataset indices to run")
    parser.add_argument("--enable_expmemory", action="store_true", help="Enable Expmemory self-evolution logging")
    parser.add_argument("--expmemory_workdir", type=str, default="./memory_store", help="Working directory for Expmemory persistence")
    parser.add_argument("--expmemory_hop", type=int, default=1, help="Hop count for Expmemory task graph expansion")
    parser.add_argument("--expmemory_start_insights_threshold", type=int, default=40, help="Minimum memory size before insights updates start")
    parser.add_argument("--expmemory_rounds_per_insights", type=int, default=101, help="Update insights every N new records")
    parser.add_argument("--expmemory_insights_point_num", type=int, default=5, help="Number of points sampled per insights update")
    parser.add_argument("--expmemory_merge_insights_interval", type=int, default=101, help="Merge insights every N records once threshold is reached")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Score threshold for label computation")
    parser.add_argument("--mapreduce_plan_mode", action="store_true", help="Enable mapreduce plan+execute two-phase mode")   
    parser.add_argument("--trial_num", type=int, default=3, help="Number of trials to run for each task")
    parser.add_argument("--enable_eval", type=str, default='True', help="Enable evaluation pipeline")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Evaluation dataset path (defaults to --infile)")
    parser.add_argument("--eval_answer_root", type=str, default="./data/Widesearch/widesearch_gold", help="Directory containing gold answers")
    parser.add_argument("--eval_result_dir", type=str, default="./output_widesearch/eval_results", help="Directory to write evaluation outputs")
    parser.add_argument("--eval_result_root", dest="eval_result_dir", type=str, help="Alias for --eval_result_dir")
    parser.add_argument("--eval_model_config_name", type=str, default="default_eval_config", help="Evaluation model config name")
    parser.add_argument("--expmemory_namespace", type=str, default="widesearch", help="Namespace for Expmemory persistence")
    parser.add_argument("--full_response_dir", type=str, default="./output_widesearch/full_responses", help="Optional directory to dump full response records (including full trajectory)")

    main(parser.parse_args())
