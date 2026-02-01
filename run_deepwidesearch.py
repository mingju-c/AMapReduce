#!/usr/bin/env python
# coding=utf-8
import os
import json
import argparse
import logging
import threading
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import dataclasses
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from FlashOAgents import OpenAIServerModel
from base_agent import ManageAgent
from utils import read_jsonl
from typing import Any, Dict, Iterable, List, Optional
from loguru import logger as loguru_logger

# Make DeepWideSearch eval modules importable when running from repo root
WS_ROOT = Path(__file__).resolve().parent / "data" / "DeepWideSearch"
if str(WS_ROOT) not in sys.path:
    sys.path.append(str(WS_ROOT))

_EVAL_IMPORT_ERROR: Optional[Exception] = None
try:
    from eval.evaluation.data_loader import (  # type: ignore
        WideSearchDataLoaderHF,
        WideSearchResponseLoader,
    )
    from eval.evaluation.evaluation import evaluate_single_query  # type: ignore
except Exception as exc:  # pragma: no cover - runtime guard
    WideSearchDataLoaderHF = None  # type: ignore[assignment]
    WideSearchResponseLoader = None  # type: ignore[assignment]
    evaluate_single_query = None  # type: ignore[assignment]
    _EVAL_IMPORT_ERROR = exc

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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


def sanitize_instance_id(instance_id: str) -> str:
    return instance_id.replace("/", "_")


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


def _derive_instance_id(item: Dict[str, Any]) -> str:
    return (
        item.get("instance_id")
        or item.get("id")
        or f"instance_{hash(json.dumps(item, ensure_ascii=False))}"
    )


def _ensure_response_text(agent_result: Any) -> Optional[str]:
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

    total = _accumulate_from_steps(result.get("agent_trajectory", []))

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
    pruned.pop("mapreduce_task_records", None)
    pruned.pop("mapreduce_last_run_trace", None)
    # Keep the last assistant message as compact response context
    prediction = pruned.get("prediction") or _ensure_response_text(pruned.get("agent_result")) or ""
    pruned["messages"] = [{"role": "assistant", "content": prediction}] if prediction else []
    return pruned


def _format_selected_tag(selected_tasks: Optional[Iterable[str]], max_items: int = 6) -> str:
    if not selected_tasks:
        return "all"
    items = [str(item) for item in selected_tasks]
    if len(items) <= max_items:
        return "selected_" + "_".join(items)
    return f"selected_{len(items)}items"


def _summarize_eval_results(
    records: Iterable[Dict[str, Any]],
    eval_result_model_dir: Path,
    trial_num: int,
    selected_tasks: Optional[Iterable[str]],
    summary_result_path: Path,
) -> Optional[Dict[str, Any]]:
    if trial_num <= 0:
        logger.info("Summary skipped: trial_num=%s", trial_num)
        return None
    if not eval_result_model_dir.exists():
        logger.info("Summary skipped: eval result dir not found %s", eval_result_model_dir)
        return None

    metrics = [
        "score",
        "entity_acc",
        "search_tool_num",
        "visit_tool_num",
        "precision_by_row",
        "recall_by_row",
        "f1_by_row",
        "precision_by_item",
        "recall_by_item",
        "f1_by_item",
        "column_precision",
        "column_recall",
        "column_f1",
    ]

    all_results: Dict[str, List[Dict[str, float]]] = {m: [] for m in metrics}
    valid_instances: List[str] = []
    skipped_instances: List[str] = []

    for record in records:
        instance_id = _derive_instance_id(record)
        safe_id = sanitize_instance_id(instance_id)
        trial_metrics: Dict[str, List[float]] = {m: [] for m in metrics}
        missing = False

        for rid in range(1, trial_num + 1):
            eval_json_path = eval_result_model_dir / f"{safe_id}_{rid}_eval_result.json"
            if not eval_json_path.exists():
                missing = True
                break
            try:
                result = json.loads(eval_json_path.read_text(encoding="utf-8"))
            except Exception:
                missing = True
                break
            for m in metrics:
                if m in result:
                    trial_metrics[m].append(result[m])

        for m in metrics:
            if len(trial_metrics[m]) < trial_num:
                missing = True
                break

        if missing:
            skipped_instances.append(instance_id)
            continue

        valid_instances.append(instance_id)
        for m in metrics:
            values = trial_metrics[m]
            avg_n = float(sum(values) / len(values))
            max_n = float(max(values))
            min_n = float(min(values))
            all_results[m].append({"avg_n": avg_n, "max_n": max_n, "min_n": min_n})

    if not valid_instances:
        logger.warning("Summary skipped: no instances with complete eval results.")
        return None

    summary: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = all_results[m]
        if not vals:
            continue
        summary[m] = {
            "avg_n": float(sum(v["avg_n"] for v in vals) / len(vals)),
            "max_n": float(sum(v["max_n"] for v in vals) / len(vals)),
            "min_n": float(sum(v["min_n"] for v in vals) / len(vals)),
        }

    payload = {
        "selected_tasks": list(selected_tasks) if selected_tasks else [],
        "trial_num": trial_num,
        "instance_count": len(valid_instances),
        "skipped_instances": skipped_instances,
        "summary": summary,
    }

    summary_result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Summary saved to %s", summary_result_path)
    return payload


def extract_messages(agent_fn):
    """Convert agent messages into DeepWideSearch friendly structure."""
    msgs = []

    try:
        raw_messages = agent_fn.write_memory_to_messages(summary_mode=False)
    except Exception:
        return [{"role": "system", "content": "You are a web search agent."}]

    for msg in raw_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ["system"]:
            role = "system"
        elif role in ["user", "tool-response"]:
            role = "user"
        else:
            role = "assistant"

        if isinstance(content, list):
            merged = ""
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    merged += c.get("text", "")
                elif isinstance(c, str):
                    merged += c
            content = merged
        elif not isinstance(content, str):
            content = str(content)

        msgs.append({"role": role, "content": content})

    return msgs


def collect_usage_stats(model) -> Dict[str, float]:
    usage: Dict[str, float] = {}
    if hasattr(model, "get_usage"):
        try:
            usage = model.get_usage(reset=True) or {}
        except Exception as exc:
            logger.warning("Failed to fetch usage stats: %s", exc)
            usage = {}

    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    request_count = int(usage.get("request_count", 0))

    prompt_cost = float(usage.get("prompt_cost_usd", 0.0))
    completion_cost = float(usage.get("completion_cost_usd", 0.0))
    total_cost = float(
        usage.get("total_cost_usd", prompt_cost + completion_cost)
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "request_count": request_count,
        "prompt_cost_usd": prompt_cost,
        "completion_cost_usd": completion_cost,
        "total_cost_usd": total_cost,
    }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def run_single_infer(
    item,
    model_kwargs,
    summary_interval,
    prompts_type,
    max_steps,
    rollout_id,
    manage_kwargs,
):
    instance_id = _derive_instance_id(item)
    question = item.get("question") or item.get("query") or ""

    effective_prompts_type = "default"

    model = OpenAIServerModel(**model_kwargs)
    if hasattr(model, "reset_usage"):
        try:
            model.reset_usage()
        except Exception:
            pass

    agent = ManageAgent(
        model,
        summary_interval=summary_interval,
        prompts_type=effective_prompts_type,
        max_steps=max_steps,
        **manage_kwargs,
    )

    try:
        reasoning_start_time = datetime.utcnow().isoformat() + "Z"
        result = agent(question) or {}
        reasoning_end_time = datetime.utcnow().isoformat() + "Z"
    except Exception as e:
        logger.error(f"Error calling agent for {instance_id}: {e}")
        return None

    prediction = _ensure_response_text(result.get("agent_result")) or ""
    messages = extract_messages(agent.agent_fn)
    if prediction:
        messages.append({"role": "assistant", "content": prediction})

    usage_stats = collect_usage_stats(model)
    safe_mr_params = {k: v for k, v in manage_kwargs.items() if k != "expmemory"}

    output = {
        "instance_id": instance_id,
        "question": question,
        "prediction": prediction,
        "messages": messages,
        "rollout_id": rollout_id,
        "token_usage": usage_stats,
        "reasoning_start_time": reasoning_start_time,
        "reasoning_end_time": reasoning_end_time,
        "tool_call_count": _count_tool_calls(result),
        "mapreduce_params": safe_mr_params,
    }
    output.update(result)
    return output


def append_jsonl(path: Path, record: dict, lock: threading.Lock):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def dump_eval_response(eval_dir: Optional[Path], record: dict, lock: Optional[threading.Lock]):
    if not eval_dir:
        return
    eval_dir.mkdir(parents=True, exist_ok=True)
    safe_id = sanitize_instance_id(record["instance_id"])
    response_path = eval_dir / f"{safe_id}_{record['rollout_id']}_response.jsonl"
    payload = {
        "instance_id": record["instance_id"],
        "response": record["prediction"],
        "messages": record["messages"],
        "trial_idx": record["rollout_id"],
    }
    file_lock = lock or threading.Lock()
    with file_lock:
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def dump_full_response(full_dir: Optional[Path], record: dict, lock: Optional[threading.Lock]):
    """Optionally dump the full record (with trajectory) to a separate directory."""
    if not full_dir:
        return
    full_dir.mkdir(parents=True, exist_ok=True)
    safe_id = sanitize_instance_id(record.get("instance_id", "unknown"))
    rollout = record.get("rollout_id") or record.get("trial_idx") or 0
    response_path = full_dir / f"{safe_id}_{rollout}_response_full.jsonl"
    payload = dict(record)
    # Ensure start/end time fields are present in the full dump
    if "reasoning_start_time" not in payload:
        payload["reasoning_start_time"] = None
    if "reasoning_end_time" not in payload:
        payload["reasoning_end_time"] = None
    if payload.get("reasoning_start_time") and payload.get("reasoning_end_time"):
        try:
            start_dt = datetime.fromisoformat(payload["reasoning_start_time"].replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(payload["reasoning_end_time"].replace("Z", "+00:00"))
            payload["reasoning_duration_seconds"] = max((end_dt - start_dt).total_seconds(), 0)
        except Exception:
            pass
    file_lock = lock or threading.Lock()
    with file_lock:
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def export_eval_query_file(records: Iterable[dict], output_path: Optional[Path]):
    if not output_path:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in records:
            processed = {
                "instance_id": item.get("instance_id", ""),
                "query": item.get("question") or item.get("query") or "",
                "evaluation": item.get("evaluation", ""),
                "language": item.get("language", ""),
                "entity": item.get("entity", ""),
                "topic": item.get("topic", ""),
            }
            handle.write(json.dumps(processed, ensure_ascii=False) + "\n")


def main(args):
    input_path = Path(args.infile)
    output_root = Path(args.outdir)
    model_dir = output_root / args.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    def _resolve_response_dir(root: Optional[str], model_name: str) -> Optional[Path]:
        if not root:
            return None
        base = Path(root).expanduser()
        candidate = base / model_name
        # If base already ends with model_name or candidate exists, use candidate; else use base.
        if base.name == model_name or candidate.exists():
            return candidate
        return base

    eval_response_dir = _resolve_response_dir(args.eval_response_root, args.model_name)
    inline_eval_enabled = _coerce_bool(args.inline_eval)
    if inline_eval_enabled and _EVAL_IMPORT_ERROR is not None:
        logger.warning(
            "Inline eval disabled: failed to import DeepWideSearch eval modules from %s (%s)",
            WS_ROOT,
            _EVAL_IMPORT_ERROR,
        )
        inline_eval_enabled = False
    eval_loader = None
    eval_result_root: Optional[Path] = None
    eval_result_model_dir: Optional[Path] = None
    inline_eval_lock = threading.Lock()
    full_response_dir = Path(args.full_response_dir).expanduser() if args.full_response_dir else None
    if full_response_dir and eval_response_dir:
        try:
            if full_response_dir.resolve() == eval_response_dir.resolve():
                full_response_dir = eval_response_dir / "full_responses"
                logger.warning(
                    "full_response_dir matches eval_response_root; writing full responses under %s",
                    full_response_dir,
                )
        except Exception:
            pass
    full_response_lock = threading.Lock() if full_response_dir else None

    if inline_eval_enabled:
        # Quiet loguru debug logs from eval utils
        try:
            loguru_logger.remove()
        except Exception:
            pass
        loguru_logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "INFO"))
        if not eval_response_dir:
            raise ValueError("inline_eval requires eval_response_root to be set.")
        if not args.eval_answer_root:
            raise ValueError("inline_eval requires --eval_answer_root.")
        eval_response_dir.mkdir(parents=True, exist_ok=True)
        eval_result_root = (
            Path(args.eval_result_root).expanduser()
            if args.eval_result_root
            else eval_response_dir.parent / "eval_results"
        )
        eval_result_model_dir = eval_result_root / args.model_name
        eval_result_model_dir.mkdir(parents=True, exist_ok=True)
        try:
            eval_loader = WideSearchDataLoaderHF(
                query_path=args.infile,
                answer_root=args.eval_answer_root,
            )
            logger.info(
                "Inline evaluation enabled using query_path=%s, answer_root=%s",
                args.infile,
                args.eval_answer_root,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            inline_eval_enabled = False
            logger.warning("Failed to initialize inline evaluator: %s", exc)

    data = read_jsonl(input_path)
    data = _filter_selected_tasks(data, args.selected_tasks)
    if args.eval_query_output:
        export_eval_query_file(data, Path(args.eval_query_output))

    default_model = os.environ.get("DEFAULT_MODEL")
    if not default_model:
        raise ValueError("DEFAULT_MODEL environment variable is required.")

    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    model_kwargs = {
        "model_id": default_model,
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 32768,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"),
        "temperature": 0.0,
        "top_p": 1.0,
    }
    manage_kwargs = {
        "mapreduce_batch_size": args.mapreduce_batch_size,
        "mapreduce_max_retries": args.mapreduce_max_retries,
        "mapreduce_workers": args.mapreduce_workers,
        "mapreduce_max_steps": args.mapreduce_max_steps or args.max_steps,
        "mapreduce_insight_topk": args.mapreduce_insight_topk,
        "inspector_text_limit": args.inspector_text_limit,
        "inspector_audio_limit": args.inspector_audio_limit,
        "mapreduce_plan_mode": args.mapreduce_plan_mode,
        "expmemory": None,
    }
    store_full_trace = _coerce_bool(args.store_full_trace)
    trace_until_mapreduce = args.trace_until_mapreduce

    expmemory = None
    mas_message_cls = None
    enable_expmemory = bool(getattr(args, "enable_expmemory", False))
    expmemory_workdir = args.expmemory_workdir
    if enable_expmemory:
        try:
            os.environ.setdefault("OPENAI_API_BASE", os.environ.get("OPENAI_API_BASE") or "")
            os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "")
            from mas.utils import EmbeddingFunc  # lazy import to avoid env issues
            from mas.memory.mas_memory.Expmemory import Expmemory
            from mas.memory.common import MASMessage

            memory_model = OpenAIServerModel(**model_kwargs)
            embedding_func = EmbeddingFunc()

            expmemory = Expmemory(
                namespace=args.expmemory_namespace,
                global_config={
                    "working_dir": expmemory_workdir,
                    "hop": args.expmemory_hop,
                    "start_insights_threshold": args.expmemory_start_insights_threshold,
                    "rounds_per_insights": args.expmemory_rounds_per_insights,
                    "insights_point_num": args.expmemory_insights_point_num,
                    "merge_insights_interval": args.expmemory_merge_insights_interval,
                },
                llm_model=MemoryLLMWrapper(memory_model),
                embedding_func=embedding_func,
            )
            mas_message_cls = MASMessage
        except Exception as exc:
            logger.warning("Failed to initialize Expmemory, continuing without it: %s", exc)
            expmemory = None
            mas_message_cls = None

    manage_kwargs["expmemory"] = expmemory

    lock = threading.Lock()
    eval_lock = threading.Lock() if eval_response_dir else None
    usage_lock = threading.Lock()
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
        "prompt_cost_usd": 0.0,
        "completion_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "tool_calls": 0,
    }
    usage_seen: Dict[tuple[str, int], Dict[str, float]] = {}
    usage_log_path = model_dir / "token_usage_log.jsonl"

    def append_usage_record(record: dict):
        usage = record.get("token_usage") or {}
        entry = {
            "instance_id": record.get("instance_id"),
            "rollout_id": record.get("rollout_id"),
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
            usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            existing: list[Dict[str, Any]] = []
            old_entry: Optional[Dict[str, Any]] = None
            if usage_log_path.exists():
                try:
                    with open(usage_log_path, "r", encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                obj = json.loads(line.strip())
                            except Exception:
                                continue
                            if (
                                obj.get("instance_id") == entry["instance_id"]
                                and obj.get("rollout_id") == entry["rollout_id"]
                            ):
                                old_entry = obj
                                continue
                            existing.append(obj)
                except Exception:
                    existing = []
            with open(usage_log_path, "w", encoding="utf-8") as handle:
                for obj in existing:
                    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

            key = (entry["instance_id"], entry["rollout_id"])
            prev = usage_seen.get(key)
            delta = {
                "prompt_tokens": entry["prompt_tokens"] - (prev.get("prompt_tokens", 0) if prev else 0),
                "completion_tokens": entry["completion_tokens"] - (prev.get("completion_tokens", 0) if prev else 0),
                "total_tokens": entry["total_tokens"] - (prev.get("total_tokens", 0) if prev else 0),
                "request_count": entry["request_count"] - (prev.get("request_count", 0) if prev else 0),
                "prompt_cost_usd": entry["prompt_cost_usd"] - (prev.get("prompt_cost_usd", 0.0) if prev else 0.0),
                "completion_cost_usd": entry["completion_cost_usd"] - (prev.get("completion_cost_usd", 0.0) if prev else 0.0),
                "total_cost_usd": entry["total_cost_usd"] - (prev.get("total_cost_usd", 0.0) if prev else 0.0),
                "tool_calls": entry["tool_calls"] - (prev.get("tool_calls", 0) if prev else 0),
            }
            usage_totals["prompt_tokens"] += delta["prompt_tokens"]
            usage_totals["completion_tokens"] += delta["completion_tokens"]
            usage_totals["total_tokens"] += delta["total_tokens"]
            usage_totals["request_count"] += delta["request_count"]
            usage_totals["prompt_cost_usd"] += delta["prompt_cost_usd"]
            usage_totals["completion_cost_usd"] += delta["completion_cost_usd"]
            usage_totals["total_cost_usd"] += delta["total_cost_usd"]
            usage_totals["tool_calls"] += delta["tool_calls"]
            usage_seen[key] = entry

    def _compute_label(result_dict: Dict[str, Any], eval_payload: Optional[Dict[str, Any]]) -> tuple[bool, Dict[str, Any]]:
        metrics: Dict[str, Any] = {}
        score = eval_payload.get("score") if isinstance(eval_payload, dict) else None
        f1_by_item = eval_payload.get("f1_by_item") if isinstance(eval_payload, dict) else None
        f1_by_row = eval_payload.get("f1_by_row") if isinstance(eval_payload, dict) else None
        column_f1 = eval_payload.get("column_f1") if isinstance(eval_payload, dict) else None
        if score is not None:
            metrics["score"] = score
        if f1_by_item is not None:
            metrics["f1_by_item"] = f1_by_item
        if f1_by_row is not None:
            metrics["f1_by_row"] = f1_by_row
        if column_f1 is not None:
            metrics["column_f1"] = column_f1
        confidence_score = None
        if score is not None or f1_by_row is not None or f1_by_item is not None:
            confidence_score = (
                (score if score is not None else 0.0) * 0.4
                + (f1_by_row if f1_by_row is not None else 0.0) * 0.3
                + (f1_by_item if f1_by_item is not None else 0.0) * 0.2
                + (column_f1 if column_f1 is not None else 0.0) * 0.1
            )
            metrics["confidence_score"] = confidence_score
        score_ok = (score is not None) and (score >= args.score_threshold)
        f1_row_ok = (f1_by_row is not None) and (f1_by_row >= 0.5)
        f1_combined_ok = False
        f1_combined_ok = f1_combined_ok or (
            (f1_by_row is not None)
            and (f1_by_item is not None)
            and (column_f1 is not None)
            and (f1_by_row >= 0.35)
            and (f1_by_item >= 0.55)
            and (column_f1 >= 0.65)
        )
        passed = bool(score_ok or f1_row_ok or f1_combined_ok)
        return passed, metrics

    def run_inline_eval(record: dict):
        if not inline_eval_enabled or eval_loader is None or eval_result_model_dir is None or not eval_response_dir:
            return
        instance_id = record.get("instance_id")
        rollout_id = record.get("rollout_id")
        if instance_id is None or rollout_id is None:
            return
        safe_id = sanitize_instance_id(str(instance_id))
        response_path = eval_response_dir / f"{safe_id}_{rollout_id}_response.jsonl"
        if not response_path.exists():
            logger.warning("Inline eval skipped: response file missing %s", response_path)
            return
        try:
            query = eval_loader.load_query_by_instance_id(instance_id)
            responses = WideSearchResponseLoader.load_response(str(response_path))
            if not responses:
                logger.warning("Inline eval skipped: no responses in %s", response_path)
                return
            result_save_path = eval_result_model_dir / f"{safe_id}_{rollout_id}_eval_result.csv"
            result_save_path.parent.mkdir(parents=True, exist_ok=True)
            eval_result = evaluate_single_query(
                query,
                responses[0],
                str(result_save_path),
                args.eval_model_config_name,
            )
            eval_json_path = result_save_path.with_suffix(".json")
            with inline_eval_lock:
                with open(eval_json_path, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(dataclasses.asdict(eval_result), ensure_ascii=False, indent=2))
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Inline eval failed for %s_%s: %s", instance_id, rollout_id, exc)

    def add_memory_record(record: dict):
        if not expmemory or not mas_message_cls:
            return
        if not isinstance(record, dict):
            return
        instance_id = record.get("instance_id")
        if not instance_id:
            return
        try:
            question = record.get("question") or instance_id
            label = bool(record.get("label"))
            extras: Dict[str, Any] = {}
            exec_inputs = record.get("mapreduce_execute_inputs")
            # If mapreduce_execute_inputs missing, try to recover from task_records
            if not isinstance(exec_inputs, dict):
                task_records = record.get("mapreduce_task_records") or []
                if isinstance(task_records, list):
                    for rec in task_records:
                        if isinstance(rec, dict) and rec.get("call_stage") == "execute":
                            exec_inputs = {
                                "num_rows": rec.get("total_rows"),
                                "num_cols": rec.get("num_cols"),
                                "task_matrix": rec.get("task_matrix"),
                                "template": rec.get("template") or rec.get("template_preview"),
                                "json_schema": rec.get("json_schema"),
                                "json_schema_keys": rec.get("json_schema_keys"),
                                "batch_strategy": rec.get("batch_strategy"),
                                "batch_size": rec.get("batch_size"),
                                "pattern_info": rec.get("pattern_info"),
                                "memory_block": rec.get("memory_block"),
                            }
                            break
            if isinstance(exec_inputs, dict):
                extras["mapreduce_execute_inputs"] = exec_inputs
                if exec_inputs.get("pattern_info") is not None:
                    extras["pattern_info"] = exec_inputs.get("pattern_info")
                if exec_inputs.get("memory_block") is not None:
                    extras["memory_block"] = exec_inputs.get("memory_block")
            eval_metrics = record.get("eval_metrics")
            if isinstance(eval_metrics, dict):
                extras["eval_metrics"] = eval_metrics
                if "confidence_score" in eval_metrics:
                    extras["confidence_score"] = eval_metrics.get("confidence_score")
            extras["instance_id"] = instance_id
            extras["rollout_id"] = record.get("rollout_id")
            extras["trial_idx"] = record.get("rollout_id")
            extras["mapreduce_call_stage"] = "execute_first"
            mas_message = mas_message_cls(
                task_main=question,
                task_description=question,
                label=label,
            )
            mas_message.extra_fields.update(extras)
            expmemory.add_memory(mas_message)
        except Exception as exc:
            logger.warning("Failed to add memory for %s: %s", instance_id, exc)

    logger.info(f"Loaded {len(data)} DeepWideSearch items.")

    rollout_ids = list(range(1, args.trial_num + 1))
    existing_pairs: set[tuple[str, int]] = set()
    if eval_response_dir and eval_response_dir.exists():
        try:
            for resp_path in eval_response_dir.glob("*_response.jsonl"):
                try:
                    with resp_path.open("r", encoding="utf-8") as handle:
                        line = handle.readline().strip()
                        if not line:
                            continue
                        record = json.loads(line)
                except Exception:
                    continue
                iid = record.get("instance_id")
                rid = record.get("trial_idx")
                if iid and isinstance(rid, int):
                    existing_pairs.add((iid, rid))
        except Exception as exc:
            logger.warning("Failed to scan eval responses %s: %s", eval_response_dir, exc)

    # Pre-run eval for existing responses that lack eval files
    if inline_eval_enabled and eval_response_dir and eval_response_dir.exists() and eval_result_model_dir:
        for item in data:
            instance_id = _derive_instance_id(item)
            for rid in rollout_ids:
                safe_id = sanitize_instance_id(str(instance_id))
                resp_path = eval_response_dir / f"{safe_id}_{rid}_response.jsonl"
                eval_json_path = eval_result_model_dir / f"{safe_id}_{rid}_eval_result.json"
                if resp_path.exists() and not eval_json_path.exists():
                    logger.info("Inline eval (pre-scan) for existing response %s (rollout %s)", instance_id, rid)
                    run_inline_eval({"instance_id": instance_id, "rollout_id": rid})

    tasks_to_run: list[tuple[Dict[str, Any], int]] = []
    for item in data:
        instance_id = _derive_instance_id(item)
        for rid in rollout_ids:
            if (instance_id, rid) in existing_pairs:
                continue
            tasks_to_run.append((item, rid))

    total_jobs = len(tasks_to_run)
    if total_jobs == 0:
        logger.warning(
            "No tasks to run (empty dataset or zero rollouts or all completed). Existing combos: %d",
            len(existing_pairs),
        )
        return
    logger.info(
        "Pending combos: %d, Already completed combos: %d",
        total_jobs,
        len(existing_pairs),
    )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_meta: Dict[Any, int] = {}
        for item, rollout_id in tasks_to_run:
            future = executor.submit(
                run_single_infer,
                item,
                model_kwargs,
                args.summary_interval,
                args.prompts_type,
                args.max_steps,
                rollout_id,
                manage_kwargs,
            )
            future_to_meta[future] = rollout_id

        for future in tqdm(as_completed(future_to_meta), total=total_jobs):
            rollout_id = future_to_meta[future]
            result = future.result()
            if result:
                # optional full dump (unpruned)
                dump_full_response(full_response_dir, result, full_response_lock)

                eval_payload = None
                stored_result = dict(result)
                if not store_full_trace:
                    stored_result = _prune_result_for_storage(stored_result, keep_until_mapreduce=trace_until_mapreduce)
                dump_eval_response(eval_response_dir, stored_result, eval_lock)
                run_inline_eval(stored_result)
                if inline_eval_enabled and eval_result_model_dir is not None:
                    safe_id = sanitize_instance_id(str(result.get("instance_id", "")))
                    eval_json_path = eval_result_model_dir / f"{safe_id}_{rollout_id}_eval_result.json"
                    if eval_json_path.exists():
                        try:
                            eval_payload = json.loads(eval_json_path.read_text(encoding="utf-8"))
                        except Exception:
                            eval_payload = None
                label, metrics = _compute_label(result, eval_payload)
                if isinstance(result, dict):
                    result["label"] = label
                    if metrics:
                        result["eval_metrics"] = metrics
                add_memory_record(result)
                append_usage_record(result)

    
    if expmemory is not None:
        try:
            expmemory.insights_layer.clear_insights()
            expmemory.insights_layer._index_done()
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Failed to cleanup insights after all tasks: %s", exc)

    summary_root = Path(args.eval_result_root).expanduser() if args.eval_result_root else None
    if summary_root is not None:
        eval_model_dir = eval_result_model_dir or (summary_root / args.model_name)
        selected_tag = _format_selected_tag(args.selected_tasks)
        summary_path = summary_root / f"{args.model_name}_trial_num_{args.trial_num}_{selected_tag}_summary.json"
        _summarize_eval_results(
            records=data,
            eval_result_model_dir=eval_model_dir,
            trial_num=args.trial_num,
            selected_tasks=args.selected_tasks,
            summary_result_path=summary_path,
        )

    logger.info(f"All rollouts completed. Total tasks: {total_jobs}")
    logger.info(
        "Token usage totals - prompt: %d, completion: %d, total: %d, requests: %d, tool_calls: %d, "
        "prompt_cost: %.4f USD, completion_cost: %.4f USD, total_cost: %.4f USD",
        usage_totals["prompt_tokens"],
        usage_totals["completion_tokens"],
        usage_totals["total_tokens"],
        usage_totals["request_count"],
        usage_totals["tool_calls"],
        usage_totals["prompt_cost_usd"],
        usage_totals["completion_cost_usd"],
        usage_totals["total_cost_usd"],
    )
    logger.info(f"Results saved under: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="data/DeepWideSearch/data/overall_20250916.jsonl", help="Input dataset path (json/jsonl)")
    parser.add_argument("--eval_data_path", dest="infile", type=str, help="Alias for --infile")
    parser.add_argument("--outdir", type=str, default="./output_deepwidesearch/instances", help="Output directory for per-trial records")
    parser.add_argument("--outfile", dest="outdir", type=str, help="Alias for --outdir")
    parser.add_argument("--model_name", type=str, default="AMapReduce", help="Model tag used for organizing DeepWideSearch outputs")
    parser.add_argument("--summary_interval", type=int, default=8, help="Summary / plan optimization interval")
    parser.add_argument("--prompts_type", type=str, default="default", help="Prompt preset type")
    parser.add_argument("--max_steps", type=int, default=40, help="Maximum agent steps")
    parser.add_argument("--mapreduce_batch_size", type=int, default=None, help="Rows per batch when calling mapreducetool (omit for auto)")
    parser.add_argument("--mapreduce_max_retries", type=int, default=2, help="Retries per batch for mapreduce execution")
    parser.add_argument("--mapreduce_workers", type=int, default=8, help="Thread pool size for mapreduce batches")
    parser.add_argument("--mapreduce_max_steps", type=int, default=40, help="Max steps for sub-agents spawned by mapreducetool")
    parser.add_argument("--mapreduce_insight_topk", type=int, default=3, help="Number of memory hints to include in mapreduce plan stage")
    parser.add_argument("--inspector_text_limit", type=int, default=100000, help="Token/char limit for text inspector tool")
    parser.add_argument("--inspector_audio_limit", type=int, default=100000, help="Token/char limit for audio inspector tool")
    parser.add_argument("--store_full_trace", default="True", help="Store full agent trajectory/messages (default: prune to final answer only)")
    parser.add_argument("--trace_until_mapreduce", action="store_true", help="When pruning, keep trajectory up to first mapreducetool call")
    parser.add_argument("--selected_tasks", nargs="*", default=[142], help="Optional list of dataset indices to run")
    parser.add_argument("--enable_expmemory", action="store_true", help="Enable Expmemory self-evolution logging")
    parser.add_argument("--expmemory_workdir", type=str, default="./memory_store", help="Working directory for Expmemory persistence")
    parser.add_argument("--expmemory_namespace", type=str, default="deepwidesearch", help="Namespace for Expmemory persistence")
    parser.add_argument("--expmemory_hop", type=int, default=1, help="Expmemory hop for query graph expansion")
    parser.add_argument("--expmemory_start_insights_threshold", type=int, default=45, help="Min records before insights training/merge")
    parser.add_argument("--expmemory_rounds_per_insights", type=int, default=45, help="Records per insights update round")
    parser.add_argument("--expmemory_insights_point_num", type=int, default=5, help="Samples per insights update round")
    parser.add_argument("--expmemory_merge_insights_interval", type=int, default=45, help="Records per insights merge")
    parser.add_argument("--mapreduce_plan_mode", action="store_true", default=False, help="Enable mapreduce plan+execute two-phase mode (default: disabled)")
    parser.add_argument("--trial_num", type=int, default=1, help="Number of trials to run for each task")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--eval_response_root", type=str, default="./output_deepwidesearch/eval_data/AMapReduce", help="If set, dump WideSearch-ready responses under this root.")
    parser.add_argument("--eval_query_output", type=str, default="./output_deepwidesearch/eval_data/AMapReduce/overall_20250916.jsonl", help="Optional path to dump query file with 'query' field for eval.")
    parser.add_argument("--inline_eval", default="True", help="Run WideSearch evaluation immediately after each rollout using stored responses.")
    parser.add_argument("--enable_eval", dest="inline_eval", action="store_true", help="Alias for --inline_eval True")
    parser.add_argument("--disable_eval", dest="inline_eval", action="store_false", help="Alias for --inline_eval False")
    parser.add_argument("--eval_answer_root", type=str, default="data/DeepWideSearch/data/overall_20250916_tables", help="Root directory containing ground-truth answers for inline evaluation (WideSearch format).")
    parser.add_argument("--eval_result_root", type=str, default="./output_deepwidesearch/eval_results", help="Directory to store inline eval outputs (CSV/JSON). Defaults to <eval_response_root>/../eval_results/<model_name>.")
    parser.add_argument("--eval_result_dir", dest="eval_result_root", type=str, help="Alias for --eval_result_root")
    parser.add_argument("--eval_model_config_name", type=str, default="default_eval_config", help="Model config name used by the WideSearch evaluator (LLM judge).")
    parser.add_argument("--full_response_dir", type=str, default="./output_deepwidesearch/instances", help="Optional directory to dump full response records (including trajectory).")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Score threshold for labeling success in inline evaluation.")

    args = parser.parse_args()

    main(args)
