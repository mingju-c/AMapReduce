# coding=utf-8
from __future__ import annotations

import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from logging import getLogger
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union
from .models import ChatMessage
from .tools import Tool
from .utils import AgentExecutionError, make_json_serializable
from .agents import ToolCallingAgent
import json
import re
from copy import deepcopy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque, OrderedDict
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Set, Tuple, TypedDict, Union
import yaml
from jinja2 import StrictUndefined, Template
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from .agent_types import AgentType, handle_agent_output_types
from .tools import FinalAnswerTool
from .models import (
    ChatMessage,
    MessageRole,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
)
from .tools import Tool
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    make_json_serializable,
    parse_json_tool_call,
)


logger = getLogger(__name__)


def get_variable_names(self, template: str) -> Set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

def parse_model_content(content: Union[str, dict]) -> dict:

    if isinstance(content, dict):
        return content
    elif isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"text": content}
    else:
        return {"unknown_type": str(content)}

class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        initial_plan (`str`): Initial plan prompt.
    """

    initial_plan: str

class SummaryPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        update_pre_messages (`str`): Progress execution prompt.
        update_post_messages (`str`): Progress execution prompt.
    """

    update_pre_messages: str
    update_post_messages: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        summary ([`~agents.SummaryPromptTemplate`]): Summary prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    summary: SummaryPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(initial_plan=""),
    summary=SummaryPromptTemplate(),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)

class MapReduceTool(Tool):
    name = "mapreducetool"
    description = (
        "Decompose a structured task matrix into independent sub-queries, let search-capable agents solve each "
        "sub-task, and aggregate the validated JSON objects into a JSONL payload."
    )
    inputs = {
        "task_matrix": {
            "description": (
                "List of task rows (M x N). Each row will be injected into the template using numbered placeholders."
            ),
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "template": {
            "description": "Template containing placeholders '__0__', '__1__', ... that will be replaced per row.",
            "type": "string",
        },
        "json_schema": {
            "description": "JSON schema describing the expected output object for each atomic task.",
            "type": "object",
        },
        "batch_size": {
            "description": (
                "Optional positive integer controlling how many task rows are processed together. "
                "If omitted, the tool will use its internal default."
            ),
            "type": "integer",
            "nullable": True,
        },
        "batch_strategy": {
            "description": (
                "Optional strategy object that controls how the task matrix is partitioned. "
                "Provide at least a 'type' field with value 'per_atom', 'by_attr', or 'open'. "
                "Additional metadata such as 'attribute_index', 'attribute_name', 'groups', 'chunk_size', "
                "or 'rationale' may be supplied to fine-tune batching."
            ),
            "type": "object",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        web_tools: Optional[List[Tool]] = None,
        *,
        prompts_type: str = "default",
        batch_size: Optional[int] = None,
        max_retries: int = 2,
        max_steps: int = 32,
        workers: int = 8,
        subagent_prompts_type: str = "searchagent",
        subagent_prompt_templates: PromptTemplates | None = None,
        plan_mode_enabled: bool = True,
        expmemory: Any = None,
        insight_topk: int = 3,
    ):
        super().__init__()
        self.model = model
        self.prompts_type = prompts_type
        self.default_batch_size = max(1, int(batch_size)) if batch_size else 1
        self.max_retries = max(1, int(max_retries))
        self.max_steps = max_steps
        self.workers = max(1, int(workers))
        self._logger = getLogger(f"{__name__}.MapReduceTool")
        self.web_tools = list(web_tools) if web_tools is not None else self._build_default_tools()
        self.task_records: List[Dict[str, Any]] = []
        self.last_run_trace: Optional[Dict[str, Any]] = None
        self.enumerated_entities: Set[str] = set()
        self.enumeration_lookup: Dict[str, str] = {}
        self.collected_entities: Set[str] = set()
        self.subagent_prompts_type = subagent_prompts_type
        self.subagent_prompt_templates = subagent_prompt_templates
        self.plan_mode_enabled = plan_mode_enabled
        self._plan_done = False
        self.expmemory = expmemory
        self.insight_topk = max(0, int(insight_topk)) if insight_topk is not None else 3
    def _build_default_tools(self) -> List[Tool]:
        tools: List[Tool] = []
        try:
            from .search_tools import CrawlPageTool, WebSearchTool

            tools.append(WebSearchTool())
            tools.append(CrawlPageTool(self.model))
        except Exception as exc:
            self._logger.warning("Failed to build default web tools: %s", exc)
        return tools

    @staticmethod
    def _render(template: str, row: List[Any]) -> str:
        sub_query = template
        for idx, field in enumerate(row):
            sub_query = sub_query.replace(f"__{idx}__", str(field))
        return sub_query

    @staticmethod
    def _format_batch_id(index: int) -> str:
        return f"batch-{index + 1:03d}"
    
    def _normalize_batch_strategy(
        self,
        strategy: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(strategy, Mapping):
            return {
                "type": "per_atom",
                "rationale": "Defaulted to per_atom because no batch_strategy was provided.",
                "auto_generated": True,
            }
        normalized = deepcopy({key: value for key, value in strategy.items()})
        normalized["auto_generated"] = False
        strategy_type = str(normalized.get("type", "per_atom")).lower()
        if strategy_type not in {"per_atom", "by_attr", "open"}:
            strategy_type = "per_atom"
        normalized["type"] = strategy_type
        if not normalized.get("rationale"):
            normalized["rationale"] = f"Using {strategy_type} batching."
        attr_index = normalized.get("attribute_index")
        if attr_index is not None:
            try:
                normalized["attribute_index"] = int(attr_index)
            except (TypeError, ValueError):
                normalized.pop("attribute_index", None)
        groups = normalized.get("groups")
        if strategy_type == "open":
            if isinstance(groups, Mapping):
                normalized["groups"] = list(groups.values())
            elif not isinstance(groups, list):
                normalized.pop("groups", None)
            chunk_size = normalized.get("chunk_size")
            if chunk_size is not None:
                try:
                    normalized["chunk_size"] = max(1, int(chunk_size))
                except (TypeError, ValueError):
                    normalized.pop("chunk_size", None)
        return normalized
    
    def _extract_attribute_value(
        self,
        row: Union[List[Any], Mapping[str, Any]],
        *,
        attribute_index: Optional[int],
        attribute_name: Optional[str],
    ) -> Any:
        if isinstance(row, Mapping):
            if attribute_name and attribute_name in row:
                return row.get(attribute_name)
            if attribute_index is not None:
                keys = list(row.keys())
                if 0 <= attribute_index < len(keys):
                    key = keys[attribute_index]
                    return row.get(key)
            return None
        if attribute_index is not None and isinstance(row, (list, tuple)):
            if 0 <= attribute_index < len(row):
                return row[attribute_index]
        return None

    @staticmethod
    def _coerce_index(candidate: Any, upper_bound: int) -> Optional[int]:
        try:
            idx = int(candidate)
        except (TypeError, ValueError):
            return None
        if idx < 0:
            idx = upper_bound + idx
        if 0 <= idx < upper_bound:
            return idx
        return None
    @staticmethod
    def _chunk_indices(indices: List[int], chunk_size: int) -> List[List[int]]:
        if chunk_size <= 0:
            chunk_size = 1
        return [indices[pos : pos + chunk_size] for pos in range(0, len(indices), chunk_size)]

    @staticmethod
    def _filter_none(payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in payload.items() if value is not None}

    def _build_default_chunk_plan(
        self,
        task_matrix: List[List[Any]],
        chunk_size: int,
        strategy: Mapping[str, Any],
        *,
        rationale: str,
        start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        indices = list(range(len(task_matrix)))
        plan: List[Dict[str, Any]] = []
        for block_offset, chunk in enumerate(self._chunk_indices(indices, max(1, chunk_size))):
            rows = [task_matrix[idx] for idx in chunk]
            batch_id = self._format_batch_id(start_index + block_offset)
            shared_context = {
                "source_indices": chunk,
                "size": len(rows),
                "note": f"default chunk of {len(rows)} rows",
            }
            manifest = {
                "batch_id": batch_id,
                "submatrix": deepcopy(rows),
                "shared_context": self._filter_none(shared_context),
                "strategy_used": deepcopy(strategy),
                "rationale": rationale,
            }
            plan.append({"batch_id": batch_id, "rows": rows, "manifest": manifest, "indices": chunk})
        return plan

    def _build_open_batches(
        self,
        task_matrix: List[List[Any]],
        strategy: Mapping[str, Any],
        fallback_batch_size: int,
    ) -> List[Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        used: Set[int] = set()
        base_rationale = strategy.get("rationale") or "Heuristic grouping supplied by manager agent."
        preferred_chunk_size = strategy.get("chunk_size")

        groups = strategy.get("groups")
        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, Mapping):
                    continue
                indices: List[int] = []
                explicit_indices = group.get("indices") or group.get("rows")
                if isinstance(explicit_indices, list):
                    for candidate in explicit_indices:
                        idx = self._coerce_index(candidate, len(task_matrix))
                        if idx is None or idx in used:
                            continue
                        indices.append(idx)
                        used.add(idx)
                if not indices and "attribute_value" in group:
                    attr_index = group.get("attribute_index", strategy.get("attribute_index"))
                    attr_name = group.get("attribute_name", strategy.get("attribute_name"))
                    target_value = group.get("attribute_value")
                    for idx, row in enumerate(task_matrix):
                        if idx in used:
                            continue
                        value = self._extract_attribute_value(
                            row,
                            attribute_index=attr_index,
                            attribute_name=attr_name,
                        )
                        if value == target_value:
                            indices.append(idx)
                            used.add(idx)
                if not indices:
                    continue
                indices.sort()
                rows = [task_matrix[idx] for idx in indices]
                batch_id = self._format_batch_id(len(plan))
                shared_context: Dict[str, Any] = {
                    "source_indices": indices,
                    "size": len(rows),
                }
                extra_context = group.get("shared_context")
                if isinstance(extra_context, Mapping):
                    shared_context.update(deepcopy(extra_context))
                snapshot = deepcopy(strategy)
                snapshot.update({k: v for k, v in group.items() if k not in {"shared_context"}})
                manifest = {
                    "batch_id": batch_id,
                    "submatrix": deepcopy(rows),
                    "shared_context": self._filter_none(shared_context),
                    "strategy_used": snapshot,
                    "rationale": group.get("rationale") or base_rationale,
                }
                plan.append({"batch_id": batch_id, "rows": rows, "manifest": manifest, "indices": indices})
        remaining = [idx for idx in range(len(task_matrix)) if idx not in used]
        effective_chunk_size = preferred_chunk_size or fallback_batch_size
        if remaining:
            start_index = len(plan)
            for offset, chunk in enumerate(self._chunk_indices(remaining, max(1, effective_chunk_size))):
                rows = [task_matrix[idx] for idx in chunk]
                batch_id = self._format_batch_id(start_index + offset)
                shared_context = {
                    "source_indices": chunk,
                    "size": len(rows),
                    "note": "fallback chunk for unassigned rows",
                }
                manifest = {
                    "batch_id": batch_id,
                    "submatrix": deepcopy(rows),
                    "shared_context": self._filter_none(shared_context),
                    "strategy_used": deepcopy(strategy),
                    "rationale": (
                        "Agent-defined chunk_size chunking."
                        if preferred_chunk_size
                        else "Fallback chunking for rows not covered by open strategy groups."
                    ),
                }
                plan.append({"batch_id": batch_id, "rows": rows, "manifest": manifest, "indices": chunk})
        return plan

    def _build_batch_plan(
        self,
        task_matrix: List[List[Any]],
        strategy: Dict[str, Any],
        fallback_batch_size: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        if not task_matrix:
            return plan, strategy

        strategy_type = strategy.get("type", "per_atom")
        base_rationale = strategy.get("rationale") or f"Using {strategy_type} batching."

        if strategy_type == "per_atom":
            if strategy.get("auto_generated") and fallback_batch_size > 1:
                plan = self._build_default_chunk_plan(
                    task_matrix,
                    fallback_batch_size,
                    strategy,
                    rationale="Default chunking derived from batch_size because no batch_strategy was provided.",
                )
            else:
                for idx, row in enumerate(task_matrix):
                    batch_id = self._format_batch_id(idx)
                    shared_context = {
                        "source_indices": [idx],
                        "size": 1,
                        "note": "single-row batch",
                    }
                    manifest = {
                        "batch_id": batch_id,
                        "submatrix": deepcopy([row]),
                        "shared_context": self._filter_none(shared_context),
                        "strategy_used": deepcopy(strategy),
                        "rationale": base_rationale,
                    }
                    plan.append({"batch_id": batch_id, "rows": [row], "manifest": manifest, "indices": [idx]})
            return plan, strategy

        if strategy_type == "by_attr":
            attr_index = strategy.get("attribute_index")
            attr_name = strategy.get("attribute_name")
            missing_label = strategy.get("missing_value_label", "undefined")
            group_rationales = strategy.get("group_rationales") if isinstance(strategy.get("group_rationales"), Mapping) else {}
            grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
            for idx, row in enumerate(task_matrix):
                attr_value = self._extract_attribute_value(
                    row,
                    attribute_index=attr_index,
                    attribute_name=attr_name,
                )
                key_object = attr_value if attr_value is not None else missing_label
                key = json.dumps(key_object, ensure_ascii=False) if isinstance(key_object, (dict, list)) else str(key_object)
                bucket = grouped.setdefault(key, {"value": attr_value, "rows": [], "indices": []})
                bucket["rows"].append(row)
                bucket["indices"].append(idx)
            for group_offset, (key, bucket) in enumerate(grouped.items()):
                rows = bucket["rows"]
                indices = bucket["indices"]
                batch_id = self._format_batch_id(group_offset)
                shared_context = self._filter_none(
                    {
                        "attribute_index": attr_index,
                        "attribute_name": attr_name,
                        "attribute_value": bucket["value"],
                        "source_indices": indices,
                        "size": len(rows),
                    }
                )
                rationale = group_rationales.get(key) or group_rationales.get(str(bucket["value"])) or base_rationale
                snapshot = deepcopy(strategy)
                snapshot.update({"attribute_value": bucket["value"]})
                manifest = {
                    "batch_id": batch_id,
                    "submatrix": deepcopy(rows),
                    "shared_context": shared_context,
                    "strategy_used": snapshot,
                    "rationale": rationale,
                }
                plan.append({"batch_id": batch_id, "rows": rows, "manifest": manifest, "indices": indices})
            if plan:
                return plan, strategy

        if strategy_type == "open":
            plan = self._build_open_batches(task_matrix, strategy, fallback_batch_size)
            if plan:
                return plan, strategy

        plan = self._build_default_chunk_plan(
            task_matrix,
            fallback_batch_size,
            strategy,
            rationale=base_rationale or "Default chunking fallback.",
        )
        return plan, strategy

    def _schema_keys(self, schema: Dict[str, Any]) -> List[str]:
        if not isinstance(schema, dict):
            return []
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return []
        keys = list(properties.keys())
        return keys

    def _build_pattern_info(
        self,
        task_matrix: List[List[Any]],
        template: str,
        json_schema: Dict[str, Any],
        batch_strategy: Mapping[str, Any],
    ) -> Dict[str, Any]:
        num_rows = len(task_matrix) if isinstance(task_matrix, list) else None
        num_cols = None
        if isinstance(task_matrix, list) and task_matrix:
            num_cols = max(len(r) for r in task_matrix if isinstance(r, list))
        schema_cols: List[str] = []
        if isinstance(json_schema, Mapping):
            props = json_schema.get("properties")
            if isinstance(props, Mapping):
                schema_cols = list(props.keys())
            elif isinstance(json_schema.get("columns"), list):
                schema_cols = list(json_schema.get("columns"))
        template_complexity = "high" if len(template) > 2000 else "medium" if len(template) > 800 else "low"
        strategy_type = batch_strategy.get("type") if isinstance(batch_strategy, Mapping) else None
        chunk_size = batch_strategy.get("chunk_size") if isinstance(batch_strategy, Mapping) else None
        base_rationale = batch_strategy.get("rationale") if isinstance(batch_strategy, Mapping) else None
        if not base_rationale and strategy_type:
            base_rationale = f"Using {strategy_type} batching."
        if strategy_type:
            if num_rows is not None:
                strat_info = (
                    f"Batch strategy [{strategy_type}] selected for {num_rows} rows"
                    f"{f' with chunk_size={chunk_size}' if chunk_size else ''}. "
                    f"Rationale: {base_rationale or 'not specified'}"
                )
            else:
                strat_info = (
                    f"Batch strategy [{strategy_type}] selected"
                    f"{f' with chunk_size={chunk_size}' if chunk_size else ''}. "
                    f"Rationale: {base_rationale or 'not specified'}"
                )
        else:
            strat_info = base_rationale
        pattern_info = {
            "task_matrix": {
                "num_rows": num_rows,
                "num_cols": num_cols,
            },
            "template": {
                "content": template,
            },
            "json_schema": {
                "columns": schema_cols,
            },
            "batching_strategy": {
                "type": batch_strategy.get("type") if isinstance(batch_strategy, Mapping) else None,
                "chunk_size": batch_strategy.get("chunk_size") if isinstance(batch_strategy, Mapping) else None,
                "rationale": batch_strategy.get("rationale") if isinstance(batch_strategy, Mapping) else None,
                "strategy_info": strat_info,
            },
        }

        # Add a lightweight natural-language summary for readability and few-shot reuse.
        summary_lines = []
        summary_lines.append(
            f"- Task matrix: {num_rows} rows x {num_cols} cols."
        )
        summary_lines.append(
            f"- Template: preview: {str(template)[:200]}"
        )
        summary_lines.append(
            f"- Output schema: {', '.join(schema_cols) if schema_cols else 'unknown'}; fixed column set."
        )
        bs_type = pattern_info["batching_strategy"]["type"]
        bs_chunk = pattern_info["batching_strategy"]["chunk_size"]
        bs_reason = pattern_info["batching_strategy"]["rationale"] or strat_info
        summary_lines.append(
            f"- Batching strategy: type={bs_type}, chunk_size={bs_chunk or 'auto'}; rationale: {bs_reason}"
        )
        pattern_info["nl_summary"] = "Pattern summary:\n" + "\n".join(summary_lines)
        return pattern_info

    def _format_pattern_info(self, pattern_info: Mapping[str, Any]) -> str:
        """Render pattern_info into a human-readable summary that is useful for few-shot planning."""
        if not isinstance(pattern_info, Mapping):
            return "pattern_info: unavailable"
        summary = pattern_info.get("nl_summary")
        if isinstance(summary, str) and summary.strip():
            return summary

        tm = pattern_info.get("task_matrix", {}) if isinstance(pattern_info.get("task_matrix", {}), Mapping) else {}
        tpl = pattern_info.get("template", {}) if isinstance(pattern_info.get("template", {}), Mapping) else {}
        schema = pattern_info.get("json_schema", {}) if isinstance(pattern_info.get("json_schema", {}), Mapping) else {}
        bs = pattern_info.get("batching_strategy", {}) if isinstance(pattern_info.get("batching_strategy", {}), Mapping) else {}
        lines = []
        lines.append(f"task_matrix rows={tm.get('num_rows')}, cols={tm.get('num_cols')}, columns={tm.get('columns')}")
        lines.append(f"schema columns={schema.get('columns')}")
        lines.append(f"template complexity={tpl.get('complexity')}, content_preview={str(tpl.get('content'))[:120]}")
        lines.append(
            f"batching type={bs.get('type')}, chunk_size={bs.get('chunk_size')}, rationale={bs.get('rationale') or bs.get('strategy_info')}"
        )
        return "\n".join(lines)

    def _build_memory_block(
        self,
        task_main: str,
        task_matrix: List[List[Any]],
        json_schema: Dict[str, Any],
        batch_strategy: Mapping[str, Any],
        template: str,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        memory_block_lines: List[str] = []
        success_case = None
        fail_case = None
        insights = []
        if self.expmemory is not None and hasattr(self.expmemory, "retrieve_memory"):
            try:
                success_list, fail_list, insights = self.expmemory.retrieve_memory(
                    query_task=task_main,
                    successful_topk=1,
                    failed_topk=1,
                    insight_topk=self.insight_topk,
                )
                
                success_case = success_list[0] if success_list else None
                fail_case = fail_list[0] if fail_list else None
            except Exception:
                pass
        projected_insights: List[str] = []
        raw_insights: List[str] = [str(h) for h in insights] if insights else []
        if raw_insights and self.expmemory is not None and hasattr(self.expmemory, "project_insights"):
            try:
                traj_context = (
                    "MapReduce call inputs:\n"
                    f"task_matrix: {task_matrix}\n"
                    f"template: {template}\n"
                    f"json_schema: {json_schema}\n"
                    f"batch_strategy: {batch_strategy}\n"
                )
                projected_insights = self.expmemory.project_insights(
                    raw_insights=raw_insights,
                    role="MapReduce batching strategist (efficiency and quality)",
                    task_traj=traj_context,
                )
                if not isinstance(projected_insights, list):
                    projected_insights = []
            except Exception:
                projected_insights = []
        if raw_insights or projected_insights:
            memory_block_lines.append("Overall batching strategy hint:")
            if projected_insights:
                memory_block_lines.extend(projected_insights)
            else:
                memory_block_lines.extend(raw_insights)
        if success_case:
            pi = success_case.get_extra_field("pattern_info")
            memory_block_lines.append("1 Successful task:")
            memory_block_lines.append(str(success_case.task_description or success_case.task_main))
            memory_block_lines.append("1.1 Pattern info:")
            memory_block_lines.append(self._format_pattern_info(pi))
        if fail_case:
            pi = fail_case.get_extra_field("pattern_info")
            memory_block_lines.append("2 Failed task:")
            memory_block_lines.append(str(fail_case.task_description or fail_case.task_main))
            memory_block_lines.append("2.1 Pattern info:")
            memory_block_lines.append(self._format_pattern_info(pi))
        memory_text = "\n".join(memory_block_lines) if memory_block_lines else "No memory available."

        task_matrix_preview = None
        if isinstance(task_matrix, list):
            task_matrix_preview = [str(r) for r in task_matrix[:5]]
            if len(task_matrix) > 5:
                task_matrix_preview.append(f"... ({len(task_matrix) - 5} more rows)")

        return {
            "call_stage": "plan",
            "memory_block": memory_text,
            "current_mapreduce_args": {
                "task_matrix_preview": task_matrix_preview,
                "template": template[:200],
                "json_schema": json_schema,
                "batch_strategy": batch_strategy,
                "batch_size": batch_size if batch_size is not None else self.default_batch_size,
            },
            "current_task_main": task_main,
            "prompt": (
                "The above items are valuable memory experiences and your current decision parameters and the description of current task; use those"
                "experiences to refine your current_mapreduce_args and re-invoke the mapreduce tool."
            ),
        }

    def _build_agent(self) -> ToolCallingAgent:
        return ToolCallingAgent(
            model=self.model,
            tools=list(self.web_tools),
            max_steps=self.max_steps,
            prompts_type=self.subagent_prompts_type,
            prompt_templates=self.subagent_prompt_templates,
        )

    def _format_schema_hints(self, keys: List[str], schema: Dict[str, Any]) -> str:
        properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
        required = schema.get("required", []) if isinstance(schema, Mapping) else []
        lines: List[str] = []
        for key in keys:
            prop = properties.get(key) if isinstance(properties, Mapping) else None
            fragments: List[str] = []
            if isinstance(prop, Mapping):
                type_hint = prop.get("type")
                if type_hint:
                    fragments.append(f"type={type_hint}")
                # Use the field name itself as the primary search keyword
                fragments.append(f"Search using the field name '{key}' as the main keyword")
            elif prop is not None:
                fragments.append(str(prop))
            if not fragments:
                fragments.append("no additional guidance provided")
            requirement = "required" if isinstance(required, list) and key in required else "optional"
            lines.append(f"- {key} ({requirement}): " + "; ".join(fragments))
        return "\n".join(lines)

    def _build_batch_prompt(self, prompts: List[str], keys: List[str], schema: Dict[str, Any]) -> str:
        numbered = "\n".join(f"{idx + 1}. {prompt}" for idx, prompt in enumerate(prompts))
        return (
            "You are coordinating a team of research assistants. Solve every sub-question below using the available "
            "tools and then call `final_answer` exactly once with the final results.\n"
            f"Return ONLY a JSON array. Find the following fields for each sub-question: {', '.join(keys)}\n\n"
            "If information is missing, use an empty string \"\" for that field.\n"
            f"{numbered}"
        )

    def _serialize_agent_trace(self, agent: ToolCallingAgent) -> Dict[str, Any]:
        trace: Dict[str, Any] = {
            "system_prompt": getattr(agent, "system_prompt", ""),
            "task": getattr(agent, "task", ""),
            "steps": agent.memory.get_full_steps(),
        }
        tool_names = list(agent.tools.keys()) if getattr(agent, "tools", None) else []
        if tool_names:
            trace["tool_names"] = tool_names
        return make_json_serializable(trace)

    def _safe_json_extract(
        self,
        text: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        keys: List[str],
        fallback_rows: Optional[List[List[Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(text, list):
            return [self._project_record(item, keys) for item in text if isinstance(item, Mapping)]
        if isinstance(text, Mapping):
            return [self._project_record(text, keys)]

        if not isinstance(text, str):
            text = str(text)

        candidate = text.strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("["), candidate.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    parsed = None
            else:
                start, end = candidate.find("{"), candidate.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(candidate[start : end + 1])
                    except json.JSONDecodeError:
                        parsed = None
                else:
                    parsed = None

        if isinstance(parsed, Mapping):
            return [self._project_record(parsed, keys)]
        if isinstance(parsed, list):
            return [self._project_record(item, keys) for item in parsed if isinstance(item, Mapping)]

        return self._make_fallback_records(keys, fallback_rows)

    def _project_record(
        self,
        record: Dict[str, Any],
        keys: List[str],
        *,
        fallback: Optional[List[Any]] = None,
    ) -> Dict[str, str]:
        projected: Dict[str, str] = {}
        for key in keys:
            value = record.get(key, "")
            if value is None:
                value = ""
            projected[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if fallback and not any(v.strip() for v in projected.values()):
            for idx, key in enumerate(keys):
                if idx < len(fallback):
                    projected[key] = str(fallback[idx])
        return projected

    def _make_fallback_records(
        self,
        keys: List[str],
        rows: Optional[List[List[Any]]],
    ) -> List[Dict[str, str]]:
        if not rows:
            return [{key: "" for key in keys}]
        fallback_records: List[Dict[str, str]] = []
        for row in rows:
            mapped: Dict[str, str] = {}
            for idx, key in enumerate(keys):
                mapped[key] = str(row[idx]) if idx < len(row) and row[idx] is not None else ""
            fallback_records.append(mapped)
        return fallback_records

    @staticmethod
    def _normalize_entity(value: str) -> str:
        return value.strip().lower()

    def _process_batch(
        self,
        batch: List[List[Any]],
        template: str,
        keys: List[str],
        schema: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        prompts = [self._render(template, row) for row in batch]
        attempt_prompt = self._build_batch_prompt(prompts, keys, schema)

        result_rows: List[Dict[str, str]] = []
        attempt_logs: List[Dict[str, Any]] = []
        for attempt in range(1, self.max_retries + 1):
            agent = self._build_agent()
            raw_output = agent.run(attempt_prompt)
            parsed = self._safe_json_extract(raw_output, keys, fallback_rows=batch)
            attempt_log: Dict[str, Any] = {
                "attempt": attempt,
                "prompt": attempt_prompt,
                "raw_output": raw_output,
                "parsed": deepcopy(parsed) if parsed is not None else None,
                "agent_trace": self._serialize_agent_trace(agent),
            }
            if parsed:
                projected_rows = [
                    self._project_record(entry, keys, fallback=batch[idx] if idx < len(batch) else None)
                    for idx, entry in enumerate(parsed)
                ]
                attempt_log["projected_rows"] = deepcopy(projected_rows)
                result_rows = projected_rows
                if len(result_rows) >= len(batch):
                    attempt_logs.append(attempt_log)
                    break
            attempt_logs.append(attempt_log)
            # build continuation instructions
            partial_json = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else "[]"
            missing_info = (
                "At least one sub-task was not answered with the required schema. "
                "Re-run the necessary searches and respond again with a JSON array only."
            )
            attempt_prompt = (
                f"{self._build_batch_prompt(prompts, keys, schema)}\n"
                f"Previous attempt ({attempt}) output:\n{partial_json}\n"
                f"{missing_info}"
            )

        if not result_rows:
            fallback_rows = self._make_fallback_records(keys, batch)
            if attempt_logs:
                attempt_logs[-1]["fallback_rows"] = deepcopy(fallback_rows)
            else:
                attempt_logs.append({"fallback_rows": deepcopy(fallback_rows)})
            result_rows = fallback_rows
        return result_rows, make_json_serializable(attempt_logs)

    def forward(
            self,
            task_matrix: List[List[Any]],
            template: str,
            json_schema: Dict[str, Any],
            batch_size: Optional[int] = None,
            batch_strategy: Optional[Dict[str, Any]] = None,
        ) -> str:
            if not isinstance(task_matrix, list):
                raise ValueError("task_matrix must be a list of rows.")
            matrix_serializable = make_json_serializable(task_matrix)
            try:
                matrix_repr = json.dumps(matrix_serializable, ensure_ascii=False, indent=2)
            except TypeError:
                matrix_repr = str(matrix_serializable)
            self._logger.info(
                "mapreducetool received task_matrix with %d rows:\n%s",
                len(task_matrix),
                matrix_repr,
            )
            keys = self._schema_keys(json_schema)
            eff_bs = self.default_batch_size
            if batch_size is not None:
                try:
                    eff_bs = max(1, int(batch_size))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}") from exc

            normalized_strategy = self._normalize_batch_strategy(batch_strategy)
            batch_plan, normalized_strategy = self._build_batch_plan(task_matrix, normalized_strategy, eff_bs)
            batches = [entry["rows"] for entry in batch_plan]
            manifest_serialized = [make_json_serializable(entry["manifest"]) for entry in batch_plan]
            strategy_snapshot = make_json_serializable(normalized_strategy)

            # Determine stage: plan or execute.
            stage = "execute"
            # Only attempt a plan-phase when enabled and we have a memory backend.
            if self.plan_mode_enabled and not self._plan_done and self.expmemory is not None:
                stage = "plan"

            run_trace: Dict[str, Any] = {
                "call_stage": stage,
                "started_at": time.time(),
                "template": template,
                "template_preview": template[:2000],
                "json_schema": make_json_serializable(json_schema),
                "json_schema_keys": keys,
                "total_rows": len(task_matrix),
                "batch_size": eff_bs,
                "batch_strategy": strategy_snapshot,
                "batch_manifest": manifest_serialized,
                "task_matrix": matrix_serializable,
                "task_matrix_preview": matrix_serializable[:2] if isinstance(matrix_serializable, list) else None,
                "batches": [],
            }
            if stage == "plan":
                self._plan_done = True
                query_task = getattr(self, "current_task_main", None) or template
                mem_payload = self._build_memory_block(
                    query_task,
                    task_matrix,
                    json_schema,
                    batch_strategy,
                    template,
                    batch_size,
                )
                run_trace["memory_block"] = mem_payload
                self.last_run_trace = run_trace
                self.task_records.append(run_trace)
                return json.dumps(mem_payload, ensure_ascii=False)
            all_rows: List[Dict[str, str]] = []
            observed_norm: Set[str] = set()
            if self.enumerated_entities:
                for row in task_matrix:
                    if not row:
                        continue
                    entity_value = row[0]
                    if entity_value is None:
                        continue
                    norm = self._normalize_entity(str(entity_value))
                    observed_norm.add(norm)
                if not self.collected_entities and self.enumerated_entities:
                    missing_norm_initial = self.enumerated_entities - observed_norm
                    if missing_norm_initial:
                        missing_entities = [
                            self.enumeration_lookup.get(norm, norm) for norm in sorted(missing_norm_initial)
                        ]
                        run_trace["remaining_entities"] = missing_entities
                        message = (
                            "Unable to launch mapreducetool: the enumerated entity inventory has not been fully "
                            f"covered yet. Missing entities: {missing_entities}. Please regenerate the "
                            "`task_matrix` (or rerun search enumeration) so that every enumerated entity is included "
                            "before invoking mapreducetool."
                        )
                        self._logger.warning(message)
                        run_trace["ended_at"] = time.time()
                        run_trace["total_results"] = 0
                        self.last_run_trace = run_trace
                        self.task_records.append(run_trace)
                        raise AgentExecutionError(message, self._logger)

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_idx = {
                    executor.submit(self._process_batch, batch, template, keys, json_schema): idx
                    for idx, batch in enumerate(batches)
                }
                for future in as_completed(future_to_idx):
                    batch_idx = future_to_idx[future]
                    manifest_snapshot = manifest_serialized[batch_idx] if batch_idx < len(manifest_serialized) else {}
                    batch_info = {
                        "batch_id": manifest_snapshot.get("batch_id"),
                        "index": batch_idx,
                        "size": len(batches[batch_idx]),
                        "started_at": run_trace["started_at"],
                        "shared_context": manifest_snapshot.get("shared_context"),
                        "strategy_used": manifest_snapshot.get("strategy_used"),
                        "rationale": manifest_snapshot.get("rationale"),
                    }
                    try:
                        batch_result, attempt_logs = future.result()
                    except Exception as exc:
                        self._logger.exception("Batch %s failed: %s", batch_idx, exc)
                        batch_result = self._make_fallback_records(keys, batches[batch_idx])
                        batch_info["error"] = str(exc)
                        attempt_logs = [{"error": str(exc), "fallback_rows": deepcopy(batch_result)}]
                    batch_info["rows"] = len(batch_result)
                    batch_info["attempts"] = attempt_logs
                    run_trace["batches"].append(batch_info)
                    all_rows.extend(batch_result)

            if not all_rows:
                all_rows = [{key: "" for key in keys}]

            is_enumeration_call = bool(task_matrix) and len(task_matrix) == 1 and (
                not isinstance(task_matrix[0], list) or len(task_matrix[0]) <= 1
            )
            if is_enumeration_call and all_rows:
                enumerated_norm: Set[str] = set()
                lookup: Dict[str, str] = {}
                for row in all_rows:
                    if not isinstance(row, Mapping):
                        continue
                    candidate_value: Optional[str] = None
                    for possible_key in ("entity", "album", "name"):
                        value = row.get(possible_key)
                        if value:
                            candidate_value = str(value)
                            break
                    if not candidate_value:
                        continue
                    norm = self._normalize_entity(candidate_value)
                    enumerated_norm.add(norm)
                    lookup.setdefault(norm, candidate_value)
                if enumerated_norm:
                    self.enumerated_entities = enumerated_norm
                    self.enumeration_lookup = lookup
                    self.collected_entities = set()
                    run_trace.setdefault("enumeration_stats", {})["total_entities"] = len(enumerated_norm)
                else:
                    self.enumerated_entities = set()
                    self.enumeration_lookup = {}
                    self.collected_entities = set()
            else:
                if self.enumerated_entities:
                    observed_norm: Set[str] = set()
                    for row in task_matrix:
                        if not row:
                            continue
                        entity_value = row[0]
                        if entity_value is None:
                            continue
                        norm = self._normalize_entity(str(entity_value))
                        observed_norm.add(norm)
                    if observed_norm:
                        self.collected_entities.update(observed_norm)
                        missing_norm = self.enumerated_entities - self.collected_entities
                        if missing_norm:
                            missing_entities = [self.enumeration_lookup.get(norm, norm) for norm in sorted(missing_norm)]
                            run_trace["remaining_entities"] = missing_entities
                            self._logger.warning("Remaining entities pending coverage: %s", missing_entities)
                        unexpected_norm = observed_norm - self.enumerated_entities
                        if unexpected_norm:
                            unexpected_entities: List[str] = []
                            for row in task_matrix:
                                if not row:
                                    continue
                                entity_value = str(row[0])
                                norm = self._normalize_entity(entity_value)
                                if norm in unexpected_norm and entity_value not in unexpected_entities:
                                    unexpected_entities.append(entity_value)
                            if not unexpected_entities:
                                unexpected_entities = [self.enumeration_lookup.get(norm, norm) for norm in sorted(unexpected_norm)]
                            run_trace["unexpected_entities"] = unexpected_entities
                            self._logger.warning("Encountered entities outside enumeration: %s", unexpected_entities)

            run_trace["ended_at"] = time.time()
            run_trace["total_results"] = len(all_rows)
            try:
                run_trace["pattern_info"] = self._build_pattern_info(task_matrix, template, json_schema, normalized_strategy)
            except Exception:
                pass
            self.last_run_trace = run_trace
            self.task_records.append(run_trace)

            return "\n".join(json.dumps(row, ensure_ascii=False) for row in all_rows)


mapreducetool = MapReduceTool
