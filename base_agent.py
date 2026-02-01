#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
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

from copy import deepcopy
from dotenv import load_dotenv
from utils import safe_json_loads
from typing import Any

from FlashOAgents import ToolCallingAgent
from FlashOAgents import ActionStep, PlanningStep, TaskStep, SummaryStep
from FlashOAgents import (
    WebSearchTool,
    CrawlPageTool,
    VisualInspectorTool,
    AudioInspectorTool,
    TextInspectorTool,
    MapReduceTool,
)

load_dotenv(override=True)

class BaseAgent:
    def __init__(self, model):
        self.model = model
        self.agent_fn = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def capture_trajectory(self, ):
        if not hasattr(self, 'agent_fn'):
            raise ValueError("[capture_trajectory] agent_fn is not defined.")
        if not isinstance(self.agent_fn, ToolCallingAgent):
            raise ValueError("[capture_trajectory] agent_fn must be an instance of ToolCallingAgent.")
        trajectory = []
        for step_num, step in enumerate(self.agent_fn.memory.steps):
            if isinstance(step, TaskStep):
                continue
            elif isinstance(step, PlanningStep):
                traj = {"name": "plan", "value": step.plan, "think": step.plan_think, "cot_think": step.plan_reasoning}
                trajectory.append(traj)
            elif isinstance(step, SummaryStep):
                traj = {"name": "summary", "value": step.summary, "cot_think": step.summary_reasoning}
                trajectory.append(traj)
            elif isinstance(step, ActionStep):
                
                
                safe_tool_calls = step.tool_calls if step.tool_calls is not None else []
                mapreduce_calls = []
                for st in safe_tool_calls:
                    try:
                        name = getattr(st, "name", None)
                        if name != "mapreducetool":
                            continue
                        payload = st.dict()
                        
                        minimal = {
                            "name": payload.get("name", "mapreducetool"),
                            "arguments": payload.get("arguments", {}),
                        }
                        mapreduce_calls.append(minimal)
                    except Exception:
                        continue
                traj = {
                    "name": "action",
                    "tool_calls": mapreduce_calls,
                    
                    "obs": [],
                    "think": step.action_think,
                    "cot_think": step.action_reasoning,
                }
                trajectory.append(traj)
            else:
                raise ValueError("[capture_trajectory] Unknown Step:", step)

        return {
            "agent_trajectory": trajectory,
        }

    def extra_payload(self) -> dict:
        return {}

    def forward(self, task, answer=None, return_json=False, max_retries=3):
        last_error = None
        for _ in range(max_retries):
            try:
                if answer is not None:
                    result = self.agent_fn.run(task, answer=answer)
                else:
                    result = self.agent_fn.run(task)
                if return_json and isinstance(result, str):
                    result = safe_json_loads(result)
                elif not return_json and isinstance(result, dict):
                    result = str(result)
                payload = {"agent_result": result}
                payload.update(self.capture_trajectory())
                extra = self.extra_payload()
                if extra:
                    payload.update(extra)
                return payload
            except Exception as e:
                last_error = e
                print(f"[BaseAgent] error: {e}")
                continue
        return {"error": str(last_error)}


class SearchAgent(BaseAgent):
    def __init__(self, model, summary_interval, prompts_type, max_steps, **kwargs):
        super().__init__(model)

        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=model)
        tools = [web_tool, crawl_tool]
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type
        )

class ManageAgent(BaseAgent):
    def __init__(self, model, summary_interval, prompts_type, max_steps, **kwargs):
        super().__init__(model)

        inspector_text_limit = kwargs.get("inspector_text_limit", 100000)
        inspector_audio_limit = kwargs.get("inspector_audio_limit", 100000)
        prompt_filename = "toolcalling_agent.yaml" if kwargs.get("mapreduce_plan_mode", True) else "toolcalling_agent_cold.yaml"

        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=model)
        visual_tool = VisualInspectorTool(model, inspector_text_limit)
        text_tool = TextInspectorTool(model, inspector_text_limit)
        audio_tool = AudioInspectorTool(model, inspector_audio_limit)

        mapreduce_tool = MapReduceTool(
            model=model,
            web_tools=[web_tool, crawl_tool],
            prompts_type=prompts_type,
            batch_size=kwargs.get("mapreduce_batch_size"),
            max_retries=kwargs.get("mapreduce_max_retries", 2),
            max_steps=kwargs.get("mapreduce_max_steps", max_steps),
            workers=kwargs.get("mapreduce_workers", 4),
            plan_mode_enabled=kwargs.get("mapreduce_plan_mode", True),
            expmemory=kwargs.get("expmemory"),
            insight_topk=kwargs.get("mapreduce_insight_topk", 3),
        )

        self.mapreduce_tool = mapreduce_tool
        tools = [mapreduce_tool,web_tool,crawl_tool]
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            prompt_filename=prompt_filename,
        )

    def extra_payload(self) -> dict:
        payload = super().extra_payload()
        if hasattr(self, "mapreduce_tool") and self.mapreduce_tool is not None:
            
            def _first_execute(rec_list):
                if not isinstance(rec_list, list):
                    return None
                for rec in rec_list:
                    if isinstance(rec, dict) and rec.get("call_stage") == "execute":
                        return rec
                return None
            try:
                exec_trace = _first_execute(self.mapreduce_tool.task_records)
                if isinstance(exec_trace, dict):
                    tm = exec_trace.get("task_matrix")
                    num_rows = len(tm) if isinstance(tm, list) else None
                    num_cols = max((len(r) for r in tm if isinstance(r, list)), default=None) if isinstance(tm, list) else None
                    memory_block = exec_trace.get("memory_block")
                    pattern_info = exec_trace.get("pattern_info")
                    payload["mapreduce_execute_inputs"] = {
                        "num_rows": num_rows,
                        "num_cols": num_cols,
                        
                        "task_matrix": tm,
                        "template": exec_trace.get("template") or exec_trace.get("template_preview"),
                        "json_schema": exec_trace.get("json_schema"),
                        "json_schema_keys": exec_trace.get("json_schema_keys"),
                        "batch_strategy": exec_trace.get("batch_strategy"),
                        "batch_size": exec_trace.get("batch_size"),
                        "pattern_info": pattern_info,
                        "memory_block": memory_block,
                    }
            except Exception:
                pass
        return payload

    def forward(self, task, answer=None, return_json=False, max_retries=3):
        # ensure plan flag reset for a new task
        if hasattr(self, "mapreduce_tool") and getattr(self, "mapreduce_tool") is not None:
            try:
                setattr(self.mapreduce_tool, "_plan_done", False)
                setattr(self.mapreduce_tool, "current_task_main", task)
            except Exception:
                pass

        # primary run
        result = super().forward(task, answer=answer, return_json=return_json, max_retries=max_retries)

        def _has_execute(payload: Any) -> bool:
            
            if isinstance(payload, dict) and isinstance(payload.get("mapreduce_execute_inputs"), dict):
                return True
            
            recs = getattr(self, "mapreduce_tool", None)
            recs = getattr(recs, "task_records", []) if recs is not None else []
            for rec in recs:
                if isinstance(rec, dict) and rec.get("call_stage") == "execute":
                    return True
            return False

        def _has_plan(payload: Any) -> bool:
            recs = getattr(self, "mapreduce_tool", None)
            recs = getattr(recs, "task_records", []) if recs is not None else []
            for rec in recs:
                if isinstance(rec, dict) and rec.get("call_stage") == "plan":
                    return True
            return False

        # If only plan-stage happened, immediately trigger a second run with plan flag already satisfied.
        if not _has_execute(result) and _has_plan(result):
            try:
                if hasattr(self, "mapreduce_tool"):
                    setattr(self.mapreduce_tool, "_plan_done", True)
                second = super().forward(task, answer=answer, return_json=return_json, max_retries=max_retries)
                if isinstance(result, dict) and isinstance(second, dict):
                    merged = dict(result)
                    merged.update(second)
                    result = merged
                else:
                    result = second
            except Exception:
                pass
        return result

class MMSearchAgent(BaseAgent):
    def __init__(self, model, summary_interval, prompts_type, max_steps, **kwargs):
        super().__init__(model)

        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=model)
        visual_tool = VisualInspectorTool(model, 100000)
        text_tool = TextInspectorTool(model, 100000)
        audio_tool = AudioInspectorTool(model, 100000)
        # tools = [web_tool, crawl_tool, visual_tool] text or audio tool may not useful during agent execution.
        tools = [web_tool, crawl_tool, visual_tool, text_tool, audio_tool]

        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type
        )
