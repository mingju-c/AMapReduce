# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Any, Iterable, List, Optional, Union
from dotenv import load_dotenv
import os
from pathlib import Path
from loguru import logger
from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from tenacity import retry, stop_after_attempt, wait_incrementing

from eval.utils.schema import LLMOutputItem, ModelResponse, ToolCall
from eval.utils.config import model_config
import ipdb
from dataclasses import asdict, dataclass



@retry(stop=stop_after_attempt(8), wait=wait_incrementing(8, 8))
def openai_complete(
    base_url: str,
    api_key: Optional[str],
    messages: Iterable[dict],
    tools: Optional[Iterable[dict]] = None,
    model_name: str = "gpt-4o-2024-05-13",
    retry_if_empty: bool = False,
    **generate_kwargs,
) -> Optional[ChatCompletionMessage]:
    """Complete a prompt with OpenAI APIs."""

    def create_openai_client(base_url, api_key):
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=300,
        )

    openai_client = create_openai_client(base_url, api_key)
    logger.debug(f"messages: {messages}")
    logger.debug(f"tools: {tools}")
    logger.debug(generate_kwargs)
    completion = openai_client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model_name,
        tools=tools,  # type: ignore
        **generate_kwargs,
    )
    message = None

    try:
        message = completion.choices[0].message
    except Exception as e:
        logger.warning(f"Error during completion: {e}")
        return None

    if retry_if_empty and not message.content and not message.tool_calls:
        raise RuntimeError(
            "[openai_complete] Got message, but content and toolcalls is empty, retry"
        )

    return message


@dataclass
class APIResponse:
    content: str

def llm_completion(
    messages: Union[str, List[dict]],
    tools: Optional[List[dict]] = None,
    model_config_name: str = "default_eval_config",
) -> Optional[ChatCompletionMessage]:
    """Complete a prompt with given LLM, raise error if the request failed."""

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    project_root = Path(__file__).resolve().parents[4]
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        # Always load the repo-root .env to avoid picking up nested .env files.
        load_dotenv(dotenv_path, override=True)

    cfg = model_config.get(model_config_name, {})
    base_url = cfg.get("base_url") or os.getenv("OPENAI_BASE_URL") or os.getenv("api_base")
    api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("api_key")
    model_name = cfg.get("model_name", model_config_name)
    generate_kwargs = dict(cfg.get("generate_kwargs", {}))
    if cfg.get("temperature") is not None and "temperature" not in generate_kwargs:
        generate_kwargs["temperature"] = cfg["temperature"]

    if not base_url or not base_url.startswith(("http://", "https://")):
        raise ValueError(
            "LLM evaluator base_url is not configured. Please set OPENAI_BASE_URL/.env "
            "or update eval/utils/config.py for the chosen model_config_name."
        )
    if not api_key:
        raise ValueError(
            "LLM evaluator api_key is missing. Please set OPENAI_API_KEY/.env "
            "or update eval/utils/config.py for the chosen model_config_name."
        )

    response = openai_complete(
        base_url=base_url,
        api_key=api_key,
        messages=messages,
        tools=tools,
        model_name=model_name,
        retry_if_empty=True,
        **generate_kwargs,
    )
    response = APIResponse(content=response.content)
    return response


def transform_model_response(response: Any | None) -> ModelResponse:
    out = ModelResponse()
    if response is None:
        out.error_marker = {"message": "Calling LLM failed."}
        return out

    # Set fields.
    item = LLMOutputItem(content=response.content)
    # Convert into dict to get optional fields.
    resp_dict = response.model_dump()
    if resp_dict.get("reasoning_content"):
        item.reasoning_content = resp_dict["reasoning_content"]
    if resp_dict.get("signature"):
        item.signature = resp_dict["signature"]

    if response.tool_calls:
        item.tool_calls = []
        for tool_call in response.tool_calls:
            item.tool_calls.append(
                ToolCall(
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    # TODO: Randomly generate the ID if not provided.
                    tool_call_id=tool_call.id,
                )
            )
    out.outputs.append(item)
    return out
