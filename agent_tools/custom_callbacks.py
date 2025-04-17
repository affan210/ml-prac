
from typing import Any, Dict, List, Optional
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

from uuid import UUID

class CustomAgentCallbackHandler(BaseCallbackHandler):
    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running.
        **ATTENTION**: This method is called for non-chat models (regular LLMs). If
            you're implementing a handler for a chat model,
            you should use on_chat_model_start instead.
        """
        print(f"\n***Prompts to LLM was:***\n{prompts[0]}")
        print("*"*30)
        print()


    async def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
        # run_id: UUID,
        # parent_run_id: Optional[UUID] = None,
        # tags: Optional[list[str]] = None,
    ) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The response which was generated.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("*"*30)
        print()
    