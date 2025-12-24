import pytest
import json
import asyncio
from typing import List, Optional, Any, Dict, TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    ToolCall,
    BaseMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from pkg.react import ReActAgent
from util.llm import load_llm


# Define a simple tool
@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers.
    Args:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The sum of a and b. add(a, b) = add(b, a)
    """
    return a + b

@tool
def get_today_weather_in_beijing() -> str:
    """
    Get the weather today in beijing.
    Returns:
        str: The weather today.
    """
    return "It is sunny today."


# Define structured output schema
class OutputSchema(BaseModel):
    answer: str = Field(description="The final answer")
    reasoning: str = Field(description="The reasoning process")


class AgentTestState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    answer: Optional[str]
    reasoning: Optional[str]


# Mock Chat Model
class MockChatModel(BaseChatModel):
    responses: List[BaseMessage]
    response_index: int = 0

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.response_index >= len(self.responses):
            # If out of responses, return the last one or raise error
            # For robustness, let's reuse the last one if available, or just error
            raise IndexError("No more responses in MockChatModel")

        response = self.responses[self.response_index]
        self.response_index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "MockChatModel":
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        async def parse_output(input_val):
            # We invoke the model (self) to get the next response
            res = await self.ainvoke(input_val)
            # We assume the content is a JSON string
            if isinstance(res, BaseMessage):
                return json.loads(res.content)
            return res

        return RunnableLambda(parse_output)


@pytest.mark.anyio
async def test_react_agent_initialization():
    llm = MockChatModel(responses=[])
    agent = ReActAgent(
        name="test_agent",
        llm=llm,
        system_prompt="You are a helpful assistant.",
        tools=[add],
    )
    assert agent.name == "test_agent"
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "add"


@pytest.mark.anyio
async def test_react_agent_simple_invocation():
    # Mock LLM response
    llm = MockChatModel(responses=[AIMessage(content="Hello! How can I help you?")])

    agent = ReActAgent(
        name="test_agent",
        llm=llm,
        system_prompt="You are a helpful assistant.",
        tools=[],
    )

    messages = [HumanMessage(content="Hi")]
    response = await agent.ainvoke(messages)

    # response is the state dict
    assert "messages" in response
    assert isinstance(response["messages"][-1], AIMessage)
    assert response["messages"][-1].content == "Hello! How can I help you?"

@pytest.mark.anyio
async def test_react_agent_structured_output():
    llm = load_llm()

    agent = ReActAgent(
        name="test_agent",
        llm=llm,
        system_prompt="You are a helpful assistant. You should follow the output_shema to reponse",
        tools=[get_today_weather_in_beijing],
        output_schema=OutputSchema,
        state_schema=AgentTestState,
    )

    messages = [HumanMessage(content="What is weather today in beijing? Only return the weather.")]

    try:
        result = await agent.ainvoke(messages)
        assert "sunny" in result["answer"].lower()
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1

    except Exception as e:
        pytest.fail(f"Structured output test failed: {e}")
