import asyncio
import json
from typing import Any, Dict, List, Optional

from langchain.agents import AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel


class ReActAgent:
    def __init__(
        self,
        name,
        llm: BaseChatModel,
        system_prompt: str | SystemMessage,
        state_schema: Optional[BaseModel | Dict] = None,
        tools: Optional[List[BaseTool]] = None,
        output_schema: Optional[type[BaseModel] | Dict] = None,
        checkpointer: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.llm = llm
        if isinstance(system_prompt, str):
            self.system_prompt = SystemMessage(content=system_prompt)
        else:
            self.system_prompt = system_prompt
        self.tools = tools or []
        self._checkpointer = checkpointer
        self.output_schema = output_schema
        self.state_schema = state_schema
        if self.state_schema is None:
            self.state_schema = AgentState

        if self.output_schema is not None:
            if isinstance(self.output_schema, type(BaseModel)):
                schema = self.output_schema.model_json_schema()
            else:
                schema = self.output_schema

            # schema_desc = json.dumps(schema, ensure_ascii=False, indent=2)

            updated_prompt = (
                self.system_prompt.content
                # + f"\n\nOutput Schema:\n{schema_desc}"
                + "\n\nWhen you have gathered enough information, please output the final answer directly without calling any more tools. Do not call the same tool with the same arguments repeatedly if you already have the result."
            )
            self.system_prompt = SystemMessage(content=updated_prompt)

        self.graph = self._build_graph()

        # Deprecated: use tools_condition instead
        self.__tools_by_name = {tool.name: tool for tool in self.tools}

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(self.state_schema)
        agent_node_name = f"{self.name}_agent"
        tools_node_name = f"{self.name}_tools"
        structured_node = f"{self.name}_structured"
        tool_node = ToolNode(self.tools, name=tools_node_name)
        if len(self.tools) > 0:
            workflow.add_node(tools_node_name, tool_node)
            workflow.add_node(agent_node_name, self._call_model)
            workflow.add_edge(START, agent_node_name)
            workflow.add_edge(tools_node_name, agent_node_name)
            if self.output_schema:
                workflow.add_node(structured_node, self._call_structured_output)
                workflow.add_conditional_edges(
                    agent_node_name,
                    tools_condition,
                    {"tools": tools_node_name, END: structured_node},
                )
                workflow.add_edge(structured_node, END)
            else:
                workflow.add_conditional_edges(
                    agent_node_name,
                    tools_condition,
                    {"tools": tools_node_name, END: END},
                )
        else:
            workflow.add_node(
                agent_node_name,
                (
                    self._call_structured_output
                    if self.output_schema
                    else self._call_model
                ),
            )
            workflow.add_edge(START, agent_node_name)
            workflow.add_edge(agent_node_name, END)

        if self._checkpointer is not None:
            return workflow.compile(checkpointer=self._checkpointer)
        return workflow.compile()

    @staticmethod
    def _tool_call_key(tool_call: ToolCall) -> str:
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

    async def _call_tools(self, state: AgentState) -> AgentState:
        """
        Deprecated: use tools_condition instead
        """
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}

        async def execute_tool(tool_call: ToolCall) -> ToolMessage:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in self.__tools_by_name:
                tool = self.__tools_by_name[tool_name]
                try:
                    result = await tool.ainvoke(tool_args)
                    content = (
                        json.dumps(result, ensure_ascii=False)
                        if not isinstance(result, str)
                        else result
                    )
                except Exception as e:
                    content = f"工具执行错误: {str(e)}"
            else:
                content = f"未知工具: {tool_name}"

            return ToolMessage(
                content=content,
                name=tool_name,
                tool_call_id=tool_call["id"],
            )

        outputs = await asyncio.gather(
            *[execute_tool(tc) for tc in last_message.tool_calls]
        )

        return AgentState(messages=list(outputs))

    async def _call_structured_output(
        self, state: AgentState
    ) -> Dict[str, Any] | BaseModel:
        SUMMARY_PROMPT = "According to the above information, summarize and output the structured data. DO NOT output any other text."
        messages = state["messages"] + [SystemMessage(content=SUMMARY_PROMPT)]
        if isinstance(self.output_schema, BaseModel):
            output_schema = self.output_schema.model_json_schema()
        else:
            output_schema = self.output_schema
        llm = self.llm.with_structured_output(
            output_schema, method="json_schema"
        ).with_config(temperature=0.0)
        response = await llm.ainvoke(messages)
        if isinstance(self.output_schema, type(BaseModel)):
            ret = self.output_schema.model_validate(response)
            return ret
        return response

    async def _call_model(self, state: AgentState) -> AgentState:
        messages = [self.system_prompt] + state["messages"]
        llm = self.llm.bind_tools(self.tools) if self.tools else self.llm
        response = await llm.ainvoke(messages)
        if isinstance(response, AIMessage) and response.tool_calls:
            seen = set()
            for msg in state["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        seen.add(self._tool_call_key(tc))

            filtered_tool_calls = [
                tc for tc in response.tool_calls if self._tool_call_key(tc) not in seen
            ]
            if len(filtered_tool_calls) != len(response.tool_calls):
                response = response.model_copy(
                    update={"tool_calls": filtered_tool_calls}
                )
        return AgentState(messages=[response])

    async def ainvoke(self, messages: List[AnyMessage]) -> Dict[str, Any] | BaseModel:
        response = await self.graph.ainvoke(AgentState(messages=messages))
        return response
