from functools import partial
from typing import Annotated, Any, Dict, List, Literal, Optional
from util import env

from langchain.agents import AgentState, create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from const import (
    QUERY_ANALYSIS_PROMPT,
    RAG_AGENT_PROMPT,
    STRUCTURED_OUTPUT_PROMPT,
    SUMMARY_PROMPT,
)
from const.prompt import AGGREGATION_AGENT_PROMPT
from model import QueryAnalysis, QuesntionAnswer


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new


class GraphState(MessagesState):
    """State for main agent graph"""

    questionIsClear: bool
    conversation_summary: str
    originalQuery: str
    rewrittenQuestions: List[str]
    agent_answers: Annotated[List[QuesntionAnswer], accumulate_or_reset]


class QuestionAnswerState(AgentState):
    """State for individual agent subgraph"""

    question: str
    question_index: int
    final_answer: str
    agent_answers: List[QuesntionAnswer]


async def analyze_chat_and_summarize(
    state: GraphState, agent_graph: CompiledStateGraph
):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg
        for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    messages = [SystemMessage(content=SUMMARY_PROMPT)] + [
        HumanMessage(content=conversation)
    ]
    summary_response = await agent_graph.ainvoke(AgentState(messages=messages))
    return {
        "conversation_summary": summary_response.content,
        "agent_answers": [{"__reset__": True}],
    }


async def analyze_and_rewrite_query(state: GraphState, agent_graph: CompiledStateGraph):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (
        f"Conversation Context:\n{conversation_summary}\n"
        if conversation_summary.strip()
        else ""
    ) + f"User Query:\n{last_message.content}\n"

    messages = [SystemMessage(content=QUERY_ANALYSIS_PROMPT)] + [
        HumanMessage(content=context_section)
    ]
    response = await agent_graph.ainvoke(AgentState(messages=messages))
    query_analysis = response["structured_response"]
    if len(query_analysis.questions) > 0 and query_analysis.is_clear:
        delete_all = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if not isinstance(m, SystemMessage)
        ]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": query_analysis.questions,
        }
    else:
        clarification = (
            query_analysis.clarification_needed
            if (
                query_analysis.clarification_needed
                and len(query_analysis.clarification_needed.strip()) > 10
            )
            else "I need more information to understand your question."
        )
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)],
        }


async def human_input_node(state: GraphState):
    return {}


async def route_after_rewrite(
    state: GraphState,
):
    if not state.get("questionIsClear", False):
        return "human_input"
    else:
        return [
            Send(
                "process_question",
                {
                    "question": query,
                    "question_index": idx,
                    "messages": [HumanMessage(content=query)],
                },
            )
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]


async def extract_final_answer(state: QuestionAnswerState):
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            res = {
                "final_answer": msg.content,
                "agent_answers": [
                    {
                        "index": state["question_index"],
                        "question": state["question"],
                        "answer": msg.content,
                    }
                ],
            }
            return res
    return {
        "final_answer": "Unable to generate an answer.",
        "agent_answers": [
            {
                "index": state["question_index"],
                "question": state["question"],
                "answer": "Unable to generate an answer.",
            }
        ],
    }


async def aggregate_responses(state: GraphState, llm):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += f"\nAnswer {i}:\n" f"{ans['answer']}\n"

    user_message = HumanMessage(
        content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}"""
    )
    synthesis_response = llm.invoke(
        [SystemMessage(content=AGGREGATION_AGENT_PROMPT)] + [user_message]
    )

    return {"messages": [AIMessage(content=synthesis_response.content)]}


def create_agent_graph(
    llm: BaseChatModel, tools: List[BaseTool], checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    is_debug = env.is_debug()
    summarize_agent: CompiledStateGraph = create_agent(
        model=llm,
        tools=tools,
        state_schema=AgentState,
        system_prompt=SUMMARY_PROMPT + "\n" + STRUCTURED_OUTPUT_PROMPT,
        response_format=ToolStrategy(QueryAnalysis),
        debug=is_debug,
    )
    rewrite_agent: CompiledStateGraph = create_agent(
        model=llm.with_config(temperature=0.2),
        tools=tools,
        state_schema=AgentState,
        system_prompt=QUERY_ANALYSIS_PROMPT + "\n" + STRUCTURED_OUTPUT_PROMPT,
        response_format=ToolStrategy(QueryAnalysis),
        debug=is_debug,
    )
    rag_agent = create_agent(
        name="rag",
        model=llm,
        tools=tools,
        state_schema=QuestionAnswerState,
        system_prompt=RAG_AGENT_PROMPT,
        debug=is_debug,
    )

    wrapper_builder = StateGraph(QuestionAnswerState)
    wrapper_builder.add_node("rag_agent", rag_agent)
    wrapper_builder.add_node("extract_answer", extract_final_answer)

    wrapper_builder.add_edge(START, "rag_agent")
    wrapper_builder.add_edge("rag_agent", "extract_answer")
    wrapper_builder.add_edge("extract_answer", END)

    agent_subgraph = wrapper_builder.compile()

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node(
        "summarize", partial(analyze_chat_and_summarize, agent_graph=summarize_agent)
    )
    graph_builder.add_node(
        "analyze_rewrite", partial(analyze_and_rewrite_query, agent_graph=rewrite_agent)
    )
    graph_builder.add_node("human_input", human_input_node)
    graph_builder.add_node("process_question", agent_subgraph)
    graph_builder.add_node("aggregate", partial(aggregate_responses, llm=llm))

    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "analyze_rewrite")
    graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
    graph_builder.add_edge("human_input", "analyze_rewrite")
    graph_builder.add_edge(["process_question"], "aggregate")
    graph_builder.add_edge("aggregate", END)

    agent_graph = graph_builder.compile(
        checkpointer=checkpointer, interrupt_before=["human_input"]
    )

    return agent_graph
