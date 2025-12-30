from functools import partial
from typing import Annotated, Any, Dict, List, Literal, Optional
from util import env
import logging

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
from const.prompt import (
    AGGREGATION_AGENT_PROMPT,
    ANSWER_INTEGRATION_PROMPT,
    CLAIM_ANALYSIS_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    FOLLOW_UP_ANSWER_PROMPT,
)
from model import Claim, ClaimAnalysis, QueryAnalysis, QuesntionAnswer

logger = logging.getLogger(__name__)


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
    # RAGentA-inspired fields
    all_claims: List[Claim] = []
    claim_analysis: Optional[ClaimAnalysis] = None


class QuestionAnswerState(AgentState):
    """State for individual agent subgraph"""

    question: str
    question_index: int
    final_answer: str
    agent_answers: List[QuesntionAnswer]
    # RAGentA-inspired fields
    claims: List[Claim] = []
    answer_with_citations: str = ""


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
                "answer_with_citations": msg.content,
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
        "answer_with_citations": "Unable to generate an answer.",
        "agent_answers": [
            {
                "index": state["question_index"],
                "question": state["question"],
                "answer": "Unable to generate an answer.",
            }
        ],
    }


# ===== RAGentA-inspired Functions =====

import re
import json


def extract_claims_from_answer(answer_with_citations: str) -> List[Claim]:
    """Extract claims and their citations from answer with [X] format."""
    if not answer_with_citations.strip():
        return []

    claims = []
    # Pattern to match claim text followed by [X] or [X,Y,Z]
    pattern = r"(.*?)\s*\[(\d+(?:,\s*\d+)*)\]"

    matches = list(re.finditer(pattern, answer_with_citations))

    for match in matches:
        claim_text = match.group(1).strip()
        citation_str = match.group(2)
        citations = [int(c.strip()) for c in citation_str.split(",")]

        if claim_text:
            claims.append(Claim(text=claim_text, citations=citations))

    return claims


async def extract_claims(state: QuestionAnswerState, llm: BaseChatModel) -> Dict:
    """Extract claims from the generated answer (Agent 4: Claim Extraction)."""
    answer = state.get("answer_with_citations", "")

    if not answer.strip():
        return {"claims": []}

    # Try regex extraction first
    claims = extract_claims_from_answer(answer)

    # If no claims found, use LLM to extract
    if not claims:
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        response = await llm.ainvoke([SystemMessage(content=prompt)])

        try:
            # Parse JSON from response
            content = response.content
            # Handle potential markdown code block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)
            claims = [
                Claim(text=c["text"], citations=c.get("citations", []))
                for c in data.get("claims", [])
            ]
        except Exception as e:
            logger.error(f"Failed to parse claims: {e}")
            claims = []

    return {"claims": claims}


def parse_claim_analysis(analysis_text: str) -> ClaimAnalysis:
    """Parse the claim analysis response from LLM."""
    result = {
        "question_structure": "SINGLE",
        "question_components": [],
        "coverage_assessment": {},
        "claims_to_remove": [],
        "unanswered_components": [],
        "follow_up_questions": [],
        "completely_answered": True,
    }

    # Extract question structure
    match = re.search(r"QUESTION STRUCTURE:\s*(SINGLE|MULTIPLE)", analysis_text, re.I)
    if match:
        result["question_structure"] = match.group(1).upper()

    # Extract question components
    components_match = re.search(
        r"QUESTION COMPONENTS:(.*?)(?=CLAIM ANALYSIS:|$)", analysis_text, re.DOTALL
    )
    if components_match:
        components_text = components_match.group(1)
        components = re.findall(r"-\s*(?:Component \d+:)?\s*(.+)", components_text)
        result["question_components"] = [c.strip() for c in components if c.strip()]

    # Extract coverage assessment
    coverage_match = re.search(
        r"COVERAGE ASSESSMENT:(.*?)(?=CLAIMS TO REMOVE:|$)", analysis_text, re.DOTALL
    )
    if coverage_match:
        coverage_text = coverage_match.group(1)
        for line in coverage_text.split("\n"):
            comp_match = re.search(
                r"-\s*(?:Component \d+:)?\s*(.+?):\s*(FULLY|PARTIALLY|NOT)\s*ANSWERED",
                line, re.I
            )
            if comp_match:
                component = comp_match.group(1).strip()
                status = comp_match.group(2).strip().upper()
                result["coverage_assessment"][component] = f"{status}_ANSWERED"

    # Extract claims to remove
    remove_match = re.search(
        r"CLAIMS TO REMOVE:(.*?)(?=UNANSWERED COMPONENTS:|$)", analysis_text, re.DOTALL
    )
    if remove_match:
        remove_text = remove_match.group(1).strip()
        if remove_text.lower() not in ["none", ""]:
            nums = re.findall(r"(?:^|\s)(?:\d+|Claim\s+\d+)", remove_text)
            result["claims_to_remove"] = [
                int(re.search(r"\d+", n).group()) for n in nums if re.search(r"\d+", n)
            ]

    # Extract unanswered components
    unanswered_match = re.search(
        r"UNANSWERED COMPONENTS:(.*?)(?=FOLLOW-UP QUESTIONS:|$)", analysis_text, re.DOTALL
    )
    if unanswered_match:
        unanswered_text = unanswered_match.group(1).strip()
        if unanswered_text.lower() not in ["none", ""]:
            components = re.findall(r"-\s*(.+)", unanswered_text)
            result["unanswered_components"] = [c.strip() for c in components if c.strip()]

    # Extract follow-up questions
    followup_match = re.search(r"FOLLOW-UP QUESTIONS:(.*?)$", analysis_text, re.DOTALL)
    if followup_match:
        followup_text = followup_match.group(1).strip()
        if followup_text.lower() not in ["none", ""]:
            questions = re.findall(r"-\s*(.+)", followup_text)
            result["follow_up_questions"] = [q.strip() for q in questions if q.strip() and "?" in q]

    # Determine if completely answered
    result["completely_answered"] = (
        len(result["unanswered_components"]) == 0
        or all(
            c.lower() in ["none", ""] for c in result["unanswered_components"]
        )
    )

    return ClaimAnalysis(**result)


async def analyze_claims(
    state: QuestionAnswerState, llm: BaseChatModel
) -> Dict:
    """Analyze claims against question components (Agent 4: Claim Judge)."""
    question = state["question"]
    answer = state.get("answer_with_citations", "")
    claims = state.get("claims", [])

    if not claims or not answer.strip():
        return {"claim_analysis": None}

    # Format claims for the prompt
    claims_text = "\n".join(
        f"Claim {i+1}: {c.text} [Citations: {', '.join(map(str, c.citations))}]"
        for i, c in enumerate(claims)
    )

    prompt = CLAIM_ANALYSIS_PROMPT.format(
        original_question=question,
        answer_with_citations=answer,
        claims_text=claims_text,
        docs_text="[Document references available in citations]",
    )

    response = await llm.ainvoke([SystemMessage(content=prompt)])
    analysis = parse_claim_analysis(response.content)

    return {"claim_analysis": analysis}


def remove_citations(text: str) -> str:
    """Remove [X] citation markers from text."""
    return re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", text)


async def process_follow_ups(
    state: QuestionAnswerState,
    llm: BaseChatModel,
    tools: List[BaseTool],
) -> Dict:
    """Process follow-up questions for unanswered components."""
    claim_analysis = state.get("claim_analysis")

    if not claim_analysis or claim_analysis.completely_answered:
        return {"final_answer": state.get("final_answer", "")}

    follow_up_questions = claim_analysis.follow_up_questions
    if not follow_up_questions:
        return {"final_answer": state.get("final_answer", "")}

    current_answer = remove_citations(state.get("answer_with_citations", ""))

    # Process each follow-up question
    for follow_up_q in follow_up_questions[:3]:  # Limit to 3 follow-ups
        # Find vector search tool
        vector_tool = None
        for tool in tools:
            if tool.name == "search_vector_chunk":
                vector_tool = tool
                break

        if not vector_tool:
            continue

        # Retrieve new documents for follow-up
        try:
            new_docs = vector_tool.invoke({"query": follow_up_q})
            if not new_docs:
                continue

            # Format docs for follow-up prompt
            docs_text = "\n\n".join(
                f"Document {i+1}: {doc.page_content if hasattr(doc, 'page_content') else str(doc)}"
                for i, doc in enumerate(new_docs[:5])
            )

            # Generate follow-up answer
            follow_up_prompt = FOLLOW_UP_ANSWER_PROMPT.format(
                original_question=state["question"],
                previous_answer=current_answer,
                follow_up_question=follow_up_q,
                docs_text=docs_text,
            )

            follow_up_response = await llm.ainvoke([SystemMessage(content=follow_up_prompt)])
            follow_up_answer = follow_up_response.content

            # Skip if can't answer
            if "cannot answer" in follow_up_answer.lower() and "available information" in follow_up_answer.lower():
                continue

            # Integrate with previous answer
            integration_prompt = ANSWER_INTEGRATION_PROMPT.format(
                original_question=state["question"],
                previous_answer=current_answer,
                new_answer=follow_up_answer,
            )

            integration_response = await llm.ainvoke([SystemMessage(content=integration_prompt)])
            current_answer = integration_response.content

        except Exception as e:
            logger.error(f"Error processing follow-up: {e}")
            continue

    return {"final_answer": current_answer}


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
    # Claim analysis nodes (RAGentA Agent 4)
    wrapper_builder.add_node("extract_claims", partial(extract_claims, llm=llm))
    wrapper_builder.add_node("analyze_claims", partial(analyze_claims, llm=llm))
    wrapper_builder.add_node(
        "process_follow_ups", partial(process_follow_ups, llm=llm, tools=tools)
    )

    # Subgraph flow with claim analysis
    wrapper_builder.add_edge(START, "rag_agent")
    wrapper_builder.add_edge("rag_agent", "extract_answer")
    wrapper_builder.add_edge("extract_answer", "extract_claims")
    wrapper_builder.add_edge("extract_claims", "analyze_claims")
    wrapper_builder.add_edge("analyze_claims", "process_follow_ups")
    wrapper_builder.add_edge("process_follow_ups", END)

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
