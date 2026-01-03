from typing import List, TypedDict
from typing import Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class QuestionAnswer(TypedDict):
    index: int
    question: str
    answer: str
    claims: List["Claim"] = []


class Claim(BaseModel):
    """A single claim with its citations."""

    text: str = Field(description="The claim text")
    citations: List[int] = Field(
        description="List of document numbers supporting this claim"
    )


class ClaimAnalysis(BaseModel):
    """Result of claim analysis by Agent 4."""

    question_structure: str = Field(
        description="SINGLE or MULTIPLE - whether question has multiple components"
    )
    question_components: List[str] = Field(
        description="List of question components/sub-questions"
    )
    coverage_assessment: dict = Field(
        description="Mapping of component -> FULLY/PARTIALLY/NOT_ANSWERED"
    )
    claims_to_remove: List[int] = Field(
        description="Indices of irrelevant claims to remove"
    )
    unanswered_components: List[str] = Field(
        description="Components that are not fully answered"
    )
    follow_up_questions: List[str] = Field(
        description="Follow-up questions for unanswered aspects"
    )
    completely_answered: bool = Field(
        description="Whether the question is completely answered"
    )


class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )
