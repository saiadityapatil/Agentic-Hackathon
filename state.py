from typing import TypedDict, Optional, Annotated
from typing_extensions import NotRequired
from langgraph.graph import add_messages


def merge_output(left, right):
    """Merge function for concurrent node outputs"""
    if right is not None:
        return right
    return left


class State(TypedDict):
    """
    State schema for the LangChain workflow.
    
    Attributes:
        repo_url: GitHub repository URL to analyze
        azure_credentials: Optional Azure service principal credentials
        resource_group_name: Optional Azure resource group name
        azure_metrics: Optional Azure metrics data
        azure_costs: Optional Azure cost data
        code_summarizer_output: Output from code summarizer agent
        architecture_output: Output from architecture agent
        performance_output: Output from performance agent (concurrent update)
        finops_output: Output from finops agent (concurrent update)
        moderator_output: Final synthesized output from moderator agent
        final_analysis: Formatted final analysis result
    """
    repo_url: str
    azure_credentials: NotRequired[Optional[dict]]
    resource_group_name: NotRequired[Optional[str]]
    azure_metrics: NotRequired[Optional[dict]]
    azure_costs: NotRequired[Optional[dict]]
    code_summarizer_output: NotRequired[Optional[dict]]
    architecture_output: NotRequired[Optional[dict]]
    performance_output: Annotated[Optional[dict], merge_output]
    finops_output: Annotated[Optional[dict], merge_output]
    moderator_output: NotRequired[Optional[dict]]
    final_analysis: NotRequired[Optional[str]]
