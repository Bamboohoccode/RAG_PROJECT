import os
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import get_settings


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
class Production_Agent:
    def __init__(self):
        settings = get_settings()
        self.primary_llm = ChatGoogleGenerativeAI(model=settings.primary_model,
                                                  temperature = 0.0,
                                                  timeout = 30,
                                                  )
        self.fallback_llm = ChatGoogleGenerativeAI(model=settings.fallback_model,
                                                  temperature = 0.0,
                                                  timeout = 30,
                                                  )
        self.max_entries = settings.max_retries
        self.graph = self._build_graph()
    def _build_graph(self):

        def process_message(state: AgentState) -> dict:
            try:
                response = self.primary_llm.invoke(state['messages'])
                return{
                    "messages" : [response],
                    "error" : None,
                    "model_used" : "primary"
                }
            except Exception as E:
                return {
                    "error" : str(E),
                    "model_used" : "",
                }
        def try_fallback(state: AgentState) -> dict:
            count = state.get('retry_count',0) + 1
            try: 
                response = self.fallback_llm.invoke(state["messages"])
                return{
                    "messages" : [response],
                    "error" : None,
                    "model_used" : "fallback",
                    "retry_count" : count
                }
            except Exception as e:
                return {
                    "error" : str(e),
                    "model_used" : "",
                    "retry_count" : count
                }

        def handle_error(state: AgentState) -> dict:
            return {
                "messages" :[AIMessage(content= (
                    "I'm sorry,I'm having trouble processing your request."
                ))],
                "model_used" : "error_handler"
            }
        def route_after_process(state: AgentState) -> str:
            if state.get("error") is None:
                return "done"
            elif state.get('retry_count',0) < self.max_entries:
                return "fallback"
            else:
                return "error"
        graph = StateGraph(AgentState)
        graph.add_node("process",process_message)
        graph.add_node("fallback",try_fallback)
        graph.add_node("error",handle_error)

        graph.add_edge(START,"process")
        FULL = {"done" : END, "fallback" : "fallback", "error" : "error"}
        graph.add_conditional_edges(
            "process",
            route_after_process,
            FULL
        )
        graph.add_conditional_edges(
            "fallback",
            route_after_process,
            FULL
        )
        graph.add_edge("error",END)
        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        """
        Invoke the agent with a user message.
        Returns: {"response": str, "model_used": str, "error": str | None}
        """
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)],
            "error": None,
            "retry_count": 0,
            "model_used": "",
        })

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }


def get_model():
    """Initialize primary model with fallback mechanism."""
    settings = get_settings()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        primary = ChatGoogleGenerativeAI(
            model=settings.primary_model,
            google_api_key=settings.google_api_key,
            temperature=0.7,
        )

        fallback = ChatGoogleGenerativeAI(
            model=settings.fallback_model,
            google_api_key=settings.google_api_key,
            temperature=0.7,
        )

        return primary.with_fallbacks([fallback])
    except ImportError:
        raise ImportError(
            "Gói 'langchain-google-genai' chưa được cài đặt. "
            "Vui lòng chạy: uv add langchain-google-genai"
        )


def call_model(state: AgentState) -> dict:
    """Agent node that invokes the model with the current conversation history."""
    settings = get_settings()
    model = get_model()

    response = model.invoke(state["messages"])
    return {
        "messages": [response],
        "model_used": settings.primary_model,
    }


# Build LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# In-memory checkpointer for multi-turn conversation threads
checkpointer = MemorySaver()
app_graph = workflow.compile(checkpointer=checkpointer)


@traceable(name="run_agent")
async def run_agent(
    message: str,
    thread_id: str = "default",
) -> tuple[str, str, int, int]:
    """
    Execute the agent workflow for a given user message and thread.

    Returns:
        tuple of (response_text, model_used, input_tokens, output_tokens)
    """
    settings = get_settings()
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {"messages": [HumanMessage(content=message)]}
    result = await app_graph.ainvoke(inputs, config=config)

    last_message = result["messages"][-1]
    if isinstance(last_message.content, str):
        response_text = last_message.content
    elif isinstance(last_message.content, list):
        texts = []
        for block in last_message.content:
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
            elif isinstance(block, str):
                texts.append(block)
        response_text = " ".join(texts) if texts else str(last_message.content)
    else:
        response_text = str(last_message.content)

    # Extract token usage metadata if available
    input_tokens = 0
    output_tokens = 0
    if hasattr(last_message, "usage_metadata") and last_message.usage_metadata:
        input_tokens = last_message.usage_metadata.get("input_tokens", 0)
        output_tokens = last_message.usage_metadata.get("output_tokens", 0)

    model_used = result.get("model_used", settings.primary_model)

    return response_text, model_used, input_tokens, output_tokens
