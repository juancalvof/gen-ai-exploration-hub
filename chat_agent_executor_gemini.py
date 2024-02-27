import json
import operator
from typing import TypedDict, Annotated, Sequence

from colorama import Fore
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import tavily_search, alpha_vantage
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_google_vertexai.chat_models import ChatVertexAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation, ToolExecutor

llm = ChatVertexAI(model_name="gemini-pro")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# The function that calls the LLM
def node_function_call_model(state):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


# The function that calls the tools
def node_function_call_tools(state):
    messages = state['messages']
    # The function call is the last message
    last_message = messages[-1]

    # We build a ToolInvocation object with the tool name and the input
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"])
    )

    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)

    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)

    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


def conditional_edge_function(state):
    messages = state['messages']
    last_message = messages[-1]

    if "function_call" in last_message.additional_kwargs:
        return "continue"
    else:
        return "end"


# TOOLS
# Setting up the tools that the agent will use
# TavilySearch tool with API wrapper
search_tavily = tavily_search.TavilySearchAPIWrapper()
description = """"A search engine optimized for comprehensive, accurate, \
and trusted results. Useful for when you need to answer questions \
about current events or about recent information. \
Input should be a search query. \
If the user is asking about something that you don't know about, \
you should probably use this tool to see if that can provide any information."""
tool_tavily = TavilySearchResults(api_wrapper=search_tavily, description=description)

# AlphaVantage tool with API wrapper
alpha_vantage = alpha_vantage.AlphaVantageAPIWrapper()
tool_alpha_vantage = StructuredTool.from_function(
    name="alpha_vantage",
    func=alpha_vantage.run,
    description="useful for when you need to answer questions about exchange rates.",
)
# WRAPPING THE TOOLS
# Wrap the tools in a ToolExecutor. It takes a ToolInvocation and returns an output.
tools = [tool_alpha_vantage, tool_tavily]
llm = llm.bind(functions=tools)
tool_executor = ToolExecutor(tools=tools)

# DEFINING THE GRAPH
# Define a Langgraph graph
workflow = StateGraph(AgentState)

# Add the nodes we will use
workflow.add_node("agent", node_function_call_model)
workflow.add_node("tools", node_function_call_tools)

# Setting where we start
workflow.set_entry_point("agent")
# We add an edge from the tools back to the agent
workflow.add_edge('tools', 'agent')
# The conditional_edge_function will decide if we continue and call the tools or if we end
workflow.add_conditional_edges(
    "agent",
    conditional_edge_function,
    {
        "continue": "tools",
        "end": END
    }
)

# We compile to a Langchain runnable
app = workflow.compile()

if __name__ == "__main__":
    inputs = [{"messages": [HumanMessage(content=
                                         """I want to travel from my original country, Spain to the United States. 
                                         What is the exchange rate?""")]},
              {"messages": [HumanMessage(content=
                                         "Please analyze recent news reports for the latest advancements in renewable "
                                         "energy. Select five articles, each from a different reputable news outlet. "
                                         "I'm interested in a diverse range of scientific and technological "
                                         "perspectives. For each article, provide the following: "
                                         "Article Name:"
                                         "Date Published:"
                                         "URL Link to Source:"
                                         "Summary: A concise overview of the key advancement(s) described in the "
                                         "article, including its potential impact on the renewable energy industry.")]},
              {"messages": [HumanMessage(content=
                                         "Hello, how are you?")]}]

    for input_workflow in inputs:
        for output in app.stream(input_workflow):
            for key, value in output.items():
                print(f"{Fore.CYAN}Output from node '{key}':")
                for k, v in value.items():
                    print(f"{Fore.GREEN}{k}: {Fore.WHITE}{v}")
                print(f"{Fore.CYAN}---\n")
