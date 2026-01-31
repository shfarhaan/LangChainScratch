"""
Basic Multi-Agent System
=========================

Simple multi-agent system using LangGraph with specialized agents.

Usage:
    python basic_multi_agent.py
"""

from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

load_dotenv()


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    task: str
    research_data: str
    draft_content: str
    final_content: str
    current_step: str
    iterations: int


def research_agent(state: AgentState):
    """Agent that gathers information"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = f"""You are a research specialist. 
Task: {state['task']}

Gather key information and facts about this topic. Be concise but informative."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=f"Research: {response.content}")],
        "research_data": response.content,
        "current_step": "research"
    }


def writer_agent(state: AgentState):
    """Agent that creates content"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = f"""You are a professional writer.
Task: {state['task']}
Research Data: {state['research_data']}

Write a well-structured article based on the research. Make it engaging and informative."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=f"Draft: {response.content}")],
        "draft_content": response.content,
        "current_step": "writing"
    }


def critic_agent(state: AgentState):
    """Agent that reviews content"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = f"""You are a critical reviewer.
Task: {state['task']}
Content to Review: {state['draft_content']}

Review this content. Is it good enough to publish? If not, provide specific feedback for improvement.
Respond with APPROVED or NEEDS_WORK followed by feedback."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Determine if content is approved
    approved = "APPROVED" in response.content.upper()
    
    return {
        "messages": [AIMessage(content=f"Review: {response.content}")],
        "final_content": state['draft_content'] if approved else "",
        "current_step": "review",
        "iterations": state.get("iterations", 0) + 1
    }


def editor_agent(state: AgentState):
    """Agent that improves content"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # Get latest review feedback
    review_messages = [m for m in state['messages'] if 'Review:' in m.content]
    last_review = review_messages[-1].content if review_messages else "No feedback available"
    
    prompt = f"""You are an editor.
Original Content: {state['draft_content']}
Feedback: {last_review}

Improve the content based on the feedback. Make it better."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=f"Edited: {response.content}")],
        "draft_content": response.content,
        "current_step": "editing"
    }


def should_continue(state: AgentState):
    """Decide whether to continue iterating or finish"""
    
    # Check if content was approved
    if state.get("final_content", ""):
        return "end"
    
    # Check max iterations
    if state.get("iterations", 0) >= 3:
        # Force approval after 3 iterations
        return "end"
    
    return "edit"


def create_workflow():
    """Create the multi-agent workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("write", writer_agent)
    workflow.add_node("review", critic_agent)
    workflow.add_node("edit", editor_agent)
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Add edges
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "review")
    workflow.add_edge("edit", "write")
    
    # Add conditional edge for iteration or end
    workflow.add_conditional_edges(
        "review",
        should_continue,
        {
            "edit": "edit",
            "end": END
        }
    )
    
    return workflow.compile()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return
    
    print("\n" + "="*70)
    print("Multi-Agent Content Creation System")
    print("="*70)
    print("\nAgents: Researcher → Writer → Critic → Editor (iterative)")
    print("Each agent specializes in one task, working together to create content.\n")
    
    # Get task from user
    task = input("What would you like to create content about? ").strip()
    
    if not task:
        task = "The benefits of artificial intelligence in healthcare"
        print(f"Using default task: {task}")
    
    print(f"\nTask: {task}")
    print("\nStarting multi-agent workflow...\n")
    
    # Initialize state
    initial_state = {
        "messages": [],
        "task": task,
        "research_data": "",
        "draft_content": "",
        "final_content": "",
        "current_step": "",
        "iterations": 0
    }
    
    # Create and run workflow
    workflow = create_workflow()
    
    print("="*70)
    
    # Stream the workflow execution
    final_state = None
    for output in workflow.stream(initial_state):
        for key, value in output.items():
            print(f"\n[{key.upper()}]")
            
            # Print the latest message
            if value.get("messages"):
                latest_msg = value["messages"][-1].content
                # Truncate long messages
                if len(latest_msg) > 500:
                    print(latest_msg[:500] + "...\n")
                else:
                    print(latest_msg + "\n")
            
            print("-"*70)
        
        final_state = value
    
    # Display final result
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    
    if final_state and final_state.get("draft_content"):
        print(final_state["draft_content"])
    else:
        print("No content generated")
    
    print("\n" + "="*70)
    print(f"Total iterations: {final_state.get('iterations', 0)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
