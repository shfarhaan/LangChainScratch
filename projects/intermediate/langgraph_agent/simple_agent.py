"""
Simple LangGraph Agent
======================
Basic agent using LangGraph with state management.

This demonstrates:
- State graph creation
- Node definitions
- State transitions
- Basic workflow

Usage:
    python simple_agent.py
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# Define the state structure
class AgentState(TypedDict):
    """State for our agent"""
    task: str
    plan: str
    execution: str
    messages: Annotated[List[str], operator.add]
    step: str


def planning_node(state: AgentState) -> dict:
    """
    Planning node: Create a plan for the task
    """
    task = state["task"]
    
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    
    prompt = f"""
    Create a simple step-by-step plan for the following task:
    
    Task: {task}
    
    Provide a clear, numbered plan with 3-5 steps.
    """
    
    response = llm.invoke(prompt)
    plan = response.content
    
    print("\n" + "="*60)
    print("PLANNING PHASE")
    print("="*60)
    print(f"Task: {task}")
    print(f"\nPlan:\n{plan}")
    print("="*60)
    
    return {
        "plan": plan,
        "messages": ["Created plan successfully"],
        "step": "execute"
    }


def execution_node(state: AgentState) -> dict:
    """
    Execution node: Execute the plan
    """
    task = state["task"]
    plan = state["plan"]
    
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    
    prompt = f"""
    Execute this plan for the task:
    
    Task: {task}
    Plan: {plan}
    
    Provide a detailed execution with actual content or steps taken.
    """
    
    response = llm.invoke(prompt)
    execution = response.content
    
    print("\n" + "="*60)
    print("EXECUTION PHASE")
    print("="*60)
    print(f"Execution:\n{execution}")
    print("="*60)
    
    return {
        "execution": execution,
        "messages": ["Executed plan successfully"],
        "step": "complete"
    }


def create_agent():
    """
    Create and compile the LangGraph agent
    """
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("plan", planning_node)
    workflow.add_node("execute", execution_node)
    
    # Set entry point
    workflow.set_entry_point("plan")
    
    # Add edges (transitions)
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def main():
    """
    Main function to run the agent
    """
    print("\n" + "="*60)
    print("LangGraph Simple Agent")
    print("="*60)
    print("\nThis agent will:")
    print("1. Plan your task")
    print("2. Execute the plan")
    print("="*60)
    
    # Get task from user
    task = input("\nEnter your task: ").strip()
    
    if not task:
        print("No task provided. Exiting.")
        return
    
    # Create agent
    print("\nInitializing agent...")
    agent = create_agent()
    
    # Run agent
    print("\nRunning agent...\n")
    
    result = agent.invoke({
        "task": task,
        "plan": "",
        "execution": "",
        "messages": [],
        "step": "start"
    })
    
    # Display final result
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"\nTask: {result['task']}")
    print(f"\nPlan:\n{result['plan']}")
    print(f"\nExecution:\n{result['execution']}")
    print("\nMessages:")
    for msg in result['messages']:
        print(f"  - {msg}")
    print("="*60)
    print("\n✓ Agent completed successfully!\n")


if __name__ == "__main__":
    main()
