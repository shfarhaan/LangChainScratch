"""
Automated Workflow Engine
==========================

Simple workflow automation engine using LangGraph.

Usage:
    python workflow_engine.py
"""

from typing import TypedDict, Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import time

load_dotenv()


# Define workflow state
class WorkflowState(TypedDict):
    workflow_id: str
    status: str
    current_step: str
    data: Dict[str, Any]
    results: List[str]
    errors: List[str]


# Task nodes
def fetch_data(state: WorkflowState):
    """Simulate data fetching"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching data...")
    time.sleep(1)
    
    return {
        "current_step": "fetch_data",
        "data": {
            **state["data"],
            "raw_data": [1, 2, 3, 4, 5]
        },
        "results": state["results"] + ["Data fetched successfully"]
    }


def process_data(state: WorkflowState):
    """Process the data"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing data...")
    time.sleep(1)
    
    raw = state["data"].get("raw_data", [])
    processed = [x * 2 for x in raw]
    
    return {
        "current_step": "process_data",
        "data": {
            **state["data"],
            "processed_data": processed
        },
        "results": state["results"] + [f"Processed {len(processed)} items"]
    }


def analyze_data(state: WorkflowState):
    """Analyze the data using LLM"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing data...")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    processed = state["data"].get("processed_data", [])
    
    prompt = f"""Analyze this processed data: {processed}

Provide:
1. Summary statistics
2. Key insights
3. Recommendations"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_step": "analyze_data",
        "data": {
            **state["data"],
            "analysis": response.content
        },
        "results": state["results"] + ["Analysis completed"]
    }


def generate_report(state: WorkflowState):
    """Generate final report"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating report...")
    time.sleep(1)
    
    analysis = state["data"].get("analysis", "No analysis available")
    
    report = f"""
WORKFLOW EXECUTION REPORT
========================

Workflow ID: {state['workflow_id']}
Status: {state['status']}

RESULTS:
{chr(10).join(f"- {r}" for r in state['results'])}

ANALYSIS:
{analysis}

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return {
        "current_step": "generate_report",
        "status": "completed",
        "data": {
            **state["data"],
            "report": report
        },
        "results": state["results"] + ["Report generated"]
    }


def create_workflow():
    """Create the workflow"""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("fetch", fetch_data)
    workflow.add_node("process", process_data)
    workflow.add_node("analyze", analyze_data)
    workflow.add_node("report", generate_report)
    
    # Define flow
    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "process")
    workflow.add_edge("process", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def execute_workflow(workflow_name: str):
    """Execute the workflow"""
    
    print("\n" + "="*70)
    print(f"Workflow: {workflow_name}")
    print("="*70 + "\n")
    
    # Initialize state
    initial_state = {
        "workflow_id": f"{workflow_name}_{int(time.time())}",
        "status": "running",
        "current_step": "",
        "data": {},
        "results": [],
        "errors": []
    }
    
    # Create and run workflow
    workflow = create_workflow()
    
    print("Starting workflow execution...\n")
    
    final_state = None
    for output in workflow.stream(initial_state):
        for key, value in output.items():
            print(f"Step: {value['current_step']}")
            print(f"Status: {value['status']}")
            print("-" * 70)
        final_state = value
    
    # Display report
    if final_state and "report" in final_state["data"]:
        print("\n" + "="*70)
        print(final_state["data"]["report"])
        print("="*70 + "\n")
    
    return final_state


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return
    
    print("\n" + "="*70)
    print("Automated Workflow Engine")
    print("="*70)
    print("\nAvailable workflows:")
    print("  1. Data Processing Pipeline")
    print("  2. Exit")
    
    choice = input("\nSelect workflow (1-2): ").strip()
    
    if choice == "1":
        result = execute_workflow("data_pipeline")
        print(f"\nWorkflow completed with status: {result['status']}")
    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()
