# LangGraph Agent System

## Overview

Build a sophisticated multi-step agent system using LangGraph that can plan, execute, and iterate on complex tasks with state management.

## What You'll Learn

- **LangGraph Fundamentals**: Nodes, edges, and state graphs
- **State Management**: Persistent state across agent actions
- **Cyclic Workflows**: Loops and conditional branching
- **Multi-Agent Coordination**: Multiple agents working together
- **Human-in-the-Loop**: Adding approval steps
- **Streaming**: Real-time updates for long operations

## Prerequisites

- Completed beginner projects
- Understanding of LangChain agents
- Familiarity with state machines (helpful)

## Project Files

```
langgraph_agent/
├── README.md                    # This file
├── simple_agent.py              # Basic LangGraph agent
├── research_agent.py            # Research agent with tools
├── multi_agent.py               # Multi-agent system
└── human_in_loop.py             # Agent with human approval
```

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs. Unlike traditional chains, LangGraph:

- **Maintains State**: Explicit state that persists across steps
- **Supports Cycles**: Create loops for iterative refinement
- **Enables Branching**: Conditional logic based on state
- **Coordinates Agents**: Multiple agents working together
- **Provides Control**: Pause for human input or approval

## Architecture

### Basic LangGraph Flow

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Node 1    │  ◄───┐
└──────┬──────┘      │
       │             │
       ▼             │
┌─────────────┐      │
│   Node 2    │      │
└──────┬──────┘      │
       │             │
       ▼             │
   Condition ────────┘
       │
       ▼
┌─────────────┐
│     END     │
└─────────────┘
```

## Implementation

### 1. Simple Agent (simple_agent.py)

Basic LangGraph agent that processes tasks with state:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

# Create nodes
def planner(state):
    """Plan what to do"""
    messages = state["messages"]
    # Use LLM to plan
    llm = ChatOpenAI()
    response = llm.invoke(f"Create a plan for: {messages[-1]}")
    return {
        "messages": [response.content],
        "next_step": "execute"
    }

def executor(state):
    """Execute the plan"""
    plan = state["messages"][-1]
    llm = ChatOpenAI()
    response = llm.invoke(f"Execute this plan: {plan}")
    return {
        "messages": [response.content],
        "next_step": "end"
    }

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("plan", planner)
workflow.add_node("execute", executor)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({
    "messages": ["Write a blog post about AI"],
    "next_step": "start"
})
```

### 2. Research Agent (research_agent.py)

Agent that can search, analyze, and synthesize information:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

class ResearchState(TypedDict):
    query: str
    search_results: List[str]
    analysis: str
    report: str
    step: str

def search_node(state):
    """Search for information"""
    query = state["query"]
    # Simulate search (replace with real tool)
    results = [
        f"Result 1 for {query}",
        f"Result 2 for {query}",
        f"Result 3 for {query}"
    ]
    return {
        "search_results": results,
        "step": "analyze"
    }

def analyze_node(state):
    """Analyze search results"""
    results = state["search_results"]
    llm = ChatOpenAI()
    
    analysis_prompt = f"""
    Analyze these search results:
    {chr(10).join(results)}
    
    Provide key insights:
    """
    
    response = llm.invoke(analysis_prompt)
    return {
        "analysis": response.content,
        "step": "report"
    }

def report_node(state):
    """Generate final report"""
    query = state["query"]
    analysis = state["analysis"]
    
    llm = ChatOpenAI()
    report_prompt = f"""
    Create a comprehensive report on: {query}
    
    Based on this analysis:
    {analysis}
    
    Format as a structured report:
    """
    
    response = llm.invoke(report_prompt)
    return {
        "report": response.content,
        "step": "complete"
    }

# Build research workflow
workflow = StateGraph(ResearchState)
workflow.add_node("search", search_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "analyze")
workflow.add_edge("analyze", "report")
workflow.add_edge("report", END)

app = workflow.compile()

# Use it
result = app.invoke({
    "query": "Latest trends in AI",
    "search_results": [],
    "analysis": "",
    "report": "",
    "step": "start"
})

print(result["report"])
```

### 3. Multi-Agent System (multi_agent.py)

Coordinate multiple specialized agents:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MultiAgentState(TypedDict):
    task: str
    research: str
    code: str
    review: str
    final: str

def researcher_agent(state):
    """Research agent"""
    task = state["task"]
    # Research logic
    return {"research": f"Research findings for {task}"}

def coder_agent(state):
    """Coding agent"""
    research = state["research"]
    # Coding logic
    return {"code": f"Code based on: {research}"}

def reviewer_agent(state):
    """Review agent"""
    code = state["code"]
    # Review logic
    return {"review": f"Review of: {code}"}

def coordinator(state):
    """Coordinate final output"""
    return {"final": "Task complete!"}

# Build multi-agent workflow
workflow = StateGraph(MultiAgentState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("reviewer", reviewer_agent)
workflow.add_node("coordinator", coordinator)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_edge("reviewer", "coordinator")
workflow.add_edge("coordinator", END)

app = workflow.compile()
```

## Features

### State Management
- Persistent state across nodes
- Type-safe state definitions
- State updates and merges

### Conditional Routing
```python
def should_continue(state):
    """Decide next step based on state"""
    if state["needs_more_research"]:
        return "research"
    else:
        return "finish"

workflow.add_conditional_edges(
    "analyze",
    should_continue,
    {
        "research": "search",
        "finish": END
    }
)
```

### Loops and Iteration
```python
# Add loop for iterative refinement
workflow.add_edge("refine", "analyze")  # Creates cycle
```

### Human-in-the-Loop
```python
def approval_node(state):
    """Wait for human approval"""
    print(f"Plan: {state['plan']}")
    approval = input("Approve? (yes/no): ")
    
    if approval.lower() == "yes":
        return {"approved": True}
    else:
        return {"approved": False, "needs_revision": True}
```

## Running the Projects

### Simple Agent
```bash
python simple_agent.py
```

### Research Agent
```bash
python research_agent.py "What are the latest developments in quantum computing?"
```

### Multi-Agent System
```bash
python multi_agent.py "Build a web scraper"
```

## Use Cases

1. **Complex Workflows**: Multi-step processes with dependencies
2. **Research Assistants**: Search, analyze, synthesize
3. **Code Generation**: Plan, implement, test, refine
4. **Content Creation**: Research, draft, review, publish
5. **Customer Support**: Route, analyze, respond, escalate

## Key Concepts

✓ State graphs and nodes  
✓ Conditional branching  
✓ Cyclic workflows  
✓ Multi-agent coordination  
✓ State persistence  
✓ Human feedback integration  

## Advanced Patterns

### 1. Retry Logic
```python
def should_retry(state):
    if state["attempts"] < 3 and state["failed"]:
        return "retry"
    return "end"
```

### 2. Parallel Execution
```python
# Multiple agents run in parallel
workflow.add_edge("start", "agent1")
workflow.add_edge("start", "agent2")
workflow.add_edge("start", "agent3")
```

### 3. State Checkpointing
```python
# Save state at key points
from langgraph.checkpoint import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

## Common Issues

### Issue: State not updating
**Solution**: Ensure node returns state updates as dict

### Issue: Infinite loops
**Solution**: Add max_iterations or proper exit conditions

### Issue: Complex state management
**Solution**: Use TypedDict for type safety

## Next Steps

- Build your own agent workflow
- Add tools to agents
- Implement streaming updates
- Create web interface
- Deploy to production

## Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/)
- [Agent Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

---

**Level**: Intermediate | **Time**: 1-2 days | **Difficulty**: ⭐⭐⭐⭐☆
