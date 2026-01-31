# Multi-Agent System with LangGraph

## Overview

Build a sophisticated multi-agent system using LangGraph where multiple specialized AI agents collaborate to solve complex tasks. This demonstrates advanced agent coordination, state management, and workflow orchestration.

## What You'll Learn

- **LangGraph Architecture**: Building stateful agent systems
- **Multi-Agent Coordination**: Agents working together
- **State Management**: Shared state across agents
- **Agent Communication**: Message passing and coordination
- **Specialized Agents**: Creating agents with specific roles
- **Workflow Orchestration**: Managing complex agent workflows
- **Human-in-the-Loop**: Adding human oversight and approval

## Prerequisites

- Completed intermediate projects
- Strong understanding of LangChain agents
- Familiarity with state machines and graphs

## Project Structure

```
multi_agent_system/
├── README.md                  # This file
├── basic_multi_agent.py       # Simple multi-agent example
├── research_team.py           # Research team simulation
├── customer_support.py        # Multi-agent customer support
├── workflow_orchestrator.py   # Complex workflow coordination
└── utils.py                   # Helper functions
```

## Architecture

### Agent Roles

```
┌─────────────────────────────────────────┐
│         Supervisor Agent                │
│   (Coordinates other agents)            │
└──────────┬──────────────────────────────┘
           │
    ┌──────┴──────┬──────────┬──────────┐
    │             │          │          │
┌───▼────┐  ┌────▼───┐  ┌──▼─────┐  ┌─▼──────┐
│Research│  │Writer  │  │Critic  │  │Editor  │
│Agent   │  │Agent   │  │Agent   │  │Agent   │
└────────┘  └────────┘  └────────┘  └────────┘
```

### State Flow

```
Initialize → Research → Write → Critique → Edit → Approve → Done
     ↑                              │
     └──────── Iterate ─────────────┘
```

## Quick Start

```bash
python basic_multi_agent.py
```

## Implementation Guide

### Step 1: Define Agent State

```python
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    task: str
    result: str
    iterations: int
```

### Step 2: Create Specialized Agents

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate

def create_research_agent():
    """Agent specialized in research"""
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research specialist. Your job is to gather information on the given topic."),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    # Add research tools
    tools = [...]  # Search tools, web scraping, etc.
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def create_writer_agent():
    """Agent specialized in writing"""
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional writer. Create well-structured, engaging content."),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    agent = create_openai_functions_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[])

def create_critic_agent():
    """Agent specialized in critique"""
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a critical reviewer. Evaluate content quality and suggest improvements."),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    agent = create_openai_functions_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[])
```

### Step 3: Create LangGraph Workflow

```python
from langgraph.graph import StateGraph, END

def create_workflow():
    """Create multi-agent workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("research", research_node)
    workflow.add_node("write", write_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("edit", edit_node)
    
    # Define edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "critique")
    
    # Conditional edge for iteration
    workflow.add_conditional_edges(
        "critique",
        should_iterate,
        {
            "edit": "edit",
            "done": END
        }
    )
    workflow.add_edge("edit", "write")
    
    return workflow.compile()

def should_iterate(state: AgentState):
    """Decide whether to iterate or finish"""
    if state["iterations"] >= 3:
        return "done"
    # Check quality score
    if quality_check(state["result"]):
        return "done"
    return "edit"
```

### Step 4: Implement Agent Nodes

```python
def research_node(state: AgentState):
    """Execute research agent"""
    agent = create_research_agent()
    result = agent.invoke({"input": state["task"]})
    
    return {
        "messages": [result],
        "current_agent": "research",
        "result": result["output"]
    }

def write_node(state: AgentState):
    """Execute writer agent"""
    agent = create_writer_agent()
    
    # Use research results
    context = state["result"]
    prompt = f"Based on this research: {context}\n\nWrite an article about: {state['task']}"
    
    result = agent.invoke({"input": prompt})
    
    return {
        "messages": [result],
        "current_agent": "writer",
        "result": result["output"]
    }

def critique_node(state: AgentState):
    """Execute critic agent"""
    agent = create_critic_agent()
    
    content = state["result"]
    result = agent.invoke({"input": f"Review this content: {content}"})
    
    return {
        "messages": [result],
        "current_agent": "critic",
        "result": result["output"],
        "iterations": state.get("iterations", 0) + 1
    }
```

### Step 5: Run the System

```python
def run_multi_agent_system(task):
    """Run the multi-agent workflow"""
    
    # Initialize state
    initial_state = {
        "messages": [],
        "current_agent": "",
        "task": task,
        "result": "",
        "iterations": 0
    }
    
    # Create and run workflow
    workflow = create_workflow()
    
    print("Starting multi-agent system...")
    for output in workflow.stream(initial_state):
        for key, value in output.items():
            print(f"\n[{key}]: {value['current_agent']}")
            print(f"Result: {value['result'][:100]}...")
    
    return output
```

## Advanced Patterns

### 1. Supervisor Pattern

```python
def create_supervisor():
    """Supervisor agent that delegates to other agents"""
    
    system_prompt = """You are a supervisor managing a team of agents: {agents}.
    Given a task, decide which agent should act next.
    Each agent has specific capabilities:
    - Researcher: Gathers information
    - Writer: Creates content
    - Analyst: Analyzes data
    
    Respond with the name of the agent to use next."""
    
    # Implementation...
```

### 2. Parallel Agent Execution

```python
from langgraph.graph import parallel

workflow.add_node("parallel_research", parallel(
    research_agent_1,
    research_agent_2,
    research_agent_3
))
```

### 3. Human-in-the-Loop

```python
from langgraph.checkpoint import MemorySaver

def approval_node(state: AgentState):
    """Request human approval"""
    print(f"\nResult: {state['result']}")
    approval = input("Approve? (yes/no): ")
    
    if approval.lower() == "yes":
        state["approved"] = True
    else:
        state["approved"] = False
        state["feedback"] = input("Feedback: ")
    
    return state

# Add to workflow
workflow.add_node("approval", approval_node)
workflow.add_conditional_edges(
    "approval",
    lambda s: "done" if s["approved"] else "edit"
)
```

### 4. Dynamic Agent Selection

```python
def route_to_agent(state: AgentState):
    """Dynamically select next agent"""
    
    task_type = classify_task(state["task"])
    
    routing = {
        "research": "research_agent",
        "analysis": "analyst_agent",
        "creative": "writer_agent",
        "technical": "technical_agent"
    }
    
    return routing.get(task_type, "general_agent")
```

## Use Cases

### 1. Research Team

```python
# Team: Researcher → Summarizer → Fact Checker → Publisher
team = create_research_team([
    "researcher",
    "summarizer", 
    "fact_checker",
    "publisher"
])

result = team.run("Research the latest AI developments")
```

### 2. Customer Support

```python
# Team: Classifier → Specialist → Quality Checker → Responder
support_team = create_support_team([
    "classifier",      # Classifies issue type
    "technical_agent", # Handles technical issues
    "billing_agent",   # Handles billing issues
    "general_agent",   # Handles general queries
    "qa_agent"         # Checks response quality
])
```

### 3. Content Creation

```python
# Team: Planner → Researcher → Writer → Editor → SEO Specialist
content_team = create_content_team([
    "planner",
    "researcher",
    "writer",
    "editor",
    "seo_specialist"
])
```

## Best Practices

### 1. State Management

```python
# Keep state minimal and focused
class MinimalState(TypedDict):
    task: str
    context: dict
    current_step: str
    output: str
```

### 2. Error Handling

```python
def safe_agent_node(agent_func):
    """Wrapper for error handling"""
    def wrapper(state):
        try:
            return agent_func(state)
        except Exception as e:
            return {
                "error": str(e),
                "current_agent": "error_handler"
            }
    return wrapper
```

### 3. Monitoring

```python
def log_agent_action(state):
    """Log agent actions"""
    print(f"[{state['current_agent']}] at {datetime.now()}")
    print(f"Task: {state['task'][:50]}...")
```

### 4. Testing

```python
def test_agent_workflow():
    """Test individual agents and workflow"""
    
    # Test each agent
    test_state = {"task": "test", ...}
    
    assert research_node(test_state) is not None
    assert write_node(test_state) is not None
    
    # Test full workflow
    result = run_workflow(test_state)
    assert result["result"] != ""
```

## Performance Optimization

### 1. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_agent_call(agent_name, task):
    """Cache agent results"""
    return agents[agent_name].invoke(task)
```

### 2. Parallel Execution

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_agent_execution(tasks):
    """Execute multiple agents in parallel"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(agent.run, task) for task in tasks]
        return [f.result() for f in futures]
```

### 3. Streaming

```python
async def stream_agent_output(agent, task):
    """Stream agent output in real-time"""
    async for chunk in agent.astream(task):
        print(chunk, end="", flush=True)
```

## Troubleshooting

### Issue: Agents stuck in loop
**Solution**: Add iteration limits and exit conditions

```python
MAX_ITERATIONS = 5

def check_iterations(state):
    if state["iterations"] >= MAX_ITERATIONS:
        return END
```

### Issue: State conflicts
**Solution**: Use proper state merging

```python
def merge_state(old_state, new_state):
    """Safely merge states"""
    return {**old_state, **new_state}
```

### Issue: High costs
**Solution**: Use cheaper models for intermediate steps

```python
# Use GPT-3.5 for simple tasks
simple_llm = ChatOpenAI(model="gpt-3.5-turbo")
# Use GPT-4 only for critical tasks
advanced_llm = ChatOpenAI(model="gpt-4")
```

## Next Steps

1. Add more specialized agents
2. Implement agent memory
3. Create web interface
4. Add monitoring and analytics
5. Deploy to production

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Multi-Agent Systems](https://python.langchain.com/docs/use_cases/agent_simulations)
- [State Management](https://langchain-ai.github.io/langgraph/concepts/low_level/)

## Related Projects

- [LangGraph Agent](../../intermediate/langgraph_agent/) - Basic LangGraph
- [Research Assistant](../research_assistant/) - Specialized application
- [Automated Workflow](../automated_workflow/) - Production workflows

---

**Next**: Build the [Research Assistant](../research_assistant/) for autonomous research!
