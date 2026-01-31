# Introduction to LangGraph

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It extends LangChain's capabilities by adding explicit state management and support for cyclic workflows.

## Why LangGraph?

Traditional chains in LangChain are linear (A → B → C). LangGraph enables:

- **Cyclic Workflows**: Create loops for iterative refinement
- **State Management**: Explicit state that persists across steps
- **Conditional Logic**: Branch based on state or outputs
- **Multi-Agent Systems**: Coordinate multiple agents
- **Human-in-the-Loop**: Pause for human input or approval
- **Better Control**: Fine-grained control over execution flow

## Core Concepts

### 1. State

State is the data that flows through your graph. Define it with TypedDict:

```python
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]  # Accumulate messages
    current_step: str
    result: str
```

**Key Points**:
- Use `TypedDict` for type safety
- `Annotated[List, operator.add]` accumulates values
- State persists across all nodes

### 2. Nodes

Nodes are functions that process state:

```python
def my_node(state: AgentState) -> dict:
    """Process state and return updates"""
    messages = state["messages"]
    
    # Do work
    new_message = "Processed successfully"
    
    # Return state updates
    return {
        "messages": [new_message],
        "current_step": "next_step"
    }
```

**Key Points**:
- Nodes receive current state
- Return dict with state updates
- Updates are merged into existing state

### 3. Edges

Edges connect nodes:

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("step1", step1_node)
workflow.add_node("step2", step2_node)

# Add edges
workflow.add_edge("step1", "step2")  # step1 → step2
workflow.add_edge("step2", END)       # step2 → END
```

### 4. Conditional Edges

Branch based on state:

```python
def route_decision(state: AgentState) -> str:
    """Decide which path to take"""
    if state["needs_revision"]:
        return "revise"
    else:
        return "finish"

workflow.add_conditional_edges(
    "check",
    route_decision,
    {
        "revise": "step1",  # Loop back
        "finish": END
    }
)
```

## Building Your First Graph

### Step 1: Define State

```python
from typing import TypedDict

class MyState(TypedDict):
    input: str
    output: str
    step: str
```

### Step 2: Create Nodes

```python
from langchain_openai import ChatOpenAI

def process_node(state: MyState) -> dict:
    llm = ChatOpenAI()
    result = llm.invoke(state["input"])
    
    return {
        "output": result.content,
        "step": "complete"
    }
```

### Step 3: Build Graph

```python
from langgraph.graph import StateGraph, END

# Create graph
workflow = StateGraph(MyState)

# Add nodes
workflow.add_node("process", process_node)

# Set entry point
workflow.set_entry_point("process")

# Add edges
workflow.add_edge("process", END)

# Compile
app = workflow.compile()
```

### Step 4: Run

```python
result = app.invoke({
    "input": "Hello, LangGraph!",
    "output": "",
    "step": "start"
})

print(result["output"])
```

## Common Patterns

### Pattern 1: Sequential Processing

```python
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", "step3")
workflow.add_edge("step3", END)
```

### Pattern 2: Conditional Branching

```python
def should_continue(state):
    return "yes" if state["score"] > 0.8 else "no"

workflow.add_conditional_edges(
    "evaluate",
    should_continue,
    {
        "yes": "finish",
        "no": "retry"
    }
)
```

### Pattern 3: Loops (Iterative Refinement)

```python
def check_quality(state):
    if state["iterations"] < 3 and state["quality"] < 0.9:
        return "improve"
    return "done"

workflow.add_conditional_edges(
    "check",
    check_quality,
    {
        "improve": "generate",  # Loop back
        "done": END
    }
)
```

### Pattern 4: Parallel Execution

```python
# All three agents run from start
workflow.add_edge("start", "agent1")
workflow.add_edge("start", "agent2")
workflow.add_edge("start", "agent3")

# All converge to coordinator
workflow.add_edge("agent1", "coordinator")
workflow.add_edge("agent2", "coordinator")
workflow.add_edge("agent3", "coordinator")
```

## Advanced Features

### Streaming

```python
for chunk in app.stream(initial_state):
    print(chunk)
```

### Checkpointing

```python
from langgraph.checkpoint import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# State is saved automatically
result = app.invoke(state, config={"thread_id": "conversation_1"})

# Resume later with same thread_id
```

### Human-in-the-Loop

```python
def approval_node(state):
    print(f"Approve this action? {state['proposed_action']}")
    approval = input("(yes/no): ")
    
    return {
        "approved": approval.lower() == "yes"
    }

workflow.add_node("approval", approval_node)
```

## Comparison: Chains vs. LangGraph

| Feature | Chains | LangGraph |
|---------|--------|-----------|
| Flow | Linear | Cyclic |
| State | Implicit | Explicit |
| Branching | Limited | Full support |
| Loops | No | Yes |
| Debugging | Harder | Easier |
| Complexity | Simple tasks | Complex workflows |

## When to Use LangGraph

**Use LangGraph when you need**:
- ✅ Iterative refinement
- ✅ Complex decision logic
- ✅ Multi-agent coordination
- ✅ State persistence
- ✅ Human approval steps
- ✅ Long-running workflows

**Use Chains when you need**:
- ✅ Simple sequential operations
- ✅ Quick prototyping
- ✅ Linear workflows
- ✅ Minimal state management

## Example: Research Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class ResearchState(TypedDict):
    topic: str
    sources: List[str]
    summary: str
    iterations: int

def search(state):
    # Search for sources
    return {"sources": ["source1", "source2"]}

def analyze(state):
    # Analyze sources
    return {"summary": "Analysis complete"}

def should_continue(state):
    return "more" if state["iterations"] < 3 else "done"

# Build graph
workflow = StateGraph(ResearchState)
workflow.add_node("search", search)
workflow.add_node("analyze", analyze)

workflow.set_entry_point("search")
workflow.add_edge("search", "analyze")

workflow.add_conditional_edges(
    "analyze",
    should_continue,
    {
        "more": "search",  # Loop for more research
        "done": END
    }
)

app = workflow.compile()
```

## Best Practices

1. **Clear State Definitions**: Use TypedDict for type safety
2. **Small Nodes**: Keep node functions focused
3. **Explicit Returns**: Always return state updates
4. **Test Branches**: Test all conditional paths
5. **Limit Loops**: Add max_iterations to prevent infinite loops
6. **Document Flow**: Add comments explaining routing logic

## Common Issues

### Issue: State not updating
**Solution**: Ensure node returns dict with updates

### Issue: Infinite loops
**Solution**: Add iteration counter and exit condition

### Issue: Type errors
**Solution**: Use TypedDict and type hints

## Resources

- [LangGraph Docs](https://python.langchain.com/docs/langgraph)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

## Next Steps

- Build your first graph
- Try the [LangGraph Agent](../projects/intermediate/langgraph_agent/) project
- Experiment with conditional routing
- Create multi-agent systems

---

**Ready to build with LangGraph?** Check out the [LangGraph Agent Project](../projects/intermediate/langgraph_agent/)
