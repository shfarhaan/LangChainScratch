# Automated Workflow Engine

## Overview

Build a production-ready automated workflow engine using LangGraph for end-to-end task automation. This system can handle complex business processes, data pipelines, and automated decision-making with state management and error handling.

## What You'll Learn

- **Workflow Orchestration**: Managing complex multi-step processes
- **State Persistence**: Durable state across workflow executions
- **Error Recovery**: Handling failures gracefully
- **Parallel Execution**: Running tasks concurrently
- **Conditional Logic**: Dynamic workflow routing
- **Monitoring & Logging**: Production-grade observability
- **API Integration**: Connecting to external systems
- **Scheduling**: Time-based workflow triggers

## Prerequisites

- Completed all previous projects
- Understanding of production systems
- Familiarity with workflow engines

## Project Structure

```
automated_workflow/
├── README.md                # This file
├── workflow_engine.py       # Main workflow engine
├── workflow_builder.py      # Workflow definition builder
├── tasks/                   # Task definitions
│   ├── data_processing.py
│   ├── api_calls.py
│   └── notifications.py
├── workflows/               # Workflow definitions
│   ├── data_pipeline.py
│   ├── report_generation.py
│   └── customer_onboarding.py
└── utils.py                 # Helper functions
```

## Architecture

### Workflow Engine

```
┌─────────────────────────────────────┐
│      Workflow Scheduler             │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│      Workflow Executor               │
│  ┌─────────────────────────────┐   │
│  │  State Manager              │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  Task Runner                │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  Error Handler              │   │
│  └─────────────────────────────┘   │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│      Monitoring & Logging           │
└─────────────────────────────────────┘
```

## Quick Start

```bash
python workflow_engine.py --workflow data_pipeline
```

## Implementation Guide

### Step 1: Define Workflow State

```python
from typing import TypedDict, List, Dict, Any
from datetime import datetime

class WorkflowState(TypedDict):
    workflow_id: str
    status: str  # pending, running, completed, failed
    current_step: str
    data: Dict[str, Any]
    results: List[Dict[str, Any]]
    errors: List[str]
    started_at: datetime
    completed_at: datetime
    metadata: Dict[str, Any]
```

### Step 2: Create Task Definitions

```python
from langchain.tools import BaseTool

class DataProcessingTask(BaseTool):
    name = "process_data"
    description = "Process raw data"
    
    def _run(self, data: Dict) -> Dict:
        """Process data"""
        try:
            # Data processing logic
            processed = transform_data(data)
            return {
                "status": "success",
                "data": processed
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _arun(self, data: Dict) -> Dict:
        """Async version"""
        return self._run(data)


class APICallTask(BaseTool):
    name = "api_call"
    description = "Make API call"
    
    def _run(self, endpoint: str, payload: Dict) -> Dict:
        """Call external API"""
        try:
            response = requests.post(endpoint, json=payload)
            return {
                "status": "success",
                "response": response.json()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
```

### Step 3: Build Workflow

```python
from langgraph.graph import StateGraph, END

def create_data_pipeline():
    """Create data processing pipeline workflow"""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("extract", extract_data)
    workflow.add_node("transform", transform_data)
    workflow.add_node("validate", validate_data)
    workflow.add_node("load", load_data)
    workflow.add_node("notify", send_notification)
    
    # Define flow
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "transform")
    workflow.add_edge("transform", "validate")
    
    # Conditional: if valid, load; else, notify error
    workflow.add_conditional_edges(
        "validate",
        lambda s: "load" if s["data"]["valid"] else "notify",
        {
            "load": "load",
            "notify": "notify"
        }
    )
    
    workflow.add_edge("load", "notify")
    workflow.add_edge("notify", END)
    
    return workflow.compile()
```

### Step 4: Add Error Handling

```python
def safe_node_execution(node_func):
    """Wrapper for safe node execution with retry"""
    
    def wrapper(state: WorkflowState):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                result = node_func(state)
                return result
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    state["errors"].append(f"Node failed: {str(e)}")
                    state["status"] = "failed"
                    return state
                time.sleep(2 ** retry_count)  # Exponential backoff
        
        return state
    
    return wrapper
```

### Step 5: State Persistence

```python
from langgraph.checkpoint import SqliteSaver

def create_persistent_workflow():
    """Create workflow with state persistence"""
    
    # Create checkpointer
    checkpointer = SqliteSaver.from_conn_string("workflow_state.db")
    
    # Create workflow
    workflow = create_data_pipeline()
    
    # Compile with checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    
    return app
```

### Step 6: Workflow Executor

```python
class WorkflowExecutor:
    """Execute and monitor workflows"""
    
    def __init__(self):
        self.workflows = {}
        self.checkpointer = SqliteSaver.from_conn_string("workflows.db")
    
    def register_workflow(self, name: str, workflow_def):
        """Register a workflow definition"""
        self.workflows[name] = workflow_def
    
    def execute(self, workflow_name: str, input_data: Dict):
        """Execute a workflow"""
        
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        # Create workflow instance
        workflow = self.workflows[workflow_name]()
        
        # Initialize state
        state = {
            "workflow_id": generate_id(),
            "status": "running",
            "current_step": "",
            "data": input_data,
            "results": [],
            "errors": [],
            "started_at": datetime.now(),
            "metadata": {"workflow_name": workflow_name}
        }
        
        # Execute
        try:
            for output in workflow.stream(state):
                self.log_progress(output)
            
            return {
                "status": "completed",
                "results": output
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def log_progress(self, output):
        """Log workflow progress"""
        print(f"[{datetime.now()}] {output.get('current_step', 'unknown')}")
```

## Use Cases

### 1. Data Pipeline

```python
def create_etl_pipeline():
    """Extract, Transform, Load pipeline"""
    
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("extract_from_api", extract_api)
    workflow.add_node("extract_from_db", extract_db)
    workflow.add_node("merge_data", merge)
    workflow.add_node("transform", transform)
    workflow.add_node("load_to_warehouse", load)
    
    # Parallel extraction
    workflow.set_entry_point("extract_from_api")
    workflow.add_edge("extract_from_api", "merge_data")
    workflow.add_edge("extract_from_db", "merge_data")
    workflow.add_edge("merge_data", "transform")
    workflow.add_edge("transform", "load_to_warehouse")
    workflow.add_edge("load_to_warehouse", END)
    
    return workflow.compile()
```

### 2. Report Generation

```python
def create_report_workflow():
    """Automated report generation"""
    
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("gather_data", gather_data)
    workflow.add_node("analyze", analyze_data)
    workflow.add_node("generate_charts", create_charts)
    workflow.add_node("generate_text", create_text)
    workflow.add_node("compile_report", compile_report)
    workflow.add_node("send_email", send_email)
    
    return workflow.compile()
```

### 3. Customer Onboarding

```python
def create_onboarding_workflow():
    """Automated customer onboarding"""
    
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("verify_identity", verify_identity)
    workflow.add_node("create_account", create_account)
    workflow.add_node("setup_services", setup_services)
    workflow.add_node("send_welcome", send_welcome)
    workflow.add_node("schedule_followup", schedule_followup)
    
    return workflow.compile()
```

## Advanced Features

### 1. Parallel Execution

```python
from langgraph.graph import parallel

workflow.add_node("parallel_tasks", parallel(
    task1,
    task2,
    task3
))
```

### 2. Scheduling

```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(
    func=execute_workflow,
    trigger='cron',
    hour=9,
    args=['daily_report']
)
scheduler.start()
```

### 3. Monitoring

```python
from prometheus_client import Counter, Histogram

workflow_executions = Counter('workflow_executions_total', 'Total workflows')
workflow_duration = Histogram('workflow_duration_seconds', 'Workflow duration')

@workflow_duration.time()
def execute_workflow(workflow_name):
    workflow_executions.inc()
    # Execute workflow
```

### 4. Event-Driven Triggers

```python
class EventTrigger:
    """Trigger workflows based on events"""
    
    def __init__(self, executor):
        self.executor = executor
        self.listeners = {}
    
    def on(self, event_type: str, workflow_name: str):
        """Register workflow for event"""
        self.listeners[event_type] = workflow_name
    
    def trigger(self, event_type: str, data: Dict):
        """Trigger workflow on event"""
        if event_type in self.listeners:
            workflow = self.listeners[event_type]
            self.executor.execute(workflow, data)
```

## Production Best Practices

### 1. Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Configuration

```python
from pydantic import BaseSettings

class WorkflowConfig(BaseSettings):
    max_retries: int = 3
    timeout: int = 300
    log_level: str = "INFO"
    database_url: str
    
    class Config:
        env_file = ".env"
```

### 3. Testing

```python
def test_workflow():
    """Test workflow execution"""
    
    test_state = {
        "data": {"test": "data"},
        "results": [],
        "errors": []
    }
    
    workflow = create_data_pipeline()
    result = workflow.invoke(test_state)
    
    assert result["status"] == "completed"
    assert len(result["errors"]) == 0
```

### 4. Monitoring Dashboard

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/workflows/<workflow_id>/status')
def get_workflow_status(workflow_id):
    status = get_status_from_db(workflow_id)
    return jsonify(status)

@app.route('/workflows/metrics')
def get_metrics():
    metrics = calculate_metrics()
    return jsonify(metrics)
```

## Next Steps

1. Add more workflow templates
2. Implement web UI dashboard
3. Add workflow versioning
4. Create workflow marketplace
5. Deploy to cloud platform

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Workflow Patterns](https://www.workflowpatterns.com/)
- [Production LLM Apps](https://blog.langchain.dev/)

## Related Projects

- [Multi-Agent System](../multi_agent_system/) - Agent coordination
- [Research Assistant](../research_assistant/) - Autonomous workflows

---

**Congratulations!** You've completed all 10 projects in the LangChain learning path! 🎉
