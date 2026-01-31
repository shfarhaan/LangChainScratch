# Research Assistant with LangGraph

## Overview

Build an autonomous research assistant that can plan research tasks, gather information from multiple sources, analyze findings, and produce comprehensive research reports. Uses LangGraph for complex multi-step workflows.

## What You'll Learn

- **Autonomous Research**: Self-directed information gathering
- **Multi-Source Integration**: Combining web, papers, and documents
- **Research Planning**: Breaking down complex research goals
- **Information Synthesis**: Analyzing and combining findings
- **Report Generation**: Creating structured research outputs
- **Iterative Refinement**: Improving research through iterations
- **Tool Integration**: Search, scraping, and analysis tools

## Prerequisites

- Completed [Multi-Agent System](../multi_agent_system/) project
- Understanding of research methodologies
- Familiarity with search tools and APIs

## Project Structure

```
research_assistant/
├── README.md                # This file
├── research_agent.py        # Main research assistant
├── research_planner.py      # Research planning agent
├── research_tools.py        # Custom research tools
├── report_generator.py      # Report generation
└── examples/                # Example research tasks
```

## Architecture

### Research Workflow

```
┌──────────────┐
│  Plan Task   │  ← Break down research question
└──────┬───────┘
       ↓
┌──────────────┐
│ Web Search   │  ← Search multiple sources
└──────┬───────┘
       ↓
┌──────────────┐
│  Analysis    │  ← Analyze and synthesize
└──────┬───────┘
       ↓
┌──────────────┐
│   Critique   │  ← Evaluate completeness
└──────┬───────┘
       ↓
┌──────────────┐
│Generate Report│ ← Create final output
└──────────────┘
```

## Quick Start

```bash
python research_agent.py "Impact of AI on climate change"
```

## Implementation Guide

### Step 1: Research Planning

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def create_research_plan(query):
    """Break down research query into sub-tasks"""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research planner. Break down the research query into:
1. Key questions to answer
2. Information sources to check
3. Analysis needed
4. Expected deliverables"""),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": query})
    
    return parse_research_plan(response.content)
```

### Step 2: Information Gathering

```python
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader

def gather_information(sub_query, num_sources=5):
    """Search and gather information"""
    
    # Web search
    search = DuckDuckGoSearchResults()
    search_results = search.run(sub_query)
    
    # Extract URLs and load content
    urls = extract_urls(search_results)[:num_sources]
    
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
        except:
            continue
    
    return documents
```

### Step 3: Analysis and Synthesis

```python
def analyze_research_findings(documents, query):
    """Analyze gathered information"""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Combine document contents
    context = "\n\n".join([doc.page_content[:1000] for doc in documents])
    
    prompt = f"""Analyze these research findings about: {query}

Sources:
{context}

Provide:
1. Key findings
2. Common themes
3. Contradictions or debates
4. Gaps in information"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
```

### Step 4: Report Generation

```python
def generate_research_report(query, findings, analysis):
    """Generate comprehensive research report"""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    template = """Create a comprehensive research report:

Topic: {query}

Key Findings:
{findings}

Analysis:
{analysis}

Generate a well-structured report with:
1. Executive Summary
2. Main Findings
3. Detailed Analysis
4. Conclusions
5. References"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    return chain.invoke({
        "query": query,
        "findings": findings,
        "analysis": analysis
    })
```

### Step 5: Complete Research Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ResearchState(TypedDict):
    query: str
    plan: dict
    findings: list
    analysis: str
    report: str
    iteration: int

def create_research_workflow():
    """Create research assistant workflow"""
    
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("plan", planning_node)
    workflow.add_node("search", search_node)
    workflow.add_node("analyze", analysis_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("report", report_node)
    
    # Define flow
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "analyze")
    workflow.add_edge("analyze", "critique")
    
    # Conditional: iterate or finalize
    workflow.add_conditional_edges(
        "critique",
        should_continue_research,
        {
            "search": "search",  # Need more info
            "report": "report",   # Ready for report
        }
    )
    
    workflow.add_edge("report", END)
    
    return workflow.compile()
```

## Research Tools

### 1. Web Search Tool

```python
from langchain.tools import Tool
from langchain.utilities import DuckDuckGoSearchAPIWrapper

def create_search_tool():
    search = DuckDuckGoSearchAPIWrapper()
    
    return Tool(
        name="web_search",
        description="Search the web for information",
        func=search.run
    )
```

### 2. Document Analyzer

```python
def analyze_document(document, aspect):
    """Analyze specific aspect of document"""
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    prompt = f"""Analyze this document focusing on: {aspect}

Document:
{document}

Analysis:"""
    
    return llm.invoke([HumanMessage(content=prompt)]).content
```

### 3. Citation Extractor

```python
def extract_citations(documents):
    """Extract and format citations"""
    
    citations = []
    for i, doc in enumerate(documents, 1):
        citation = {
            "id": i,
            "url": doc.metadata.get("source", "Unknown"),
            "title": extract_title(doc),
            "snippet": doc.page_content[:200]
        }
        citations.append(citation)
    
    return citations
```

## Use Cases

### 1. Academic Research

```python
research_query = "Latest developments in quantum computing for drug discovery"
assistant = create_research_assistant()
report = assistant.research(research_query)
```

### 2. Market Research

```python
research_query = "Market trends in electric vehicle adoption in Asia"
assistant = create_research_assistant(
    sources=["news", "reports", "statistics"]
)
report = assistant.research(research_query)
```

### 3. Competitive Analysis

```python
research_query = "Compare features of top 5 project management tools"
assistant = create_research_assistant(
    analysis_type="comparative"
)
report = assistant.research(research_query)
```

### 4. Literature Review

```python
research_query = "Machine learning applications in healthcare 2020-2024"
assistant = create_research_assistant(
    sources=["arxiv", "pubmed", "scholar"]
)
report = assistant.research(research_query)
```

## Advanced Features

### 1. Multi-Source Integration

```python
def gather_from_multiple_sources(query):
    """Gather from web, papers, and documents"""
    
    sources = {
        "web": search_web(query),
        "papers": search_arxiv(query),
        "news": search_news(query)
    }
    
    return sources
```

### 2. Fact Checking

```python
def verify_facts(claims):
    """Verify factual claims"""
    
    verified = []
    for claim in claims:
        sources = search_for_verification(claim)
        confidence = calculate_confidence(sources)
        verified.append({
            "claim": claim,
            "verified": confidence > 0.7,
            "confidence": confidence,
            "sources": sources
        })
    
    return verified
```

### 3. Research Memory

```python
from langchain.memory import VectorStoreRetrieverMemory

def create_research_memory():
    """Maintain research context across sessions"""
    
    memory = VectorStoreRetrieverMemory(
        retriever=vectorstore.as_retriever()
    )
    
    return memory
```

### 4. Progressive Summarization

```python
def progressive_summarization(documents):
    """Progressively summarize large amounts of information"""
    
    summaries = []
    
    # First pass: summarize each document
    for doc in documents:
        summary = summarize_document(doc)
        summaries.append(summary)
    
    # Second pass: combine summaries
    combined = combine_summaries(summaries)
    
    # Final pass: create executive summary
    final = create_executive_summary(combined)
    
    return final
```

## Best Practices

### 1. Source Diversity

```python
# Use multiple search engines and databases
sources = [
    DuckDuckGoSearch(),
    GoogleScholarSearch(),
    ArxivSearch(),
    NewsSearch()
]
```

### 2. Quality Control

```python
def assess_source_quality(source):
    """Assess source credibility"""
    
    factors = {
        "domain_authority": check_domain(source.url),
        "recency": check_date(source.date),
        "relevance": calculate_relevance(source.content),
        "citations": count_citations(source)
    }
    
    return calculate_quality_score(factors)
```

### 3. Iterative Refinement

```python
def should_continue_research(state):
    """Decide if more research is needed"""
    
    completeness = assess_completeness(state["findings"])
    quality = assess_quality(state["analysis"])
    
    if completeness > 0.8 and quality > 0.7:
        return "report"
    
    if state["iteration"] >= 3:
        return "report"
    
    return "search"
```

### 4. Cost Management

```python
# Use cheaper models for initial searches
search_llm = ChatOpenAI(model="gpt-3.5-turbo")

# Use advanced models for final analysis
analysis_llm = ChatOpenAI(model="gpt-4")
```

## Performance Tips

1. **Parallel Searches**: Run multiple searches concurrently
2. **Caching**: Cache search results and summaries
3. **Streaming**: Stream long report generation
4. **Batching**: Process documents in batches
5. **Pruning**: Remove low-quality sources early

## Troubleshooting

### Issue: Incomplete research
**Solution**: Adjust completeness threshold

### Issue: Too many sources
**Solution**: Implement relevance scoring

### Issue: Conflicting information
**Solution**: Add fact verification step

### Issue: Long execution time
**Solution**: Set time limits per step

## Next Steps

1. Add more data sources
2. Implement citation management
3. Create interactive UI
4. Add export formats (PDF, DOCX)
5. Integrate with note-taking apps

## Resources

- [LangChain Research Use Cases](https://python.langchain.com/docs/use_cases/)
- [Research Automation Best Practices](https://blog.langchain.dev/)

## Related Projects

- [Multi-Agent System](../multi_agent_system/) - Agent coordination
- [Document Q&A](../../intermediate/document_qa/) - RAG basics
- [Web Scraper](../../intermediate/web_scraper_chatbot/) - Web scraping

---

**Next**: Build the [Automated Workflow](../automated_workflow/) for production automation!
