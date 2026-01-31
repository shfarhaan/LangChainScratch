"""
Research Assistant
==================

Autonomous research agent that plans, gathers information, and generates reports.

Usage:
    python research_agent.py "your research query"
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv
import sys
import os

load_dotenv()


def plan_research(query):
    """Create research plan"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = f"""Create a research plan for: {query}

Break this into 3-5 specific sub-questions that need to be answered.
Format as a numbered list."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def search_information(sub_query):
    """Search for information"""
    try:
        search = DuckDuckGoSearchResults()
        results = search.run(sub_query)
        return results
    except Exception as e:
        return f"Search error: {e}"


def analyze_findings(query, all_findings):
    """Analyze research findings"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    combined = "\n\n".join(all_findings)
    
    prompt = f"""Analyze these research findings about: {query}

Findings:
{combined[:3000]}

Provide:
1. Key insights
2. Main themes
3. Important facts"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def generate_report(query, plan, findings, analysis):
    """Generate final research report"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    prompt = f"""Generate a comprehensive research report:

Topic: {query}

Research Plan:
{plan}

Analysis:
{analysis}

Create a well-structured report with:
- Executive Summary
- Key Findings
- Detailed Analysis
- Conclusion"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return
    
    print("\n" + "="*70)
    print("Research Assistant")
    print("="*70 + "\n")
    
    # Get research query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("What would you like to research? ").strip()
    
    if not query:
        query = "Benefits of renewable energy"
        print(f"Using default query: {query}\n")
    
    print(f"Research Query: {query}\n")
    
    # Step 1: Plan
    print("[1/4] Planning research...")
    plan = plan_research(query)
    print(f"\nResearch Plan:\n{plan}\n")
    
    # Step 2: Gather information
    print("[2/4] Gathering information...")
    
    # Extract sub-questions from plan
    lines = plan.split('\n')
    sub_queries = [line.strip('0123456789. -') for line in lines if line.strip() and any(c.isdigit() for c in line[:5])][:3]
    
    all_findings = []
    for i, sub_query in enumerate(sub_queries, 1):
        print(f"  Searching: {sub_query[:60]}...")
        findings = search_information(sub_query)
        all_findings.append(f"Query {i}: {sub_query}\nFindings: {findings}")
    
    print(f"\nGathered information from {len(sub_queries)} searches")
    
    # Step 3: Analyze
    print("\n[3/4] Analyzing findings...")
    analysis = analyze_findings(query, all_findings)
    print(f"\nAnalysis:\n{analysis}\n")
    
    # Step 4: Generate report
    print("[4/4] Generating report...\n")
    report = generate_report(query, plan, all_findings, analysis)
    
    # Display final report
    print("\n" + "="*70)
    print("RESEARCH REPORT")
    print("="*70 + "\n")
    print(report)
    print("\n" + "="*70 + "\n")
    
    # Save report
    filename = f"research_report_{query[:30].replace(' ', '_')}.txt"
    with open(filename, 'w') as f:
        f.write(f"Research Report: {query}\n\n")
        f.write("="*70 + "\n\n")
        f.write(report)
    
    print(f"Report saved to: {filename}\n")


if __name__ == "__main__":
    main()
