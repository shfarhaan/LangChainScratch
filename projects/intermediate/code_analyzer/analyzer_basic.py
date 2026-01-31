"""
Code Analyzer - Basic
======================

Analyze code using AI to provide explanations, find bugs, and suggest improvements.

Usage:
    python analyzer_basic.py <file_path>
    python analyzer_basic.py examples/example.py
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import sys

load_dotenv()


def read_code_file(file_path):
    """Read code from file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def analyze_code(code, analysis_type="explain"):
    """Analyze code using LLM"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    templates = {
        "explain": """Explain the following code in a clear and detailed way.
Include:
1. What the code does
2. How it works
3. Key concepts used
4. Any notable patterns or techniques

Code:
{code}

Explanation:""",
        
        "bugs": """Analyze this code for potential bugs, errors, or issues.
List each issue with:
1. The problem
2. Why it's an issue
3. How to fix it

Code:
{code}

Analysis:""",
        
        "improve": """Suggest improvements for this code.
Include:
1. Code quality improvements
2. Performance optimizations
3. Better practices
4. Refactoring suggestions

Code:
{code}

Suggestions:""",
        
        "document": """Generate comprehensive documentation for this code.
Include:
1. Purpose and functionality
2. Parameters and return values
3. Usage examples
4. Any important notes

Code:
{code}

Documentation:"""
    }
    
    template = templates.get(analysis_type, templates["explain"])
    prompt = PromptTemplate(template=template, input_variables=["code"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run(code=code)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return
    
    print("\n" + "="*70)
    print("Code Analyzer")
    print("="*70 + "\n")
    
    # Get file path from command line or prompt
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter code file path: ").strip()
    
    if not file_path or not os.path.exists(file_path):
        print("File not found. Creating example...")
        
        # Create example directory and file
        os.makedirs("examples", exist_ok=True)
        example_code = '''def fibonacci(n):
    """Calculate the nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
'''
        example_path = "examples/example.py"
        with open(example_path, 'w') as f:
            f.write(example_code)
        
        print(f"Created {example_path}")
        file_path = example_path
    
    # Read code
    code = read_code_file(file_path)
    if not code:
        return
    
    print(f"Analyzing: {file_path}")
    print(f"Code length: {len(code)} characters\n")
    
    # Interactive analysis
    while True:
        print("\nAnalysis Options:")
        print("  1. Explain code")
        print("  2. Find bugs")
        print("  3. Suggest improvements")
        print("  4. Generate documentation")
        print("  5. Load new file")
        print("  6. Quit")
        
        choice = input("\nChoice (1-6): ").strip()
        
        if choice == '6' or choice.lower() in ['quit', 'exit']:
            break
        
        if choice == '5':
            file_path = input("Enter code file path: ").strip()
            code = read_code_file(file_path)
            if not code:
                print("Failed to load file")
                continue
            print(f"\nLoaded: {file_path}\n")
            continue
        
        analysis_map = {
            '1': 'explain',
            '2': 'bugs',
            '3': 'improve',
            '4': 'document'
        }
        
        analysis_type = analysis_map.get(choice)
        if not analysis_type:
            print("Invalid choice")
            continue
        
        print(f"\nAnalyzing...")
        result = analyze_code(code, analysis_type)
        
        print("\n" + "-"*70)
        print("Analysis Result:")
        print("-"*70)
        print(result)
        print("-"*70)
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
