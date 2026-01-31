# Text Summarizer Project

## Overview

Build a text summarization tool that can condense long documents into concise summaries using LangChain and LLMs.

## What You'll Learn

- Document loading and processing
- Text splitting strategies
- Different summarization approaches
- Working with long documents
- Chain composition

## Prerequisites

- Completed [Getting Started](../../../docs/01-getting-started.md)
- Completed [Simple Chatbot](../simple_chatbot/) project
- Understanding of chains

## Project Files

```
text_summarizer/
├── README.md              # This file
├── summarizer_basic.py    # Basic summarization
├── summarizer_advanced.py # Advanced with chunking
└── sample_text.txt        # Sample document
```

## Approaches to Summarization

### 1. Simple Summarization (Short Texts)

For documents that fit within token limits:

```python
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def simple_summarize(text: str) -> str:
    """Summarize short text directly"""
    
    llm = OpenAI(temperature=0.3)  # Lower temperature for focused summaries
    
    template = """
    Summarize the following text in 3-5 bullet points.
    Focus on the main ideas and key information.
    
    Text: {text}
    
    Summary:
    """
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"text": text})
    
    return result["text"]

# Example usage
text = """
Artificial intelligence (AI) is transforming industries worldwide...
[Short text content]
"""

summary = simple_summarize(text)
print(summary)
```

### 2. Chunked Summarization (Long Texts)

For documents exceeding token limits:

```python
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

def chunk_summarize(text: str) -> str:
    """Summarize long text by chunking"""
    
    llm = OpenAI(temperature=0.3)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Use map-reduce summarization
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=True
    )
    
    summary = chain.invoke(docs)
    return summary["output_text"]
```

### 3. Refine Summarization

Iteratively refine summary by processing chunks:

```python
def refine_summarize(text: str) -> str:
    """Summarize using refine method"""
    
    llm = OpenAI(temperature=0.3)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Refine approach: each chunk refines previous summary
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine"
    )
    
    summary = chain.invoke(docs)
    return summary["output_text"]
```

## Implementation

### summarizer_basic.py

```python
"""
Basic Text Summarizer
====================
Simple summarization for short to medium texts.
"""

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import sys

load_dotenv()


class TextSummarizer:
    """Simple text summarization tool"""
    
    def __init__(self, summary_style="bullet"):
        """
        Initialize summarizer
        
        Args:
            summary_style: Style of summary (bullet, paragraph, key_points)
        """
        self.llm = OpenAI(temperature=0.3)
        self.style = summary_style
        self.templates = {
            "bullet": """
            Summarize the following text in 5-7 bullet points.
            Focus on the most important information.
            
            Text: {text}
            
            Bullet Point Summary:
            """,
            "paragraph": """
            Summarize the following text in a single concise paragraph.
            Capture the main ideas and conclusions.
            
            Text: {text}
            
            Paragraph Summary:
            """,
            "key_points": """
            Extract the 3 most important key points from the following text.
            
            Text: {text}
            
            Key Points:
            1.
            2.
            3.
            """
        }
    
    def summarize(self, text: str) -> str:
        """Generate summary"""
        
        template = self.templates.get(self.style, self.templates["bullet"])
        
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.invoke({"text": text})
        
        return result["text"]
    
    def summarize_file(self, filepath: str) -> str:
        """Summarize a text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.summarize(text)
        except FileNotFoundError:
            return f"Error: File '{filepath}' not found"
        except Exception as e:
            return f"Error reading file: {e}"


def main():
    """CLI interface"""
    print("\n" + "="*60)
    print("Text Summarizer")
    print("="*60)
    
    # Get summary style
    print("\nChoose summary style:")
    print("1. Bullet points (default)")
    print("2. Paragraph")
    print("3. Key points")
    
    choice = input("\nEnter choice (1-3) or press Enter for default: ").strip()
    
    style_map = {
        "1": "bullet",
        "2": "paragraph",
        "3": "key_points",
        "": "bullet"
    }
    
    style = style_map.get(choice, "bullet")
    summarizer = TextSummarizer(summary_style=style)
    
    print("\nEnter your text (type 'END' on a new line when done):")
    print("-" * 60)
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    
    text = "\n".join(lines)
    
    if not text.strip():
        print("No text provided. Exiting.")
        return
    
    print("\nGenerating summary...")
    summary = summarizer.summarize(text)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary)
    print("="*60)


if __name__ == "__main__":
    main()
```

### summarizer_advanced.py

```python
"""
Advanced Text Summarizer
========================
Handles long documents with chunking and different strategies.
"""

from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

load_dotenv()


class AdvancedSummarizer:
    """Advanced text summarization with multiple strategies"""
    
    def __init__(self, method="map_reduce"):
        """
        Initialize advanced summarizer
        
        Args:
            method: Summarization method (map_reduce, refine, stuff)
        """
        self.llm = OpenAI(temperature=0.3, max_tokens=500)
        self.method = method
    
    def summarize(self, text: str, max_chunk_size=3000) -> dict:
        """
        Summarize text with chunking
        
        Returns:
            dict with summary and metadata
        """
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        print(f"Split into {len(docs)} chunks")
        
        # Load summarization chain
        chain = load_summarize_chain(
            self.llm,
            chain_type=self.method,
            verbose=False
        )
        
        # Generate summary with token tracking
        with get_openai_callback() as cb:
            result = chain.invoke(docs)
            
            return {
                "summary": result["output_text"],
                "chunks": len(docs),
                "tokens": cb.total_tokens,
                "cost": cb.total_cost,
                "method": self.method
            }
    
    def summarize_file(self, filepath: str) -> dict:
        """Summarize a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.summarize(text)
        except Exception as e:
            return {"error": str(e)}


def compare_methods(text: str):
    """Compare different summarization methods"""
    methods = ["map_reduce", "refine", "stuff"]
    results = {}
    
    print("\nComparing summarization methods...")
    print("="*60)
    
    for method in methods:
        try:
            print(f"\nTesting {method}...")
            summarizer = AdvancedSummarizer(method=method)
            result = summarizer.summarize(text)
            results[method] = result
            
            print(f"✓ {method}: {result['chunks']} chunks, "
                  f"{result['tokens']} tokens, ${result['cost']:.4f}")
        except Exception as e:
            print(f"✗ {method}: {e}")
            results[method] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage
    sample_text = """
    [Long article or document text goes here...]
    """
    
    # Test different methods
    results = compare_methods(sample_text)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for method, result in results.items():
        if "error" not in result:
            print(f"\n{method.upper()}:")
            print("-"*60)
            print(result["summary"])
```

## Running the Summarizer

### Basic Version

```bash
python summarizer_basic.py
```

### Advanced Version

```bash
python summarizer_advanced.py
```

### From Code

```python
from summarizer_basic import TextSummarizer

summarizer = TextSummarizer(summary_style="bullet")
summary = summarizer.summarize("Your long text here...")
print(summary)
```

## Testing

### Test Case 1: Short Article

```python
text = """
Climate change is one of the most pressing challenges of our time.
Rising temperatures, extreme weather events, and sea level rise
are just some of the impacts we're seeing...
[Continue with a few paragraphs]
"""

summarizer = TextSummarizer()
print(summarizer.summarize(text))
```

### Test Case 2: Long Document

```python
# Read a long file
with open("long_article.txt") as f:
    text = f.read()

# Use advanced summarizer
from summarizer_advanced import AdvancedSummarizer

summarizer = AdvancedSummarizer(method="map_reduce")
result = summarizer.summarize(text)

print(f"Summary: {result['summary']}")
print(f"Cost: ${result['cost']:.4f}")
```

## Enhancements

1. **Multiple Output Formats**: JSON, Markdown, Plain Text
2. **Language Detection**: Summarize in multiple languages
3. **Keyword Extraction**: Include key terms in summary
4. **Summary Lengths**: Short, Medium, Long options
5. **Web Integration**: Summarize web pages
6. **PDF Support**: Summarize PDF documents
7. **Batch Processing**: Summarize multiple files
8. **GUI**: Build a Streamlit interface

## Key Concepts Learned

✓ Document loading and processing  
✓ Text splitting strategies  
✓ Different summarization methods (map-reduce, refine, stuff)  
✓ Token management for long documents  
✓ Chain composition and customization  
✓ Error handling in document processing  

## Common Issues

### Issue: "Token limit exceeded"
**Solution**: Reduce chunk_size or use map_reduce method

### Issue: Summary is too long/short
**Solution**: Adjust prompts or max_tokens parameter

### Issue: Loss of context
**Solution**: Increase chunk_overlap

## Next Steps

- Try different summarization methods
- Experiment with chunk sizes
- Add support for different file formats
- Build a web interface
- Move on to [Q&A System](../qa_system/) project

## Resources

- [LangChain Summarization](https://python.langchain.com/docs/use_cases/summarization)
- [Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Chains Documentation](https://python.langchain.com/docs/modules/chains/)

---

**Level**: Beginner | **Time**: 3-4 hours | **Difficulty**: ⭐⭐☆☆☆
