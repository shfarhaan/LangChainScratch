# Q&A System with RAG

## Overview

Build an intelligent question-answering system that can answer questions about your documents using Retrieval-Augmented Generation (RAG).

## What You'll Learn

- Document loading and processing
- Text embeddings and vector stores
- Semantic search and retrieval
- RAG (Retrieval-Augmented Generation)
- Building Q&A interfaces

## Prerequisites

- Completed [Getting Started](../../../docs/01-getting-started.md)
- Completed [Simple Chatbot](../simple_chatbot/) project
- Understanding of embeddings concept

## Project Files

```
qa_system/
├── README.md           # This file
├── qa_basic.py         # Basic Q&A implementation
├── qa_advanced.py      # Advanced with conversation
└── sample_data.txt     # Sample document
```

## How It Works

### RAG Architecture

1. **Document Loading**: Load and split documents into chunks
2. **Embedding**: Convert text chunks into vector representations
3. **Storage**: Store embeddings in a vector database
4. **Retrieval**: Find relevant chunks for a query
5. **Generation**: Use LLM to generate answer from retrieved context

### Implementation

The system uses:
- **ChromaDB** for vector storage
- **OpenAI Embeddings** for text vectorization
- **RetrievalQA** chain for question answering

## Quick Start

### 1. Prepare Your Documents

Create a text file with content you want to query:

```bash
echo "LangChain is a framework for developing applications powered by language models..." > my_document.txt
```

### 2. Run Basic Q&A

```bash
python qa_basic.py
```

### 3. Ask Questions

```
Enter your document path: my_document.txt
Loading and processing document...
✓ Document loaded successfully!

Ask a question (or 'quit' to exit): What is LangChain?
Answer: LangChain is a framework for developing applications powered by language models...

Sources: Chunk 1, Chunk 3
```

## Implementation

### qa_basic.py

```python
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


class QASystem:
    """Basic Q&A system with RAG"""
    
    def __init__(self, document_path: str):
        """
        Initialize Q&A system
        
        Args:
            document_path: Path to document file
        """
        self.document_path = document_path
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self):
        """Load and process documents"""
        # Load document
        loader = TextLoader(self.document_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        print(f"✓ Loaded and processed {len(chunks)} chunks")
        
    def create_qa_chain(self):
        """Create Q&A chain"""
        llm = OpenAI(temperature=0)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        
    def ask(self, question: str) -> dict:
        """
        Ask a question
        
        Args:
            question: Question to ask
            
        Returns:
            dict with answer and sources
        """
        if not self.qa_chain:
            raise ValueError("Q&A chain not initialized. Call create_qa_chain() first.")
        
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }


def main():
    """CLI interface"""
    print("\n" + "="*60)
    print("Document Q&A System")
    print("="*60)
    
    # Get document path
    doc_path = input("\nEnter document path: ").strip()
    
    if not doc_path:
        print("No path provided. Exiting.")
        return
    
    try:
        # Initialize system
        print("\nLoading and processing document...")
        qa_system = QASystem(doc_path)
        qa_system.load_documents()
        qa_system.create_qa_chain()
        
        print("\n" + "="*60)
        print("Ready! Ask questions about your document.")
        print("Type 'quit' to exit")
        print("="*60 + "\n")
        
        # Q&A loop
        while True:
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            try:
                result = qa_system.ask(question)
                
                print(f"\nAnswer: {result['answer']}")
                print(f"\nSources: {len(result['sources'])} relevant chunks")
                print("-" * 60 + "\n")
                
            except Exception as e:
                print(f"\nError: {e}\n")
                
    except FileNotFoundError:
        print(f"\nError: File '{doc_path}' not found")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
```

## Features

### Basic Q&A (qa_basic.py)
- Load and process documents
- Answer questions using RAG
- Show source chunks
- Simple CLI interface

### Advanced Q&A (qa_advanced.py)
- Conversation memory
- Follow-up questions
- Multiple document support
- Batch processing
- Export answers

## Usage Examples

### Example 1: Tech Documentation

```python
from qa_basic import QASystem

qa = QASystem("docs/api_reference.txt")
qa.load_documents()
qa.create_qa_chain()

result = qa.ask("How do I authenticate with the API?")
print(result["answer"])
```

### Example 2: Research Papers

```python
qa = QASystem("research/paper.txt")
qa.load_documents()
qa.create_qa_chain()

questions = [
    "What is the main contribution?",
    "What datasets were used?",
    "What are the results?"
]

for q in questions:
    result = qa.ask(q)
    print(f"Q: {q}")
    print(f"A: {result['answer']}\n")
```

## Enhancements

1. **Multiple Documents**: Load entire directories
2. **PDF Support**: Process PDF files
3. **Web Pages**: Query web content
4. **Conversation Mode**: Remember previous Q&A
5. **Export**: Save Q&A sessions
6. **Advanced Retrieval**: MMR, similarity thresholds
7. **Citations**: Show exact text sources

## Common Issues

### Issue: "No relevant information found"
**Solution**: 
- Check document content is relevant
- Reduce chunk size
- Increase k (number of chunks retrieved)

### Issue: Slow performance
**Solution**:
- Use smaller documents
- Reduce chunk_overlap
- Cache embeddings

### Issue: Irrelevant answers
**Solution**:
- Lower temperature (more focused)
- Increase retrieval k
- Improve document quality

## Key Concepts Learned

✓ RAG architecture and workflow  
✓ Document loading and chunking  
✓ Text embeddings and vector stores  
✓ Semantic similarity search  
✓ Context-aware answer generation  
✓ Source attribution  

## Next Steps

- Try different document types
- Experiment with chunk sizes
- Add conversation memory
- Build web interface with Streamlit
- Move on to [Intermediate Projects](../../intermediate/)

## Resources

- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

---

**Level**: Beginner | **Time**: 3-4 hours | **Difficulty**: ⭐⭐⭐☆☆
