# Document Q&A with Advanced RAG

## Overview

Build an enterprise-grade document question-answering system using advanced Retrieval-Augmented Generation (RAG) techniques. This project demonstrates production-ready RAG patterns including hybrid search, re-ranking, and context optimization.

## What You'll Learn

- **Advanced RAG Techniques**: Hybrid search, re-ranking, and context compression
- **Vector Stores**: ChromaDB, FAISS, and Pinecone integration
- **Document Processing**: Advanced chunking strategies and metadata
- **Multi-Document QA**: Querying across multiple documents
- **Source Attribution**: Tracking and citing sources in answers
- **Performance Optimization**: Caching and query optimization
- **Production Patterns**: Error handling and monitoring

## Prerequisites

- Completed [Q&A System](../../beginner/qa_system/) project
- Understanding of embeddings and vector similarity
- Familiarity with document processing

## Project Structure

```
document_qa/
├── README.md              # This file
├── qa_basic.py            # Basic RAG implementation
├── qa_advanced.py         # Advanced RAG with re-ranking
├── qa_conversational.py   # Conversational QA with history
├── qa_multi_doc.py        # Multi-document QA
├── documents/             # Sample documents
│   ├── sample1.txt
│   ├── sample2.pdf
│   └── sample3.md
└── utils.py               # Helper functions
```

## Architecture

### Basic RAG Flow

```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Answer
```

### Advanced RAG Flow

```
User Query → Query Enhancement → Hybrid Search → Re-ranking → 
Context Compression → LLM with Sources → Answer + Citations
```

## Quick Start

### 1. Install Dependencies

```bash
pip install langchain langchain-openai chromadb pypdf sentence-transformers
```

### 2. Set Up Environment

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. Run Basic Version

```bash
python qa_basic.py
```

## Implementation Guide

### Step 1: Basic RAG System

The basic system implements simple document loading, chunking, and retrieval:

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# Load and process documents
loader = TextLoader("documents/sample1.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create QA chain
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Ask questions
result = qa_chain({"query": "What is this document about?"})
print(result["result"])
```

**Key Points:**
- Uses simple text splitting
- Basic vector similarity search
- Returns top-k most similar chunks
- No source tracking

### Step 2: Advanced RAG with Re-ranking

Improve relevance using re-ranking and better chunking:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Add contextual compression
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Use in QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)
```

**Improvements:**
- Retrieves more candidates (k=5)
- Uses LLM to compress and filter context
- Better context relevance
- Reduced token usage

### Step 3: Conversational QA

Add conversation memory for follow-up questions:

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# Ask follow-up questions
result1 = qa_chain({"question": "What is RAG?"})
result2 = qa_chain({"question": "Can you explain that in simpler terms?"})
```

**Features:**
- Maintains conversation context
- Handles follow-up questions
- Reformulates queries based on history
- Natural conversation flow

### Step 4: Multi-Document QA

Query across multiple documents with source tracking:

```python
from langchain_community.document_loaders import DirectoryLoader

# Load multiple documents
loader = DirectoryLoader("documents/", glob="**/*.txt")
documents = loader.load()

# Add metadata for tracking
for i, doc in enumerate(documents):
    doc.metadata["source_id"] = i
    doc.metadata["filename"] = doc.metadata.get("source", "unknown")

# Process and create vector store
chunks = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(chunks, embeddings)

# Query with source attribution
results = qa_chain({"query": "Compare the main themes across documents"})
print(f"Answer: {results['result']}")
print(f"Sources: {[doc.metadata['filename'] for doc in results['source_documents']]}")
```

## Advanced Features

### 1. Hybrid Search (Keyword + Semantic)

Combine traditional keyword search with vector similarity:

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Create keyword retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# Create vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Combine both
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

### 2. Custom Prompts

Improve answer quality with custom prompts:

```python
from langchain.prompts import PromptTemplate

template = """Use the following context to answer the question.
If you cannot answer based on the context, say so clearly.
Always cite the sources you used.

Context: {context}

Question: {question}

Answer with citations:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_PROMPT}
)
```

### 3. Metadata Filtering

Filter documents by metadata:

```python
# Add metadata during chunking
for chunk in chunks:
    chunk.metadata["date"] = "2024-01-01"
    chunk.metadata["category"] = "technical"

# Query with metadata filter
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"category": "technical"}
    }
)
```

### 4. Streaming Responses

Stream answers for better UX:

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

## Production Best Practices

### 1. Error Handling

```python
try:
    result = qa_chain({"query": user_query})
except Exception as e:
    print(f"Error: {e}")
    result = {"result": "I encountered an error processing your question."}
```

### 2. Caching

```python
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()
```

### 3. Cost Management

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = qa_chain({"query": query})
    print(f"Tokens: {cb.total_tokens}, Cost: ${cb.total_cost}")
```

### 4. Monitoring

```python
import time

start_time = time.time()
result = qa_chain({"query": query})
latency = time.time() - start_time
print(f"Query latency: {latency:.2f}s")
```

## Performance Optimization

### 1. Chunk Size Tuning

```python
# Smaller chunks for precise retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Larger chunks for more context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)
```

### 2. Retrieval Optimization

```python
# Fetch more candidates, compress later
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={
        "k": 10,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)
```

### 3. Vector Store Selection

- **ChromaDB**: Fast, simple, good for development
- **FAISS**: Very fast, good for large datasets
- **Pinecone**: Managed, scalable, good for production

## Common Patterns

### Pattern 1: Academic Papers QA

```python
# Use larger context windows
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ". ", " "]
)

# Enable source citations
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
```

### Pattern 2: Customer Support KB

```python
# Add metadata for categorization
for doc in documents:
    doc.metadata["category"] = extract_category(doc)
    doc.metadata["priority"] = extract_priority(doc)

# Filter by category in queries
retriever = vectorstore.as_retriever(
    search_kwargs={
        "filter": {"category": user_category}
    }
)
```

### Pattern 3: Legal Documents

```python
# Use precise chunking at section level
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n## ", "\n\n### ", "\n\n"]
)

# Strict prompting for accuracy
template = """You are a legal document assistant. 
Only answer based on the exact text provided.
Do not infer or assume. Quote directly when possible.

Context: {context}
Question: {question}
Answer:"""
```

## Troubleshooting

### Issue: Poor answer quality
**Solution**: 
- Increase chunk overlap
- Use contextual compression
- Improve prompt specificity
- Adjust temperature (lower = more factual)

### Issue: Slow queries
**Solution**:
- Use smaller embeddings model
- Reduce k parameter
- Implement caching
- Use FAISS instead of Chroma

### Issue: Missing relevant information
**Solution**:
- Increase k parameter
- Use MMR search
- Try hybrid search
- Improve chunking strategy

### Issue: Hallucinations
**Solution**:
- Set temperature to 0
- Use strict prompts
- Add source verification
- Limit context window

## Next Steps

1. Experiment with different chunk sizes and overlap
2. Try different vector stores (FAISS, Pinecone)
3. Implement custom re-rankers
4. Add metadata filtering
5. Build a web interface with Streamlit
6. Deploy to production

## Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Store Comparison](https://python.langchain.com/docs/integrations/vectorstores/)
- [Advanced RAG Techniques](https://blog.langchain.dev/semi-structured-multi-modal-rag/)

## Related Projects

- [Basic Q&A System](../../beginner/qa_system/) - Foundation concepts
- [Web Scraper Chatbot](../web_scraper_chatbot/) - Web-based RAG
- [Research Assistant](../../advanced/research_assistant/) - Multi-source RAG

---

**Next**: Try building the [Web Scraper Chatbot](../web_scraper_chatbot/) to apply RAG to web content!
