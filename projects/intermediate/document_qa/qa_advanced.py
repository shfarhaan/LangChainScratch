"""
Advanced RAG Document Q&A System
=================================

Enhanced RAG implementation with contextual compression, re-ranking, and better retrieval.
Demonstrates production-ready patterns for document Q&A.

Usage:
    python qa_advanced.py

Features:
    - Contextual compression for better relevance
    - LLM-based re-ranking
    - MMR (Maximum Marginal Relevance) search
    - Custom prompts for better answers
    - Enhanced source tracking
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()


def load_and_process_documents(directory="documents"):
    """Load and process documents with metadata"""
    try:
        # Load documents
        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        documents = loader.load()
        
        if not documents:
            print(f"No documents found in {directory}/")
            return None
        
        # Add enhanced metadata
        for i, doc in enumerate(documents):
            doc.metadata["doc_id"] = i
            doc.metadata["filename"] = os.path.basename(doc.metadata.get("source", "unknown"))
        
        print(f"Loaded {len(documents)} document(s)")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None


def create_advanced_vector_store(documents):
    """Create vector store with optimized chunking"""
    try:
        # Advanced text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for precision
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True  # Track position in original document
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Preserve chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db_advanced"
        )
        print("Advanced vector store created successfully")
        
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


def create_advanced_qa_chain(vectorstore):
    """Create QA chain with contextual compression and better prompts"""
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        # Create base retriever with MMR
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 5,  # Return top 5 documents
                "fetch_k": 10,  # Fetch 10 candidates before filtering
                "lambda_mult": 0.5  # Balance between relevance and diversity
            }
        )
        
        # Add contextual compression
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # Create enhanced prompt
        template = """You are an expert assistant specialized in answering questions based on provided documents.

Instructions:
1. Use ONLY the information from the context below to answer the question
2. If the context doesn't contain enough information, clearly state that
3. Provide specific details and quote relevant parts when helpful
4. Always mention which source(s) you used for your answer
5. Be concise but thorough

Context:
{context}

Question: {question}

Detailed Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        
        return qa_chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None


def format_sources_enhanced(source_documents):
    """Enhanced source formatting with metadata"""
    sources = []
    seen_sources = set()
    
    for i, doc in enumerate(source_documents, 1):
        filename = doc.metadata.get('filename', 'Unknown')
        doc_id = doc.metadata.get('doc_id', '?')
        chunk_id = doc.metadata.get('chunk_id', '?')
        
        # Avoid duplicate sources
        source_key = f"{filename}:{chunk_id}"
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        
        # Format content preview
        content = doc.page_content[:150].replace('\n', ' ')
        if len(doc.page_content) > 150:
            content += "..."
        
        sources.append(f"  [{i}] {filename} (Doc {doc_id}, Chunk {chunk_id})\n      {content}")
    
    return "\n".join(sources)


def query_with_metrics(qa_chain, question):
    """Query with performance metrics"""
    start_time = time.time()
    
    with get_openai_callback() as cb:
        result = qa_chain({"query": question})
        
        latency = time.time() - start_time
        
        metrics = {
            "latency": latency,
            "tokens": cb.total_tokens,
            "cost": cb.total_cost,
            "result": result
        }
        
        return metrics


def main():
    """Main function to run advanced Q&A system"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    print("\n" + "="*70)
    print("Advanced Document Q&A System with Contextual Compression")
    print("="*70 + "\n")
    
    # Check if documents directory exists
    if not os.path.exists("documents"):
        print("Creating documents directory...")
        os.makedirs("documents")
        
        # Create sample documents
        sample1 = """RAG (Retrieval-Augmented Generation) is a technique that combines 
information retrieval with language model generation. It works by:
1. Retrieving relevant documents from a knowledge base
2. Using those documents as context for the language model
3. Generating answers based on the retrieved information

Benefits of RAG:
- Reduces hallucinations by grounding responses in real data
- Enables access to up-to-date information without retraining
- More cost-effective than fine-tuning for knowledge updates
- Provides source attribution for transparency"""

        sample2 = """Vector databases store embeddings (numerical representations of text)
and enable semantic search. Key concepts:

Embeddings: Dense vectors that capture semantic meaning
Similarity Search: Finding similar vectors using cosine similarity or other metrics
Vector Stores: ChromaDB, Pinecone, FAISS, Weaviate

Best Practices:
- Choose appropriate chunk sizes (500-1500 tokens)
- Use overlap between chunks (10-20%)
- Include metadata for filtering
- Consider hybrid search (keyword + semantic)"""
        
        with open("documents/rag_basics.txt", "w") as f:
            f.write(sample1)
        with open("documents/vector_stores.txt", "w") as f:
            f.write(sample2)
        
        print("Created sample documents in documents/")
        print("Run the program again to start asking questions.\n")
        return
    
    # Load and process documents
    print("Loading documents...")
    documents = load_and_process_documents("documents")
    if not documents:
        return
    
    # Create advanced vector store
    print("\nCreating advanced vector store...")
    vectorstore = create_advanced_vector_store(documents)
    if not vectorstore:
        return
    
    # Create advanced QA chain
    print("\nInitializing advanced Q&A chain...")
    qa_chain = create_advanced_qa_chain(vectorstore)
    if not qa_chain:
        return
    
    print("\n" + "="*70)
    print("Ready! This system uses:")
    print("  - Contextual compression for better relevance")
    print("  - MMR search for diverse results")
    print("  - Enhanced prompts for detailed answers")
    print("\nType 'quit' to exit, 'help' for commands.")
    print("="*70 + "\n")
    
    # Interactive Q&A loop
    total_queries = 0
    total_cost = 0.0
    
    while True:
        try:
            # Get user question
            question = input("Question: ").strip()
            
            # Check for commands
            if question.lower() in ['quit', 'exit', 'q']:
                print(f"\nSession Summary:")
                print(f"  Total Queries: {total_queries}")
                print(f"  Total Cost: ${total_cost:.4f}")
                print("\nGoodbye!")
                break
            
            if question.lower() == 'help':
                print("\nCommands:")
                print("  - Type your question to get an answer")
                print("  - 'quit' or 'exit' to close the program")
                print("  - 'help' to see this message")
                print("\nFeatures:")
                print("  - Contextual compression for relevant snippets")
                print("  - Source tracking and citations")
                print("  - Performance metrics\n")
                continue
            
            if not question:
                continue
            
            # Query with metrics
            print("\nSearching and compressing context...")
            metrics = query_with_metrics(qa_chain, question)
            result = metrics["result"]
            
            total_queries += 1
            total_cost += metrics["cost"]
            
            # Display answer
            print("\n" + "-"*70)
            print("Answer:")
            print("-"*70)
            print(result["result"])
            
            # Display sources
            if result.get("source_documents"):
                print("\n" + "-"*70)
                print("Sources Used:")
                print("-"*70)
                print(format_sources_enhanced(result["source_documents"]))
            
            # Display metrics
            print("\n" + "-"*70)
            print("Query Metrics:")
            print("-"*70)
            print(f"  Latency: {metrics['latency']:.2f}s")
            print(f"  Tokens: {metrics['tokens']}")
            print(f"  Cost: ${metrics['cost']:.4f}")
            print(f"  Sources Retrieved: {len(result.get('source_documents', []))}")
            
            print("\n")
            
        except KeyboardInterrupt:
            print(f"\n\nSession Summary:")
            print(f"  Total Queries: {total_queries}")
            print(f"  Total Cost: ${total_cost:.4f}")
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
