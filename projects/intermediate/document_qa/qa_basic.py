"""
Basic RAG Document Q&A System
==============================

A simple implementation of Retrieval-Augmented Generation (RAG) for document Q&A.
Loads documents, creates embeddings, and answers questions using retrieved context.

Usage:
    python qa_basic.py

Features:
    - Document loading and processing
    - Vector store creation with ChromaDB
    - Question answering with source citations
    - Simple CLI interface
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()


def load_documents(directory="documents"):
    """Load documents from a directory"""
    try:
        # Try to load from documents directory
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
            
        print(f"Loaded {len(documents)} document(s)")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None


def create_vector_store(documents):
    """Create a vector store from documents"""
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("Vector store created successfully")
        
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


def create_qa_chain(vectorstore):
    """Create a QA chain with the vector store"""
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        # Create custom prompt
        template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always provide source information when possible.

Context: {context}

Question: {question}

Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        
        return qa_chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None


def format_sources(source_documents):
    """Format source documents for display"""
    sources = []
    for i, doc in enumerate(source_documents, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        sources.append(f"  [{i}] {source}\n      {content}")
    return "\n".join(sources)


def main():
    """Main function to run the Q&A system"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    print("\n" + "="*70)
    print("Document Q&A System - Basic RAG Implementation")
    print("="*70 + "\n")
    
    # Check if documents directory exists
    if not os.path.exists("documents"):
        print("Creating documents directory...")
        os.makedirs("documents")
        print("\nPlease add .txt files to the 'documents' directory and run again.")
        
        # Create a sample document
        sample_text = """LangChain is a framework for developing applications powered by language models.
It enables applications to:
- Be context-aware: connect a language model to sources of context
- Reason: rely on a language model to reason about how to answer based on provided context

The main value props of LangChain are:
1. Components: abstractions for working with language models
2. Off-the-shelf chains: structured assemblies of components for common tasks

LangChain makes it easy to build LLM applications by providing:
- Simple and intuitive APIs
- Rich ecosystem of integrations
- Flexible architecture for customization"""
        
        with open("documents/sample.txt", "w") as f:
            f.write(sample_text)
        print("Created sample document: documents/sample.txt")
        return
    
    # Load documents
    print("Loading documents...")
    documents = load_documents("documents")
    if not documents:
        return
    
    # Create vector store
    print("\nCreating vector store...")
    vectorstore = create_vector_store(documents)
    if not vectorstore:
        return
    
    # Create QA chain
    print("\nInitializing Q&A chain...")
    qa_chain = create_qa_chain(vectorstore)
    if not qa_chain:
        return
    
    print("\n" + "="*70)
    print("Ready! Ask questions about your documents.")
    print("Type 'quit' to exit, 'help' for commands.")
    print("="*70 + "\n")
    
    # Interactive Q&A loop
    while True:
        try:
            # Get user question
            question = input("Question: ").strip()
            
            # Check for commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'help':
                print("\nCommands:")
                print("  - Type your question to get an answer")
                print("  - 'quit' or 'exit' to close the program")
                print("  - 'help' to see this message\n")
                continue
            
            if not question:
                continue
            
            # Query the QA system
            print("\nSearching documents...")
            result = qa_chain({"query": question})
            
            # Display answer
            print("\n" + "-"*70)
            print("Answer:")
            print("-"*70)
            print(result["result"])
            
            # Display sources
            if result.get("source_documents"):
                print("\n" + "-"*70)
                print("Sources:")
                print("-"*70)
                print(format_sources(result["source_documents"]))
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
