"""
Basic Q&A System with RAG
=========================
Question-answering system using Retrieval-Augmented Generation.

Usage:
    python qa_basic.py
"""

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

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
