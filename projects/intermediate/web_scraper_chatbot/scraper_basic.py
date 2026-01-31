"""
Web Scraper Chatbot - Basic
============================

Scrape websites and ask questions about their content using RAG.

Usage:
    python scraper_basic.py
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from urllib.parse import urlparse
import os

load_dotenv()


def is_valid_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception:
        return False


def scrape_and_qa(url):
    """Scrape a URL and create a Q&A system"""
    try:
        print(f"\nScraping {url}...")
        
        # Load webpage
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        if not documents:
            print("No content found at URL")
            return None
        
        print(f"Loaded content ({len(documents[0].page_content)} characters)")
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Create vector store
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(chunks, embeddings)
        
        # Create QA chain
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        template = """Use the following web page content to answer the question.
If you cannot answer based on the content, say so clearly.

Content: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return
    
    print("\n" + "="*70)
    print("Web Scraper Chatbot")
    print("="*70)
    print("\nScrape any webpage and ask questions about its content!")
    print("Type 'quit' to exit, 'new' for a new URL.\n")
    
    qa_chain = None
    current_url = None
    
    while True:
        try:
            # Get URL if we don't have one
            if qa_chain is None:
                url = input("\nEnter URL: ").strip()
                
                if url.lower() in ['quit', 'exit']:
                    break
                
                if not url:
                    continue
                
                if not is_valid_url(url):
                    print("Invalid URL. Please include http:// or https://")
                    continue
                
                qa_chain = scrape_and_qa(url)
                if qa_chain:
                    current_url = url
                    print(f"\nReady! Ask questions about {current_url}")
                    print("Type 'new' for a different URL\n")
                continue
            
            # Get question
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                break
            
            if question.lower() == 'new':
                qa_chain = None
                current_url = None
                continue
            
            if not question:
                continue
            
            # Answer question
            print("\nSearching content...")
            result = qa_chain({"query": question})
            
            print("\n" + "-"*70)
            print("Answer:")
            print("-"*70)
            print(result["result"])
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
    
    print("Goodbye!")


if __name__ == "__main__":
    main()
