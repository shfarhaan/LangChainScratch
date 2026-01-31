"""
Conversational Document Q&A
============================

QA system with conversation memory for follow-up questions.

Usage:
    python qa_conversational.py
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv
import os

load_dotenv()


def setup_conversational_qa(directory="documents"):
    """Set up conversational QA system"""
    
    # Load documents
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Split and create vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create conversational chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return qa_chain


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return
    
    if not os.path.exists("documents"):
        print("Please create a 'documents' directory with .txt files")
        return
    
    print("\nConversational Document Q&A")
    print("="*70)
    print("This system remembers conversation history for follow-up questions.")
    print("Type 'quit' to exit, 'memory' to see history.\n")
    
    qa_chain = setup_conversational_qa("documents")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                break
            
            if question.lower() == 'memory':
                history = qa_chain.memory.load_memory_variables({})
                print(f"\n{history}\n")
                continue
            
            if not question:
                continue
            
            result = qa_chain({"question": question})
            print(f"\nAssistant: {result['answer']}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("Goodbye!")


if __name__ == "__main__":
    main()
