"""
Simple Chatbot with Memory
===========================

A conversational AI chatbot that remembers context throughout the conversation.

Usage:
    python chatbot.py
    
Commands:
    - Type your message to chat
    - 'quit' or 'exit' to end the conversation
    - 'memory' to see conversation history
"""

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def chatbot_with_memory():
    """
    Main chatbot function with conversation memory
    """
    
    # Verify API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    # Initialize components
    print("Initializing chatbot...")
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False  # Set to True to see the full prompt being sent
    )
    
    # Welcome message
    print("\n" + "="*60)
    print("Simple Chatbot with Memory")
    print("="*60)
    print("\nChatbot: Hi! I'm your AI assistant. I'll remember our conversation.")
    print("         Type 'quit' to exit, 'memory' to see history.")
    print("="*60 + "\n")
    
    # Main conversation loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\nChatbot: Goodbye! It was nice talking to you. 👋\n")
            break
        
        # Check for memory display command
        if user_input.lower() == 'memory':
            history = memory.load_memory_variables({}).get('history', 'No history yet.')
            print("\n" + "-"*60)
            print("Conversation History:")
            print("-"*60)
            print(history)
            print("-"*60 + "\n")
            continue
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Get response from chatbot
        try:
            response = conversation.predict(input=user_input)
            print(f"\nChatbot: {response}\n")
        except Exception as e:
            print(f"\nError: {e}")
            print("Chatbot: Sorry, I encountered an error. Please try again.\n")


if __name__ == "__main__":
    chatbot_with_memory()
