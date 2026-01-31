# Simple Chatbot Project

## Overview

Build a conversational AI chatbot that remembers context and can have natural conversations.

## What You'll Learn

- Working with chat models
- Implementing conversation memory
- Handling user input/output
- Managing conversation history
- Building a simple CLI interface

## Prerequisites

- Completed [Getting Started](../../../docs/01-getting-started.md)
- Understanding of LLMs and prompts
- Python basics

## Project Structure

```
simple_chatbot/
├── README.md           # This file
├── chatbot.py          # Main chatbot implementation
├── chatbot_basic.py    # Simplified version without memory
└── chatbot_advanced.py # Advanced version with features
```

## Step-by-Step Guide

### Step 1: Basic Chatbot (No Memory)

Create `chatbot_basic.py`:

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

def basic_chatbot():
    """Simple chatbot without memory"""
    chat = ChatOpenAI(temperature=0.7)
    
    system_message = SystemMessage(content="You are a helpful assistant.")
    
    print("Chatbot: Hi! I'm a simple chatbot. Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        messages = [
            system_message,
            HumanMessage(content=user_input)
        ]
        
        response = chat.invoke(messages)
        print(f"Chatbot: {response.content}")

if __name__ == "__main__":
    basic_chatbot()
```

**Limitations:**
- No memory - can't remember previous messages
- Each interaction is independent

### Step 2: Chatbot with Memory

Create `chatbot.py`:

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

def chatbot_with_memory():
    """Chatbot with conversation memory"""
    
    # Initialize components
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False  # Set to True to see the full prompt
    )
    
    print("Chatbot: Hi! I'm your AI assistant. I'll remember our conversation.")
    print("         Type 'quit' to exit, 'memory' to see history.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye! It was nice talking to you.")
            break
        
        if user_input.lower() == 'memory':
            print("\n--- Conversation History ---")
            print(memory.load_memory_variables({})['history'])
            print("--- End of History ---\n")
            continue
        
        if not user_input.strip():
            continue
        
        try:
            response = conversation.predict(input=user_input)
            print(f"Chatbot: {response}\n")
        except Exception as e:
            print(f"Error: {e}")
            print("Chatbot: Sorry, I encountered an error. Please try again.\n")

if __name__ == "__main__":
    chatbot_with_memory()
```

### Step 3: Advanced Chatbot

Create `chatbot_advanced.py`:

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import sys

load_dotenv()

class AdvancedChatbot:
    """Advanced chatbot with multiple features"""
    
    def __init__(self, personality="helpful", memory_window=5):
        """
        Initialize chatbot
        
        Args:
            personality: Type of personality (helpful, funny, professional)
            memory_window: Number of conversation turns to remember
        """
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        
        # Use window memory to avoid token limits
        self.memory = ConversationBufferWindowMemory(k=memory_window)
        
        # Custom prompt based on personality
        personalities = {
            "helpful": "You are a helpful and friendly AI assistant.",
            "funny": "You are a witty AI assistant who loves jokes and humor.",
            "professional": "You are a professional AI assistant focused on clear, concise answers."
        }
        
        self.system_prompt = personalities.get(personality, personalities["helpful"])
        
        # Customize the prompt template
        template = f"""
        {self.system_prompt}
        
        Current conversation:
        {{history}}
        
        Human: {{input}}
        AI:"""
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=False
        )
        
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def chat(self, user_input):
        """Process user input and return response"""
        with get_openai_callback() as cb:
            response = self.conversation.predict(input=user_input)
            self.total_tokens += cb.total_tokens
            self.total_cost += cb.total_cost
        
        return response
    
    def get_stats(self):
        """Return usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost
        }
    
    def clear_memory(self):
        """Clear conversation history"""
        self.memory.clear()
    
    def run(self):
        """Run the chatbot CLI"""
        print(f"\n{'='*60}")
        print("Advanced Chatbot")
        print(f"{'='*60}")
        print(f"Personality: {self.system_prompt}")
        print("\nCommands:")
        print("  'quit' or 'exit' - Exit the chatbot")
        print("  'clear' - Clear conversation memory")
        print("  'stats' - Show usage statistics")
        print(f"{'='*60}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    stats = self.get_stats()
                    print(f"\nChatbot: Goodbye!")
                    print(f"\nSession Statistics:")
                    print(f"  Total Tokens Used: {stats['total_tokens']}")
                    print(f"  Total Cost: ${stats['total_cost']:.6f}")
                    break
                
                if user_input.lower() == 'clear':
                    self.clear_memory()
                    print("Chatbot: Memory cleared! Starting fresh.\n")
                    continue
                
                if user_input.lower() == 'stats':
                    stats = self.get_stats()
                    print(f"\nUsage Statistics:")
                    print(f"  Total Tokens: {stats['total_tokens']}")
                    print(f"  Total Cost: ${stats['total_cost']:.6f}\n")
                    continue
                
                response = self.chat(user_input)
                print(f"Chatbot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nChatbot: Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Chatbot: Sorry, something went wrong. Please try again.\n")

if __name__ == "__main__":
    # You can customize the personality here
    personality = "helpful"  # Options: helpful, funny, professional
    
    # Or get from command line
    if len(sys.argv) > 1:
        personality = sys.argv[1]
    
    chatbot = AdvancedChatbot(personality=personality, memory_window=5)
    chatbot.run()
```

## Running the Chatbot

### Basic Version
```bash
python chatbot_basic.py
```

### With Memory
```bash
python chatbot.py
```

### Advanced Version
```bash
# Default personality
python chatbot_advanced.py

# Custom personality
python chatbot_advanced.py funny
python chatbot_advanced.py professional
```

## Testing Your Chatbot

Try these conversation flows:

### Test 1: Memory
```
You: My name is Alice
Chatbot: Nice to meet you, Alice!
You: What's my name?
Chatbot: Your name is Alice.
```

### Test 2: Context
```
You: I'm planning a trip to Paris
Chatbot: That sounds exciting!
You: What should I visit there?
Chatbot: [Gives recommendations about Paris]
```

### Test 3: Multi-turn conversation
```
You: I'm learning Python
Chatbot: Great choice!
You: Can you recommend resources?
Chatbot: [Provides resources]
You: What about practice projects?
Chatbot: [Suggests projects related to Python]
```

## Challenges

Enhance your chatbot with these features:

1. **Personality Switcher**: Allow users to change personality mid-conversation
2. **Save/Load**: Save conversation history to a file and reload it
3. **Sentiment Analysis**: Detect user sentiment and adjust responses
4. **Topic Detection**: Identify what the user is talking about
5. **Web Search**: Integrate with search APIs for current information
6. **Voice Input/Output**: Add speech recognition and text-to-speech

## Key Concepts Learned

✓ Chat models vs. completion models  
✓ Conversation memory management  
✓ Prompt customization  
✓ Error handling in LLM applications  
✓ Token counting and cost monitoring  
✓ Building interactive CLI applications  

## Common Issues

### Issue: Chatbot forgets after a few messages
**Solution**: Increase `memory_window` in advanced chatbot or use `ConversationSummaryMemory`

### Issue: Responses are too formal/informal
**Solution**: Adjust the system prompt to set desired tone

### Issue: High token usage
**Solution**: Use ConversationBufferWindowMemory with smaller window or ConversationSummaryMemory

## Next Steps

- Add features from the challenges section
- Try different memory types
- Experiment with different models and parameters
- Build a GUI version using Streamlit or Gradio
- Move on to [Text Summarizer](../text_summarizer/) project

## Resources

- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/)
- [OpenAI Chat Models](https://platform.openai.com/docs/guides/chat)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

**Ready to code?** Start with `chatbot_basic.py` and work your way up!
