# Getting Started with LangChain

## Introduction

Welcome to your LangChain journey! This guide will help you set up your environment and understand the basics to start building with LangChain.

## What You'll Learn

- Setting up your development environment
- Understanding LangChain architecture
- Making your first LLM call
- Basic prompt engineering
- Understanding chains

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **pip** package manager
- **Git** for version control
- A **text editor** or IDE (VS Code, PyCharm, etc.)
- An **OpenAI API key** (or other LLM provider)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/shfarhaan/LangChainScratch.git
cd LangChainScratch
```

### 2. Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

3. Get your OpenAI API key:
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Sign up or log in
   - Create a new API key
   - Copy and paste it into your `.env` file

## Your First LangChain Program

Let's create a simple script to test your setup:

### Step 1: Create a test file

Create `test_setup.py`:

```python
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

print("✓ Environment variables loaded")

# Initialize the LLM
llm = OpenAI(temperature=0.7)
print("✓ LLM initialized")

# Make a simple call
response = llm.invoke("Say 'Hello, LangChain!'")
print("✓ LLM responded:", response)

print("\n🎉 Setup successful! You're ready to start learning LangChain.")
```

### Step 2: Run the test

```bash
python test_setup.py
```

You should see:
```
✓ Environment variables loaded
✓ LLM initialized
✓ LLM responded: Hello, LangChain!

🎉 Setup successful! You're ready to start learning LangChain.
```

## Understanding LangChain Components

LangChain is built around several core abstractions:

### 1. Models (LLMs)

Language models are the foundation. They generate text based on input.

```python
from langchain_openai import OpenAI

# Initialize with default settings
llm = OpenAI()

# Initialize with custom settings
llm = OpenAI(
    temperature=0.7,  # Controls randomness (0-1)
    max_tokens=100,   # Maximum response length
    model_name="gpt-3.5-turbo-instruct"
)

# Use the model
response = llm.invoke("What is LangChain?")
print(response)
```

**Key Parameters:**
- `temperature`: Controls randomness (0 = deterministic, 1 = creative)
- `max_tokens`: Maximum length of the response
- `model_name`: Which model to use

### 2. Prompts

Prompts are structured inputs to language models.

```python
from langchain.prompts import PromptTemplate

# Create a template
template = """
You are a helpful assistant that explains concepts simply.

Topic: {topic}
Audience: {audience}

Provide a clear explanation:
"""

prompt = PromptTemplate(
    input_variables=["topic", "audience"],
    template=template
)

# Format the prompt
formatted_prompt = prompt.format(
    topic="LangChain",
    audience="beginners"
)

print(formatted_prompt)
```

### 3. Chains

Chains combine models and prompts into workflows.

```python
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create components
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run chain
result = chain.invoke({"product": "AI-powered notebooks"})
print(result)
```

## Common Patterns

### Pattern 1: Simple Q&A

```python
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
question = "What is the capital of France?"
answer = llm.invoke(question)
print(answer)
```

### Pattern 2: Template-Based Generation

```python
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

llm = OpenAI()
template = "Write a {length} {content_type} about {topic}"
prompt = PromptTemplate(
    input_variables=["length", "content_type", "topic"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({
    "length": "short",
    "content_type": "poem",
    "topic": "artificial intelligence"
})
print(result)
```

### Pattern 3: Sequential Processing

```python
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# First chain: Generate a topic
llm = OpenAI()
template1 = "Give me a random {subject} topic"
prompt1 = PromptTemplate(input_variables=["subject"], template=template1)
chain1 = LLMChain(llm=llm, prompt=prompt1)

# Second chain: Write about the topic
template2 = "Write 3 bullet points about: {topic}"
prompt2 = PromptTemplate(input_variables=["topic"], template=template2)
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

result = overall_chain.invoke({"input": "science"})
print(result)
```

## Best Practices for Beginners

### 1. Always Load Environment Variables

```python
from dotenv import load_dotenv
load_dotenv()  # Call this at the start of your script
```

### 2. Handle Errors Gracefully

```python
try:
    response = llm.invoke(prompt)
except Exception as e:
    print(f"Error: {e}")
    # Handle the error appropriately
```

### 3. Monitor Token Usage

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"input": "test"})
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
```

### 4. Start with Low Temperature

For consistent, factual responses, use `temperature=0`:

```python
llm = OpenAI(temperature=0)  # More deterministic
```

For creative tasks, use higher temperature:

```python
llm = OpenAI(temperature=0.9)  # More creative
```

### 5. Test Incrementally

Build your application step by step:
1. Test the LLM alone
2. Add prompt template
3. Create a simple chain
4. Add complexity gradually

## Troubleshooting

### Issue: "API key not found"

**Solution:**
1. Ensure `.env` file exists in your project root
2. Check that `OPENAI_API_KEY` is set correctly
3. Verify you're calling `load_dotenv()` before using the key

### Issue: "Rate limit exceeded"

**Solution:**
1. You're making too many requests too quickly
2. Add delays between requests
3. Consider using a paid OpenAI plan
4. Implement exponential backoff

```python
import time

for i in range(5):
    try:
        response = llm.invoke(prompt)
        break
    except Exception as e:
        if "rate limit" in str(e).lower():
            wait_time = 2 ** i  # Exponential backoff
            print(f"Rate limit hit, waiting {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise e
```

### Issue: "Module not found"

**Solution:**
1. Ensure virtual environment is activated
2. Install dependencies: `pip install -r requirements.txt`
3. Restart your IDE/terminal

### Issue: Response is too long/short

**Solution:**
Adjust `max_tokens` parameter:

```python
llm = OpenAI(max_tokens=100)  # Shorter responses
llm = OpenAI(max_tokens=500)  # Longer responses
```

## Next Steps

Now that you have a working setup:

1. **Explore Core Concepts**: Read [02-core-concepts.md](02-core-concepts.md)
2. **Try Examples**: Check out the `examples/` directory
3. **Build a Project**: Start with [simple_chatbot](../projects/beginner/simple_chatbot/)
4. **Join the Community**: Connect with other learners

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)

---

**Ready to dive deeper?** → [Core Concepts](02-core-concepts.md)
