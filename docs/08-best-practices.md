# Best Practices for LangChain Development

## Introduction

This guide covers essential best practices for building production-ready LangChain applications. Follow these guidelines to create robust, efficient, and maintainable LLM-powered systems.

## Table of Contents

1. [Prompt Engineering](#prompt-engineering)
2. [Error Handling](#error-handling)
3. [Cost Optimization](#cost-optimization)
4. [Security](#security)
5. [Performance](#performance)
6. [Testing](#testing)
7. [Monitoring](#monitoring)
8. [Production Deployment](#production-deployment)

---

## Prompt Engineering

### Be Specific and Clear

❌ **Bad:**
```python
prompt = "Tell me about AI"
```

✅ **Good:**
```python
prompt = """
Provide a 3-paragraph explanation of artificial intelligence suitable for 
high school students. Include:
1. A simple definition
2. Real-world applications
3. Future implications
"""
```

### Use Few-Shot Examples

```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "big", "output": "small"},
    {"input": "fast", "output": "slow"}
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)
```

### Include Context and Constraints

```python
template = """
Context: You are a professional email assistant.

Task: Convert the following casual message into a professional email.

Constraints:
- Keep it under 150 words
- Maintain a formal tone
- Include a clear subject line

Casual message: {message}

Professional email:
"""
```

### Iterate and Test

1. Start with a simple prompt
2. Test with various inputs
3. Identify failure cases
4. Refine and add constraints
5. Repeat

---

## Error Handling

### Always Wrap LLM Calls

```python
from langchain_openai import OpenAI
import time

def safe_llm_call(llm, prompt, max_retries=3):
    """Make LLM call with error handling and retries"""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if "rate limit" in str(e).lower():
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                raise
            else:
                time.sleep(1)
```

### Handle Specific Exceptions

```python
from openai import OpenAIError, RateLimitError, APIError

try:
    response = llm.invoke(prompt)
except RateLimitError:
    print("Rate limit exceeded. Please try again later.")
except APIError as e:
    print(f"API error occurred: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Validate Inputs

```python
def validate_input(text: str, max_length: int = 10000) -> bool:
    """Validate user input before sending to LLM"""
    if not text or not text.strip():
        raise ValueError("Input cannot be empty")
    
    if len(text) > max_length:
        raise ValueError(f"Input too long (max {max_length} chars)")
    
    return True

# Usage
try:
    validate_input(user_input)
    response = llm.invoke(user_input)
except ValueError as e:
    print(f"Invalid input: {e}")
```

---

## Cost Optimization

### Monitor Token Usage

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"input": "test"})
    
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost:.6f}")
```

### Use Appropriate Models

```python
# For simple tasks
llm_simple = OpenAI(model="gpt-3.5-turbo-instruct")  # Cheaper

# For complex reasoning
llm_complex = OpenAI(model="gpt-4")  # More expensive but better
```

### Implement Caching

```python
from langchain.cache import InMemoryCache
import langchain

# Enable caching
langchain.llm_cache = InMemoryCache()

# Subsequent calls with same input are cached
llm = OpenAI()
llm.invoke("What is 2+2?")  # Calls API
llm.invoke("What is 2+2?")  # Returns from cache
```

### Limit Response Length

```python
llm = OpenAI(max_tokens=100)  # Limit response length
```

### Batch Processing

```python
# Instead of multiple individual calls
results = []
for prompt in prompts:
    results.append(llm.invoke(prompt))

# Use batch processing
results = llm.batch(prompts)
```

---

## Security

### Never Hardcode API Keys

❌ **Bad:**
```python
openai_api_key = "sk-..."
```

✅ **Good:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

### Sanitize User Inputs

```python
import re

def sanitize_input(text: str) -> str:
    """Remove potentially harmful content"""
    # Remove system prompts injection attempts
    text = re.sub(r'(?i)ignore (previous|all).*(instructions|prompts)', '', text)
    
    # Limit length
    text = text[:5000]
    
    return text.strip()
```

### Implement Rate Limiting

```python
from time import time, sleep

class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            now = time()
            # Remove old calls
            self.calls = [c for c in self.calls if now - c < self.period]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                sleep(sleep_time)
            
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

@RateLimiter(max_calls=10, period=60)  # 10 calls per minute
def call_llm(prompt):
    return llm.invoke(prompt)
```

### Use Environment-Specific Configs

```python
import os

ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    # Production settings
    DEBUG = False
    MAX_TOKENS = 500
else:
    # Development settings
    DEBUG = True
    MAX_TOKENS = 1000
```

---

## Performance

### Use Streaming for Long Responses

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(streaming=True)

for chunk in chat.stream("Write a long essay about AI"):
    print(chunk.content, end="", flush=True)
```

### Parallel Processing

```python
import asyncio
from langchain_openai import ChatOpenAI

async def process_prompt(prompt: str):
    chat = ChatOpenAI()
    return await chat.ainvoke(prompt)

async def process_all(prompts: list):
    tasks = [process_prompt(p) for p in prompts]
    return await asyncio.gather(*tasks)

# Usage
prompts = ["prompt1", "prompt2", "prompt3"]
results = asyncio.run(process_all(prompts))
```

### Optimize Document Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Good chunk size for most use cases
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Not too big (expensive) or small (loses context)
    chunk_overlap=200,  # Maintain context between chunks
    separators=["\n\n", "\n", " ", ""]  # Try logical breaks first
)
```

---

## Testing

### Unit Tests

```python
import pytest
from your_app import process_query

def test_basic_query():
    result = process_query("What is 2+2?")
    assert "4" in result

def test_empty_query():
    with pytest.raises(ValueError):
        process_query("")

def test_long_query():
    long_text = "x" * 10000
    with pytest.raises(ValueError):
        process_query(long_text)
```

### Integration Tests

```python
def test_full_chain():
    """Test complete chain execution"""
    from langchain_openai import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    
    llm = OpenAI(temperature=0)  # Deterministic for testing
    prompt = PromptTemplate(
        input_variables=["number"],
        template="What is {number} + 1?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    result = chain.invoke({"number": "5"})
    assert "6" in result["text"]
```

### Prompt Testing

```python
def test_prompt_variations():
    """Test prompt with various inputs"""
    test_cases = [
        ("simple", "expected_output"),
        ("complex", "expected_output"),
        ("edge_case", "expected_output")
    ]
    
    for input_text, expected in test_cases:
        result = chain.invoke({"input": input_text})
        assert expected in result["text"]
```

---

## Monitoring

### Log All LLM Calls

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_llm_call(prompt: str, response: str, tokens: int, cost: float):
    logger.info(f"""
    LLM Call:
    - Prompt: {prompt[:100]}...
    - Response: {response[:100]}...
    - Tokens: {tokens}
    - Cost: ${cost:.6f}
    """)

# Usage
with get_openai_callback() as cb:
    result = llm.invoke(prompt)
    log_llm_call(prompt, result, cb.total_tokens, cb.total_cost)
```

### Track Metrics

```python
class MetricsTracker:
    def __init__(self):
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.errors = 0
    
    def record_call(self, tokens: int, cost: float, error: bool = False):
        self.total_calls += 1
        self.total_tokens += tokens
        self.total_cost += cost
        if error:
            self.errors += 1
    
    def get_stats(self):
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "error_rate": self.errors / self.total_calls if self.total_calls > 0 else 0
        }
```

---

## Production Deployment

### Use Environment Variables

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
```

### Implement Health Checks

```python
def health_check():
    """Verify system health"""
    try:
        # Check API connectivity
        llm = OpenAI()
        llm.invoke("test")
        
        # Check database connection
        # check_db_connection()
        
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Use Async for Scalability

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI

app = FastAPI()
chat = ChatOpenAI()

@app.post("/chat")
async def chat_endpoint(message: str):
    response = await chat.ainvoke(message)
    return {"response": response.content}
```

### Implement Graceful Degradation

```python
def get_response(query: str):
    try:
        # Try primary model
        return llm_primary.invoke(query)
    except Exception as e:
        logger.error(f"Primary model failed: {e}")
        try:
            # Fallback to secondary model
            return llm_fallback.invoke(query)
        except Exception as e2:
            logger.error(f"Fallback model failed: {e2}")
            # Return default response
            return "I'm sorry, I'm having trouble processing your request."
```

---

## Summary Checklist

✅ **Development**
- [ ] Use environment variables for configuration
- [ ] Implement comprehensive error handling
- [ ] Add input validation
- [ ] Write clear, specific prompts
- [ ] Monitor token usage

✅ **Testing**
- [ ] Write unit tests for components
- [ ] Test edge cases
- [ ] Validate prompt variations
- [ ] Test error scenarios

✅ **Production**
- [ ] Enable logging and monitoring
- [ ] Implement rate limiting
- [ ] Set up health checks
- [ ] Use async for scalability
- [ ] Have fallback strategies

✅ **Cost Management**
- [ ] Use appropriate models
- [ ] Implement caching
- [ ] Limit response lengths
- [ ] Batch requests when possible

✅ **Security**
- [ ] Never hardcode secrets
- [ ] Sanitize user inputs
- [ ] Implement authentication
- [ ] Use HTTPS in production

---

**Previous**: [Advanced Topics](07-advanced-topics.md) | **Back to**: [Main README](../README.md)
