# LangChain Quick Reference Guide

Quick reference for common LangChain patterns and code snippets.

## Table of Contents

- [Basic Setup](#basic-setup)
- [Models](#models)
- [Prompts](#prompts)
- [Chains](#chains)
- [Memory](#memory)
- [Document Processing](#document-processing)
- [Vector Stores](#vector-stores)
- [Agents](#agents)
- [Common Patterns](#common-patterns)

---

## Basic Setup

```python
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import common modules
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
```

---

## Models

### OpenAI LLM

```python
from langchain_openai import OpenAI

# Basic usage
llm = OpenAI(temperature=0.7)
response = llm.invoke("Your prompt here")

# With parameters
llm = OpenAI(
    temperature=0.9,
    max_tokens=100,
    model_name="gpt-3.5-turbo-instruct"
)
```

### Chat Model

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0.7)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!")
]

response = chat.invoke(messages)
```

---

## Prompts

### Simple Template

```python
from langchain.prompts import PromptTemplate

template = "Tell me about {topic}"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template
)

formatted = prompt.format(topic="AI")
```

### Chat Prompt

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("human", "{question}")
])

messages = template.format_messages(
    role="teacher",
    question="Explain gravity"
)
```

### Few-Shot

```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3+3", "output": "6"}
]

template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)
```

---

## Chains

### LLM Chain

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"input": "test"})
```

### Sequential Chain

```python
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

result = overall_chain.invoke("input")
```

### With Memory

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

response = conversation.predict(input="Hello")
```

---

## Memory

### Buffer Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "Hi"}, 
    {"output": "Hello!"}
)

history = memory.load_memory_variables({})
```

### Window Memory

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3)  # Keep last 3 exchanges
```

### Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
```

---

## Document Processing

### Load Documents

```python
# Text file
from langchain.document_loaders import TextLoader
loader = TextLoader("file.txt")
docs = loader.load()

# PDF
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("file.pdf")
pages = loader.load_and_split()

# Web page
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
docs = loader.load()
```

### Split Text

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
```

---

## Vector Stores

### Create Vector Store

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### Search

```python
# Similarity search
results = vectorstore.similarity_search("query", k=3)

# With scores
results = vectorstore.similarity_search_with_score("query", k=3)
```

### As Retriever

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

docs = retriever.get_relevant_documents("query")
```

---

## Agents

### Basic Agent

```python
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.invoke("What is the population of Tokyo?")
```

### Custom Tool

```python
from langchain.tools import Tool

def custom_function(input_text):
    return f"Processed: {input_text}"

custom_tool = Tool(
    name="CustomTool",
    func=custom_function,
    description="Useful for processing text"
)

tools = [custom_tool]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

---

## Common Patterns

### Token Counting

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"input": "test"})
    print(f"Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost}")
```

### Error Handling

```python
try:
    response = llm.invoke(prompt)
except Exception as e:
    print(f"Error: {e}")
```

### Streaming

```python
for chunk in chat.stream("Write a story"):
    print(chunk.content, end="", flush=True)
```

### Caching

```python
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()
```

### Async

```python
async def async_call():
    response = await llm.ainvoke("prompt")
    return response

import asyncio
result = asyncio.run(async_call())
```

---

## RAG Pattern

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

result = qa_chain.invoke("What is the document about?")
```

---

## Output Parsers

### Comma-Separated List

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
result = parser.parse("apple, banana, cherry")
# ['apple', 'banana', 'cherry']
```

### Structured Output

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

schemas = [
    ResponseSchema(name="answer", description="The answer"),
    ResponseSchema(name="source", description="The source")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
```

---

## Debugging

### Verbose Mode

```python
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
```

### Callbacks

```python
from langchain.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler()
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    callbacks=[handler]
)
```

---

## Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACEHUB_API_TOKEN=hf_...
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

---

## Useful Commands

```bash
# Install LangChain
pip install langchain langchain-openai

# Install with extras
pip install langchain[all]

# Update
pip install --upgrade langchain

# Check version
python -c "import langchain; print(langchain.__version__)"
```

---

## Common Imports

```python
# Core
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

# Document processing
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS

# Agents
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool

# Utilities
from dotenv import load_dotenv
import os
```

---

**For more details, see the full [documentation](01-getting-started.md).**
