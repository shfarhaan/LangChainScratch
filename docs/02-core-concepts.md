# Core Concepts in LangChain

## Overview

This guide covers the fundamental building blocks of LangChain. Understanding these concepts is crucial for building effective LLM applications.

## Table of Contents

1. [Language Models (LLMs)](#language-models-llms)
2. [Chat Models](#chat-models)
3. [Prompts and Prompt Templates](#prompts-and-prompt-templates)
4. [Output Parsers](#output-parsers)
5. [Chains](#chains)
6. [Memory](#memory)
7. [Document Loaders](#document-loaders)
8. [Text Splitters](#text-splitters)
9. [Embeddings](#embeddings)
10. [Vector Stores](#vector-stores)

---

## Language Models (LLMs)

LLMs are the core of LangChain applications. They take text as input and generate text as output.

### Basic Usage

```python
from langchain_openai import OpenAI

# Initialize
llm = OpenAI(temperature=0.7)

# Invoke
response = llm.invoke("Explain quantum computing in one sentence")
print(response)
```

### Key Parameters

- **temperature** (0-1): Controls randomness
  - 0: Deterministic, focused
  - 1: Creative, diverse
  
- **max_tokens**: Maximum length of response

- **model_name**: Specific model to use
  - `gpt-3.5-turbo-instruct`
  - `gpt-4`

### Multiple Providers

```python
# OpenAI
from langchain_openai import OpenAI
llm = OpenAI()

# Anthropic
from langchain_anthropic import Anthropic
llm = Anthropic()

# HuggingFace
from langchain_community.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="google/flan-t5-xl")
```

---

## Chat Models

Chat models are optimized for conversational interfaces, using message-based inputs.

### Messages

```python
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

chat = ChatOpenAI(temperature=0)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What is LangChain?")
]

response = chat.invoke(messages)
print(response.content)
```

### Message Types

1. **SystemMessage**: Sets behavior/context
2. **HumanMessage**: User input
3. **AIMessage**: Assistant response

### Streaming

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(streaming=True)

for chunk in chat.stream("Write a poem about AI"):
    print(chunk.content, end="", flush=True)
```

---

## Prompts and Prompt Templates

Prompts structure your inputs to LLMs. Templates make prompts reusable and dynamic.

### Simple Template

```python
from langchain.prompts import PromptTemplate

template = "Tell me a {adjective} joke about {content}"

prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template=template
)

formatted = prompt.format(adjective="funny", content="programming")
print(formatted)
# Output: "Tell me a funny joke about programming"
```

### Chat Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}")
])

messages = template.format_messages(
    input_language="English",
    output_language="French",
    text="Hello, how are you?"
)
```

### Few-Shot Prompts

```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_template = """
Input: {input}
Output: {output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every word",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"]
)

print(few_shot_prompt.format(adjective="big"))
```

---

## Output Parsers

Output parsers structure LLM responses into usable formats.

### Simple Parser

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

# Get format instructions
format_instructions = parser.get_format_instructions()
print(format_instructions)
# Output: "Your response should be a list of comma separated values, eg: `foo, bar, baz`"

# Parse output
output = "apple, banana, cherry"
result = parser.parse(output)
print(result)
# Output: ['apple', 'banana', 'cherry']
```

### Structured Output Parser

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="name", description="The name of the person"),
    ResponseSchema(name="age", description="The age of the person"),
    ResponseSchema(name="occupation", description="The person's job")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# Use in a prompt
from langchain.prompts import PromptTemplate

template = """
Extract information about the person mentioned in the text.

{format_instructions}

Text: {text}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
)
```

### Pydantic Parser

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's job")

parser = PydanticOutputParser(pydantic_object=Person)
format_instructions = parser.get_format_instructions()
```

---

## Chains

Chains combine multiple components into a single workflow.

### LLMChain

The most basic chain: LLM + Prompt.

```python
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI()
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"product": "eco-friendly shoes"})
print(result)
```

### Sequential Chains

Run multiple chains in sequence.

```python
from langchain.chains import SimpleSequentialChain

# Chain 1: Generate a topic
chain1 = LLMChain(llm=llm, prompt=prompt1)

# Chain 2: Write about the topic
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Combine
overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

result = overall_chain.invoke("technology")
```

### Router Chains

Dynamically route to different chains based on input.

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

# Define destination chains
physics_template = """You are a physics professor. Answer: {input}"""
math_template = """You are a math professor. Answer: {input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for physics questions",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for math questions",
        "prompt_template": math_template
    }
]

# Create router chain (simplified example)
# See docs for full implementation
```

---

## Memory

Memory allows chains to remember previous interactions.

### Conversation Buffer Memory

Stores all messages as-is.

```python
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.invoke("Hi, my name is Alice")
conversation.invoke("What's my name?")
# Output: "Your name is Alice"
```

### Conversation Summary Memory

Summarizes conversation over time.

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=OpenAI())

conversation = ConversationChain(
    llm=llm,
    memory=memory
)
```

### Conversation Buffer Window Memory

Keeps only the last N interactions.

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)  # Keep last 2 exchanges
```

---

## Document Loaders

Load data from various sources.

### Text Files

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("path/to/file.txt")
documents = loader.load()
```

### PDF Files

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("path/to/file.pdf")
pages = loader.load_and_split()
```

### Web Pages

```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
documents = loader.load()
```

### CSV Files

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader("path/to/file.csv")
documents = loader.load()
```

---

## Text Splitters

Break large documents into smaller chunks for processing.

### Character Text Splitter

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_text(long_text)
```

### Recursive Character Text Splitter

Tries multiple separators hierarchically.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

documents = text_splitter.split_documents(docs)
```

---

## Embeddings

Convert text to numerical vectors for semantic search.

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Embed a query
query_vector = embeddings.embed_query("What is LangChain?")

# Embed documents
doc_vectors = embeddings.embed_documents([
    "LangChain is a framework",
    "It helps build LLM apps"
])
```

### HuggingFace Embeddings

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## Vector Stores

Store and search embeddings efficiently.

### Chroma

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Similarity search
results = vectorstore.similarity_search("query text", k=3)
```

### FAISS

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

# Load later
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## Putting It All Together

Here's an example combining multiple concepts:

```python
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 1. Load documents
loader = TextLoader("data.txt")
documents = loader.load()

# 2. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# 4. Create retrieval chain
llm = OpenAI()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. Ask questions
result = qa_chain.invoke("What is the main topic?")
print(result)
```

---

## Best Practices

1. **Choose the Right Model**: Balance cost, speed, and quality
2. **Optimize Prompts**: Clear, specific instructions yield better results
3. **Handle Errors**: Always wrap LLM calls in try-except blocks
4. **Monitor Costs**: Track token usage with callbacks
5. **Test Incrementally**: Build and test each component separately
6. **Use Memory Wisely**: Choose memory type based on use case
7. **Chunk Documents Properly**: Balance between context and specificity

---

## Next Steps

- **Practice**: Try the [examples](../examples/) in this repo
- **Build Projects**: Start with [beginner projects](../projects/beginner/)
- **Learn Advanced Topics**: Move on to [03-prompts-chains.md](03-prompts-chains.md)

---

**Previous**: [Getting Started](01-getting-started.md) | **Next**: [Prompts and Chains](03-prompts-chains.md)
