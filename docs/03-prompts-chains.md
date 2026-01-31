# Prompts and Chains in LangChain

## Introduction

Prompts and chains are fundamental building blocks in LangChain. This guide covers how to effectively use them to build powerful LLM applications.

## Prompts

### What are Prompts?

Prompts are the instructions you give to language models. They determine what the model will generate.

### Prompt Templates

Make your prompts reusable and dynamic with templates.

#### Basic Template

```python
from langchain.prompts import PromptTemplate

template = """
You are an expert {role}.

Question: {question}

Provide a detailed answer:
"""

prompt = PromptTemplate(
    input_variables=["role", "question"],
    template=template
)

# Use it
formatted = prompt.format(
    role="software engineer",
    question="How do I optimize database queries?"
)
```

#### Chat Prompt Templates

For chat models that use message-based inputs:

```python
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {personality} assistant."),
    ("human", "Tell me about {topic}"),
    ("ai", "I'd be happy to explain {topic}. Let me start by..."),
    ("human", "{follow_up_question}")
])

messages = chat_template.format_messages(
    personality="helpful and friendly",
    topic="machine learning",
    follow_up_question="What are neural networks?"
)
```

### Few-Shot Prompting

Teach the model by example:

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Define examples
examples = [
    {
        "question": "What is 2+2?",
        "answer": "2+2 = 4"
    },
    {
        "question": "What is 10*3?",
        "answer": "10*3 = 30"
    }
]

# Create example template
example_template = """
Question: {question}
Answer: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=example_template
)

# Create few-shot prompt
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Solve these math problems:",
    suffix="Question: {input}\nAnswer:",
    input_variables=["input"]
)

print(few_shot_prompt.format(input="What is 5+7?"))
```

### Dynamic Few-Shot Prompting

Select examples dynamically based on input similarity:

```python
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "cloudy"},
    {"input": "fast", "output": "slow"}
]

# Create example selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=2  # Select 2 most similar examples
)

# Create dynamic few-shot prompt
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym:",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"]
)
```

---

## Chains

Chains combine multiple components into workflows.

### LLMChain

The basic chain: combines an LLM with a prompt template.

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

# Run it
result = chain.invoke({"product": "colorful socks"})
print(result["text"])
```

### Sequential Chains

Run multiple chains in sequence, where output of one becomes input to next.

#### SimpleSequentialChain

For single input/output chains:

```python
from langchain.chains import SimpleSequentialChain

# Chain 1: Generate topic
chain_one = LLMChain(llm=llm, prompt=prompt_one)

# Chain 2: Write about topic
chain_two = LLMChain(llm=llm, prompt=prompt_two)

# Combine
overall_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

result = overall_chain.invoke("artificial intelligence")
```

#### SequentialChain

For multiple inputs/outputs:

```python
from langchain.chains import SequentialChain

# Chain 1: Generate synopsis
synopsis_chain = LLMChain(
    llm=llm,
    prompt=synopsis_prompt,
    output_key="synopsis"
)

# Chain 2: Write review
review_chain = LLMChain(
    llm=llm,
    prompt=review_prompt,
    output_key="review"
)

# Combine with named inputs/outputs
overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["title", "genre"],
    output_variables=["synopsis", "review"],
    verbose=True
)

result = overall_chain({
    "title": "The AI Revolution",
    "genre": "science fiction"
})

print(result["synopsis"])
print(result["review"])
```

### Transform Chain

Apply custom transformations:

```python
from langchain.chains import TransformChain

def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    # Custom transformation
    shortened_text = text[:100]
    return {"output_text": shortened_text}

transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["output_text"],
    transform=transform_func
)

# Use in sequential chain
full_chain = SequentialChain(
    chains=[transform_chain, llm_chain],
    input_variables=["text"],
    output_variables=["final_output"]
)
```

### Router Chains

Route to different chains based on input:

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

# Define destination chains
physics_template = """You are a physics expert. Answer this question:

{input}"""

math_template = """You are a math expert. Answer this question:

{input}"""

# Create prompts
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

# Create destination chains
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# Create router
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = f"""Given a user question, choose which expert should answer it.

{destinations_str}

Question: {{input}}

Expert:"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"]
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Create multi-prompt chain
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

# Use it
result = chain.invoke("What is the speed of light?")
```

---

## Advanced Prompt Techniques

### Prompt Composition

Combine multiple prompts:

```python
from langchain.prompts import PipelinePromptTemplate

# Full template with multiple parts
final_template = """{introduction}

{example}

{start}"""

# Define each part
introduction_template = """You are an expert at {expertise}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """Here's an example:
{example_text}"""
example_prompt = PromptTemplate.from_template(example_template)

start_template = """Now answer: {question}"""
start_prompt = PromptTemplate.from_template(start_template)

# Combine
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=PromptTemplate.from_template(final_template),
    pipeline_prompts=[
        ("introduction", introduction_prompt),
        ("example", example_prompt),
        ("start", start_prompt)
    ]
)

# Use it
result = pipeline_prompt.format(
    expertise="cooking",
    example_text="To make pasta, boil water first.",
    question="How do I make pizza?"
)
```

### Partial Prompts

Set some variables in advance:

```python
from langchain.prompts import PromptTemplate
from datetime import datetime

def get_date():
    return datetime.now().strftime("%Y-%m-%d")

prompt = PromptTemplate(
    template="Tell me a joke about {topic} for {date}",
    input_variables=["topic"],
    partial_variables={"date": get_date}
)

# Date is automatically filled
result = prompt.format(topic="programming")
```

---

## Best Practices

### 1. Clear Instructions

```python
# Bad
template = "Summarize this: {text}"

# Good
template = """
Summarize the following text in 3 bullet points.
Focus on the main ideas and key takeaways.

Text: {text}

Summary:
"""
```

### 2. Provide Context

```python
template = """
Context: You are a customer service agent for a tech company.
Tone: Professional and empathetic.

Customer message: {message}

Your response:
"""
```

### 3. Use Examples

```python
template = """
Convert temperatures. Examples:

32F = 0C
100F = 37.8C

Now convert: {temp}
"""
```

### 4. Specify Format

```python
template = """
List the capitals of these countries: {countries}

Format your answer as:
- Country: Capital
"""
```

### 5. Handle Edge Cases

```python
template = """
Answer the question: {question}

If you don't know the answer, say "I don't know" rather than making something up.
"""
```

---

## Common Patterns

### Pattern 1: Extract-Transform-Load

```python
# Extract information
extract_chain = LLMChain(llm=llm, prompt=extract_prompt)

# Transform it
transform_chain = TransformChain(transform=custom_transform)

# Load/use it
load_chain = LLMChain(llm=llm, prompt=load_prompt)

# Combine
etl_chain = SequentialChain(
    chains=[extract_chain, transform_chain, load_chain],
    input_variables=["raw_data"],
    output_variables=["final_result"]
)
```

### Pattern 2: Multi-Step Reasoning

```python
# Step 1: Understand the problem
step1_chain = LLMChain(llm=llm, prompt=understand_prompt)

# Step 2: Generate solution
step2_chain = LLMChain(llm=llm, prompt=solve_prompt)

# Step 3: Verify solution
step3_chain = LLMChain(llm=llm, prompt=verify_prompt)

reasoning_chain = SequentialChain(
    chains=[step1_chain, step2_chain, step3_chain]
)
```

### Pattern 3: Conditional Processing

```python
def route_based_on_content(inputs):
    if "technical" in inputs["text"].lower():
        return "technical_chain"
    else:
        return "general_chain"

# Use with router chain
```

---

## Next Steps

- Practice with the [examples](../examples/) in this repo
- Build the [projects](../projects/) to apply these concepts
- Learn about [Memory](04-memory-context.md) for stateful chains
- Explore [Agents](05-agents-tools.md) for dynamic chains

---

**Previous**: [Core Concepts](02-core-concepts.md) | **Next**: [Memory and Context](04-memory-context.md)
