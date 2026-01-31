# LangChain & LangGraph Learning Guide: From Theory to Practice

<div align="center">

**A comprehensive, project-based learning repository for mastering LangChain, LangGraph, and LLM Technologies**

*Learn to build powerful AI applications with LangChain and LangGraph through hands-on projects and in-depth tutorials*

</div>

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [What is LangChain?](#what-is-langchain)
- [Getting Started](#getting-started)
- [Learning Path](#learning-path)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
- [Hands-On Projects](#hands-on-projects)
- [Best Practices](#best-practices)
- [Resources](#resources)
- [Contributing](#contributing)

---

## 🎯 Introduction

Welcome to the ultimate LangChain and LangGraph learning repository! This guide is designed to take you from zero to hero in building AI-powered applications using LangChain, LangGraph, and related LLM technologies. Whether you're a beginner or an experienced developer, you'll find structured tutorials, real-world projects, and best practices to master these powerful frameworks.

### What You'll Learn

- 🧠 **Core LangChain Concepts**: Models, Prompts, Chains, Agents, Memory
- 🔄 **LangGraph Fundamentals**: State machines, cyclic graphs, agent workflows
- 🔨 **Practical Implementation**: Build real-world applications step-by-step
- 🚀 **Advanced Techniques**: RAG, Function Calling, Custom Tools, Streaming
- 📊 **Best Practices**: Error handling, optimization, production deployment
- 🎓 **Project-Based Learning**: Complete projects from simple to complex
- 🌐 **Ecosystem Integration**: LangSmith, LangServe, and related tools

---

## 🤔 What is LangChain?

LangChain is a powerful framework for developing applications powered by language models. It provides:

- **Composable Components**: Modular tools for working with LLMs
- **Chains**: Sequences of calls to components and other chains
- **Agents**: Systems that use LLMs to decide which actions to take
- **Memory**: Persistence of state between chain/agent calls
- **Integrations**: Connect to various data sources and APIs

### Why LangChain?

1. **Abstraction**: Simplifies complex LLM interactions
2. **Modularity**: Mix and match components easily
3. **Production-Ready**: Built for scalable applications
4. **Rich Ecosystem**: Extensive integrations and tools
5. **Active Community**: Rapidly evolving with best practices

## 🔄 What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with:

- **State Management**: Built-in state persistence for complex workflows
- **Cyclic Graphs**: Create loops and conditional branches in agent flows
- **Multi-Agent Systems**: Coordinate multiple agents working together
- **Human-in-the-Loop**: Easy integration of human feedback and approval
- **Streaming Support**: Real-time updates for long-running operations

### Why LangGraph?

1. **Complex Workflows**: Handle sophisticated agent interactions
2. **State Control**: Explicit control over agent state and memory
3. **Flexibility**: Build both simple and complex agent architectures
4. **Debugging**: Better visibility into agent decision-making
5. **Production Scale**: Designed for reliable, stateful applications

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic understanding of Python programming
- OpenAI API key (or other LLM provider credentials)
- Git for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shfarhaan/LangChainScratch.git
   cd LangChainScratch
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Quick Start Example

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short introduction about {topic}"
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.invoke({"topic": "LangChain"})
print(result["text"])
```

---

## 📖 Learning Path

Follow this structured path to master LangChain:

### 🟢 Beginner Level (Weeks 1-2)

1. **Fundamentals**
   - Setting up your environment
   - Understanding LLMs and their capabilities
   - Basic prompt engineering
   - Working with LangChain models

2. **Simple Chains**
   - Creating basic chains
   - Prompt templates
   - Output parsers
   - Sequential chains

### 🟡 Intermediate Level (Weeks 3-4)

3. **Advanced Components**
   - Memory systems (Conversation, Summary, Entity)
   - Document loaders and text splitters
   - Vector stores and embeddings
   - Retrieval systems

4. **Practical Applications**
   - Question-answering systems
   - Chatbots with memory
   - Document analysis tools
   - Data extraction pipelines

### 🔴 Advanced Level (Weeks 5-6)

5. **Complex Architectures**
   - Agents and tools
   - Custom tools and toolkits
   - RAG (Retrieval-Augmented Generation)
   - Multi-agent systems

6. **Production Deployment**
   - Error handling and retries
   - Streaming responses
   - Monitoring and logging
   - Performance optimization
   - Cost management

---

## 📁 Project Structure

```
LangChainScratch/
├── README.md                     # Main guide (you are here)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
│
├── docs/                         # Detailed documentation
│   ├── 01-getting-started.md    # Setup and basics
│   ├── 02-core-concepts.md      # LangChain fundamentals
│   ├── 03-prompts-chains.md     # Prompts and chains
│   ├── 04-memory-context.md     # Memory management
│   ├── 05-agents-tools.md       # Agents and custom tools
│   ├── 06-rag-vectorstores.md   # RAG implementation
│   ├── 07-advanced-topics.md    # Advanced patterns
│   ├── 08-best-practices.md     # Production guidelines
│   ├── 09-langgraph-intro.md    # LangGraph fundamentals
│   └── QUICK_REFERENCE.md       # Quick reference guide
│
├── notebooks/                    # Jupyter tutorials
│   ├── 01_basic_llm_usage.ipynb
│   ├── 02_prompt_templates.ipynb
│   ├── 03_chains_demo.ipynb
│   ├── 04_memory_systems.ipynb
│   ├── 05_agents_tools.ipynb
│   ├── 06_rag_implementation.ipynb
│   └── 07_langgraph_basics.ipynb
│
├── projects/                     # Hands-on projects
│   ├── beginner/
│   │   ├── simple_chatbot/      # ✅ Complete with code
│   │   ├── text_summarizer/     # ✅ Complete with code
│   │   └── qa_system/           # ✅ Complete with code (RAG)
│   ├── intermediate/
│   │   ├── langgraph_agent/     # ✅ Complete with code
│   │   ├── document_qa/         # ✅ Complete with code
│   │   ├── web_scraper_chatbot/ # ✅ Complete with code
│   │   └── code_analyzer/       # ✅ Complete with code
│   └── advanced/
│       ├── multi_agent_system/  # ✅ Complete with code
│       ├── research_assistant/  # ✅ Complete with code
│       └── automated_workflow/  # ✅ Complete with code
│
├── examples/                     # Code snippets
│   ├── basic_examples.py        # ✅ 10 working examples
│   ├── chain_examples.py
│   ├── agent_examples.py
│   ├── langgraph_examples.py
│   └── rag_examples.py
│
└── utils/                        # Helper utilities
    ├── config.py
    ├── helpers.py
    └── custom_tools.py
```

---

## 🧩 Core Concepts

### 1. **Models (LLMs)**
Language models are the foundation. LangChain supports multiple providers:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- HuggingFace
- Local models (Ollama, LlamaCPP)

### 2. **Prompts**
Structured inputs to models with:
- Templates for dynamic content
- Few-shot examples
- System messages
- Output parsers

### 3. **Chains**
Sequences of operations:
- LLMChain: Basic LLM + prompt
- Sequential chains: Multiple steps
- Router chains: Dynamic routing
- Custom chains: Your logic

### 4. **Memory**
Maintaining context:
- ConversationBufferMemory
- ConversationSummaryMemory
- EntityMemory
- VectorStoreMemory

### 5. **Agents**
Autonomous decision-makers:
- ReAct agents
- Conversational agents
- Custom agents
- Tool selection logic

### 6. **Tools**
Extend capabilities:
- Search tools (Google, DuckDuckGo)
- Calculators
- APIs and databases
- Custom tools

### 7. **Vector Stores**
Semantic search:
- ChromaDB
- Pinecone
- FAISS
- Weaviate

### 8. **LangGraph**
Stateful, multi-actor applications:
- State graphs with nodes and edges
- Cyclic workflows and loops
- Conditional branching
- Multi-agent coordination
- Human-in-the-loop integration

---

## 🛠️ Hands-On Projects

### Beginner Projects

#### 1. Simple Chatbot ✅
Build a basic conversational AI with memory
- **Skills**: LLMs, Prompts, Memory
- **Duration**: 2-3 hours
- **Status**: Complete with implementation
- **[View Project](projects/beginner/simple_chatbot/)**

#### 2. Text Summarizer ✅
Create a document summarization tool
- **Skills**: Chains, Document loaders
- **Duration**: 2-3 hours
- **Status**: Complete with implementation
- **[View Project](projects/beginner/text_summarizer/)**

#### 3. Q&A System ✅
Question-answering with RAG (Retrieval-Augmented Generation)
- **Skills**: Embeddings, Retrieval, Vector Stores
- **Duration**: 3-4 hours
- **Status**: Complete with implementation
- **[View Project](projects/beginner/qa_system/)**

### Intermediate Projects

#### 4. LangGraph Agent ✅
Build stateful agent workflows with LangGraph
- **Skills**: LangGraph, State machines, Multi-step agents
- **Duration**: 1-2 days
- **Status**: Complete with implementation
- **[View Project](projects/intermediate/langgraph_agent/)**

#### 5. Document Q&A with Advanced RAG ✅
Enterprise-grade document Q&A system
- **Skills**: Advanced RAG, Vector stores, Chains
- **Duration**: 1-2 days
- **Status**: Complete with implementation
- **[View Project](projects/intermediate/document_qa/)**

#### 6. Web Scraper Chatbot ✅
Chat about web content in real-time
- **Skills**: Web loaders, Agents, Tools
- **Duration**: 1-2 days
- **Status**: Complete with implementation
- **[View Project](projects/intermediate/web_scraper_chatbot/)**

#### 7. Code Analyzer ✅
Analyze and explain code with AI
- **Skills**: Custom parsers, Chains
- **Duration**: 2-3 days
- **Status**: Complete with implementation
- **[View Project](projects/intermediate/code_analyzer/)**

### Advanced Projects

#### 8. Multi-Agent System ✅
Collaborative AI agents with LangGraph
- **Skills**: LangGraph, Multi-agents, Coordination
- **Duration**: 3-5 days
- **Status**: Complete with implementation
- **[View Project](projects/advanced/multi_agent_system/)**

#### 9. Research Assistant ✅
Autonomous research tool with planning
- **Skills**: Agents, RAG, Web search, LangGraph
- **Duration**: 3-5 days
- **Status**: Complete with implementation
- **[View Project](projects/advanced/research_assistant/)**

#### 10. Automated Workflow ✅
End-to-end automation with state management
- **Skills**: All concepts, Production, LangGraph
- **Duration**: 5-7 days
- **Status**: Complete with implementation
- **[View Project](projects/advanced/automated_workflow/)**

---

## 💡 Best Practices

### Prompt Engineering
- Be specific and clear
- Use examples (few-shot learning)
- Iterate and test
- Handle edge cases

### Error Handling
```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    try:
        result = chain.run(input_text)
        print(f"Tokens used: {cb.total_tokens}")
    except Exception as e:
        print(f"Error: {e}")
```

### Cost Optimization
- Cache results when possible
- Use appropriate model sizes
- Implement rate limiting
- Monitor token usage

### Production Deployment
- Environment variable management
- Logging and monitoring
- Error handling and retries
- Security best practices
- Testing and validation

---

## 📚 Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Blog](https://blog.langchain.dev/)

### Learning Materials
- [LangChain Handbook](https://www.pinecone.io/learn/langchain/)
- [DeepLearning.AI Course](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- [YouTube Tutorials](https://www.youtube.com/results?search_query=langchain+tutorial)

### Community
- [Discord](https://discord.gg/langchain)
- [Twitter](https://twitter.com/langchainai)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/langchain)

### Related Tools
- [LangSmith](https://smith.langchain.com/) - Debugging and monitoring
- [LangServe](https://github.com/langchain-ai/langserve) - Deployment
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Found a bug? Open an issue
2. **Suggest Projects**: Ideas for new projects? We'd love to hear
3. **Improve Docs**: Better explanations or examples
4. **Add Examples**: Share your code snippets
5. **Fix Bugs**: Submit pull requests

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain Team**: For creating this amazing framework
- **OpenAI**: For pioneering language model APIs
- **Community Contributors**: For sharing knowledge and code
- **You**: For taking the time to learn and build!

---

## 🚦 Getting Help

- **Questions?** Open a [GitHub Issue](https://github.com/shfarhaan/LangChainScratch/issues)
- **Discussions?** Join our [Discord Community](https://discord.gg/langchain)
- **Quick Tips?** Check the [FAQ](docs/faq.md)

---

<div align="center">

**Happy Learning! 🎉**

*Star ⭐ this repository if you find it helpful!*

[Getting Started](docs/01-getting-started.md) | [Core Concepts](docs/02-core-concepts.md) | [First Project](projects/beginner/simple_chatbot/)

</div>
