# Web Scraper Chatbot

## Overview

Build an intelligent chatbot that can scrape web pages, extract content, and answer questions about the information in real-time. This project combines web scraping with RAG to create a dynamic Q&A system.

## What You'll Learn

- **Web Scraping**: Extract content from websites using BeautifulSoup and requests
- **Dynamic RAG**: Build vector stores from web content on-the-fly
- **Content Processing**: Clean and structure web content for Q&A
- **Tool Integration**: Combine web scraping tools with LangChain agents
- **Real-time Processing**: Handle fresh content without pre-processing
- **URL Validation**: Check and sanitize URLs safely

## Prerequisites

- Completed [Document Q&A](../document_qa/) project
- Understanding of web scraping basics
- Familiarity with HTML structure

## Project Structure

```
web_scraper_chatbot/
├── README.md              # This file
├── scraper_basic.py       # Basic web scraping Q&A
├── scraper_agent.py       # Agent-based scraper with tools
├── scraper_advanced.py    # Multi-URL scraper with caching
└── utils.py               # Helper functions
```

## Quick Start

### 1. Install Dependencies

```bash
pip install langchain langchain-openai beautifulsoup4 requests html2text
```

### 2. Run Basic Version

```bash
python scraper_basic.py
```

### 3. Try with a URL

```
Enter URL: https://example.com
Question: What is this page about?
```

## Implementation Guide

### Step 1: Basic Web Scraping

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load web page
loader = WebBaseLoader("https://example.com")
documents = loader.load()

# Process and create QA system
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
qa_chain = RetrievalQA.from_llm(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever()
)
```

### Step 2: Agent with Web Scraping Tool

```python
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool

class WebScraperTool(BaseTool):
    name = "web_scraper"
    description = "Scrapes a webpage and answers questions about its content"
    
    def _run(self, url: str) -> str:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content

# Create agent with tool
tools = [WebScraperTool()]
agent = initialize_agent(
    tools,
    llm=ChatOpenAI(),
    agent="zero-shot-react-description"
)
```

### Step 3: Multi-URL Scraper

```python
from langchain_community.document_loaders import WebBaseLoader

# Scrape multiple URLs
urls = [
    "https://example.com/page1",
    "https://example.com/page2"
]

loader = WebBaseLoader(urls)
documents = loader.load()

# Add metadata for source tracking
for i, doc in enumerate(documents):
    doc.metadata["url"] = urls[i]
    doc.metadata["page_num"] = i
```

## Features

### 1. Content Cleaning

```python
import html2text
from bs4 import BeautifulSoup

def clean_html_content(html):
    """Convert HTML to clean markdown"""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    return h.handle(html)
```

### 2. URL Validation

```python
from urllib.parse import urlparse

def is_valid_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
```

### 3. Caching

```python
import pickle
import hashlib

def cache_webpage(url, content):
    """Cache webpage content"""
    cache_key = hashlib.md5(url.encode()).hexdigest()
    with open(f"cache/{cache_key}.pkl", "wb") as f:
        pickle.dump(content, f)
```

### 4. Rate Limiting

```python
import time

class RateLimiter:
    def __init__(self, delay=1.0):
        self.delay = delay
        self.last_request = 0
    
    def wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()
```

## Use Cases

### 1. News Article Q&A

```python
# Scrape news article
url = "https://news-site.com/article"
loader = WebBaseLoader(url)
article = loader.load()

# Ask questions about the article
qa_chain = create_qa_chain(article)
result = qa_chain("What are the main points of this article?")
```

### 2. Documentation Assistant

```python
# Scrape documentation pages
docs_urls = [
    "https://docs.example.com/intro",
    "https://docs.example.com/api",
    "https://docs.example.com/examples"
]

# Create searchable knowledge base
loader = WebBaseLoader(docs_urls)
documents = loader.load()
vectorstore = create_vectorstore(documents)
```

### 3. Research Helper

```python
# Scrape multiple research papers
papers = [
    "https://arxiv.org/abs/paper1",
    "https://arxiv.org/abs/paper2"
]

# Compare and analyze
agent = create_research_agent(papers)
result = agent.run("Compare the methodologies of these papers")
```

## Best Practices

### 1. Respect robots.txt

```python
from urllib.robotparser import RobotFileParser

def can_scrape(url):
    """Check if scraping is allowed"""
    rp = RobotFileParser()
    rp.set_url(f"{url}/robots.txt")
    rp.read()
    return rp.can_fetch("*", url)
```

### 2. Error Handling

```python
try:
    loader = WebBaseLoader(url)
    docs = loader.load()
except Exception as e:
    print(f"Failed to load {url}: {e}")
    return None
```

### 3. Content Size Limits

```python
MAX_CONTENT_SIZE = 50000  # characters

def truncate_content(content, max_size=MAX_CONTENT_SIZE):
    """Limit content size"""
    if len(content) > max_size:
        return content[:max_size] + "..."
    return content
```

### 4. User-Agent Headers

```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Educational Bot)'
}

loader = WebBaseLoader(url, header_template=headers)
```

## Advanced Features

### 1. JavaScript-Rendered Pages

For pages that require JavaScript:

```python
# Use Playwright or Selenium
from langchain_community.document_loaders import PlaywrightURLLoader

loader = PlaywrightURLLoader(urls=[url])
documents = loader.load()
```

### 2. Recursive Scraping

```python
def scrape_site_recursive(base_url, max_depth=2):
    """Scrape a website recursively"""
    visited = set()
    to_visit = [(base_url, 0)]
    documents = []
    
    while to_visit:
        url, depth = to_visit.pop(0)
        if url in visited or depth > max_depth:
            continue
        
        visited.add(url)
        # Scrape and find links
        # Add new links to to_visit
        
    return documents
```

### 3. Content Extraction Strategies

```python
from bs4 import BeautifulSoup

def extract_main_content(html):
    """Extract main article content"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
        tag.decompose()
    
    # Find main content
    main = soup.find('main') or soup.find('article') or soup.find('body')
    return main.get_text(strip=True)
```

## Troubleshooting

### Issue: Connection errors
**Solution**: Add retry logic and timeouts

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
```

### Issue: HTML parsing errors
**Solution**: Use different parsers

```python
# Try lxml first, fall back to html.parser
try:
    soup = BeautifulSoup(html, 'lxml')
except:
    soup = BeautifulSoup(html, 'html.parser')
```

### Issue: Rate limiting
**Solution**: Add delays and respect rate limits

```python
time.sleep(1)  # Wait between requests
```

## Security Considerations

1. **Validate URLs**: Check for malicious URLs
2. **Limit Domains**: Whitelist allowed domains
3. **Content Filtering**: Filter sensitive content
4. **Timeout Limits**: Prevent hanging requests
5. **Size Limits**: Prevent memory issues

## Next Steps

1. Add support for PDFs and other file types
2. Implement intelligent link following
3. Add content summarization
4. Create web UI with Streamlit
5. Deploy as API service

## Resources

- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/)
- [Requests Documentation](https://requests.readthedocs.io/)
- [Web Scraping Best Practices](https://www.scrapingbee.com/blog/web-scraping-best-practices/)

## Related Projects

- [Document Q&A](../document_qa/) - Core RAG concepts
- [Research Assistant](../../advanced/research_assistant/) - Multi-source research
- [Code Analyzer](../code_analyzer/) - Code-specific scraping

---

**Next**: Try building the [Code Analyzer](../code_analyzer/) to analyze code repositories!
