"""
Configuration and Helper Utilities
===================================

This module provides configuration management and helper functions
for LangChain applications.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """
    Configuration manager for LangChain applications
    """
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    SERPAPI_API_KEY: Optional[str] = os.getenv("SERPAPI_API_KEY")
    
    # Model Settings
    DEFAULT_MODEL: str = "gpt-3.5-turbo"
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 500
    
    # Memory Settings
    DEFAULT_MEMORY_WINDOW: int = 5
    
    # Vector Store Settings
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    FAISS_INDEX_DIR: str = "./faiss_index"
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required API keys are present
        
        Returns:
            bool: True if all required keys are present
        """
        if not cls.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not found in environment variables")
            return False
        return True
    
    @classmethod
    def display(cls) -> None:
        """Display current configuration (without showing keys)"""
        print("\nCurrent Configuration:")
        print("-" * 50)
        print(f"OpenAI API Key: {'✓ Set' if cls.OPENAI_API_KEY else '✗ Not Set'}")
        print(f"Anthropic API Key: {'✓ Set' if cls.ANTHROPIC_API_KEY else '✗ Not Set'}")
        print(f"HuggingFace Token: {'✓ Set' if cls.HUGGINGFACEHUB_API_TOKEN else '✗ Not Set'}")
        print(f"Default Model: {cls.DEFAULT_MODEL}")
        print(f"Default Temperature: {cls.DEFAULT_TEMPERATURE}")
        print(f"Default Max Tokens: {cls.DEFAULT_MAX_TOKENS}")
        print("-" * 50)


def check_env_setup() -> bool:
    """
    Check if the environment is properly set up
    
    Returns:
        bool: True if environment is properly configured
    """
    print("Checking environment setup...")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("✗ .env file not found")
        print("  → Copy .env.example to .env and add your API keys")
        return False
    else:
        print("✓ .env file found")
    
    # Validate configuration
    if not Config.validate():
        print("✗ Configuration invalid")
        print("  → Ensure OPENAI_API_KEY is set in .env")
        return False
    else:
        print("✓ Configuration valid")
    
    print("\n✓ Environment setup complete!\n")
    return True


def format_docs(docs) -> str:
    """
    Format a list of documents into a single string
    
    Args:
        docs: List of document objects
        
    Returns:
        str: Formatted document string
    """
    return "\n\n".join(doc.page_content for doc in docs)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for a given text
    
    Args:
        text: Input text
        model: Model name for token counting
        
    Returns:
        int: Estimated token count
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Rough approximation if tiktoken not available
        return len(text) // 4


def truncate_text(text: str, max_tokens: int = 500, model: str = "gpt-3.5-turbo") -> str:
    """
    Truncate text to fit within token limit
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        model: Model name for token counting
        
    Returns:
        str: Truncated text
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens) + "..."
    except ImportError:
        # Rough approximation
        char_limit = max_tokens * 4
        if len(text) <= char_limit:
            return text
        return text[:char_limit] + "..."


def print_section(title: str) -> None:
    """
    Print a formatted section header
    
    Args:
        title: Section title
    """
    print("\n" + "="*60)
    print(title.center(60))
    print("="*60 + "\n")


def print_result(label: str, content: str, width: int = 60) -> None:
    """
    Print a formatted result
    
    Args:
        label: Result label
        content: Result content
        width: Width of the output
    """
    print(f"\n{label}:")
    print("-" * width)
    print(content)
    print("-" * width)


# Example usage
if __name__ == "__main__":
    # Check environment
    check_env_setup()
    
    # Display config
    Config.display()
    
    # Test token counting
    sample_text = "This is a sample text for testing token counting functionality."
    token_count = count_tokens(sample_text)
    print(f"\nSample text token count: {token_count}")
