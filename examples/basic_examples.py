"""
Basic LangChain Examples
========================

This file contains simple, self-contained examples demonstrating
core LangChain functionality.

Run each example independently by uncommenting the function call at the bottom.
"""

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def example_basic_llm():
    """Example 1: Basic LLM usage"""
    from langchain_openai import OpenAI
    
    print("\n=== Example 1: Basic LLM Usage ===")
    
    # Initialize LLM
    llm = OpenAI(temperature=0.7)
    
    # Simple query
    response = llm.invoke("What is artificial intelligence in one sentence?")
    print(f"Response: {response}")


def example_prompt_template():
    """Example 2: Using prompt templates"""
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    
    print("\n=== Example 2: Prompt Template ===")
    
    llm = OpenAI()
    
    # Create a template
    template = """
    You are a creative writer. Write a {length} {content_type} about {topic}.
    Make it engaging and {tone}.
    """
    
    prompt = PromptTemplate(
        input_variables=["length", "content_type", "topic", "tone"],
        template=template
    )
    
    # Format and use
    formatted_prompt = prompt.format(
        length="short",
        content_type="story",
        topic="a robot learning to paint",
        tone="inspiring"
    )
    
    response = llm.invoke(formatted_prompt)
    print(f"Generated story:\n{response}")


def example_simple_chain():
    """Example 3: Simple LLM Chain"""
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    print("\n=== Example 3: Simple Chain ===")
    
    llm = OpenAI()
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["product"],
        template="Generate 3 creative marketing slogans for {product}:"
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    result = chain.invoke({"product": "eco-friendly water bottles"})
    print(f"Marketing slogans:\n{result['text']}")


def example_sequential_chain():
    """Example 4: Sequential chains"""
    from langchain.chains import SimpleSequentialChain, LLMChain
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    
    print("\n=== Example 4: Sequential Chain ===")
    
    llm = OpenAI()
    
    # First chain: Generate a company name
    template1 = "Generate a creative name for a company that makes {product}:"
    prompt1 = PromptTemplate(input_variables=["product"], template=template1)
    chain1 = LLMChain(llm=llm, prompt=prompt1)
    
    # Second chain: Create a slogan for that company
    template2 = "Write a catchy slogan for a company named: {company_name}"
    prompt2 = PromptTemplate(input_variables=["company_name"], template=template2)
    chain2 = LLMChain(llm=llm, prompt=prompt2)
    
    # Combine chains
    overall_chain = SimpleSequentialChain(
        chains=[chain1, chain2],
        verbose=True
    )
    
    # Run
    result = overall_chain.invoke("smart home devices")
    print(f"Final result: {result}")


def example_chat_model():
    """Example 5: Chat models with message history"""
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage, AIMessage
    
    print("\n=== Example 5: Chat Model ===")
    
    chat = ChatOpenAI(temperature=0)
    
    messages = [
        SystemMessage(content="You are a helpful assistant that speaks like Shakespeare."),
        HumanMessage(content="Tell me about computers")
    ]
    
    response = chat.invoke(messages)
    print(f"Shakespeare-style response:\n{response.content}")


def example_conversation_memory():
    """Example 6: Conversation with memory"""
    from langchain_openai import OpenAI
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    
    print("\n=== Example 6: Conversation Memory ===")
    
    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory()
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # First interaction
    response1 = conversation.predict(input="Hi, my name is Alice and I love Python programming")
    print(f"Response 1: {response1}")
    
    # Second interaction - should remember the name
    response2 = conversation.predict(input="What's my name and what do I love?")
    print(f"Response 2: {response2}")


def example_token_counting():
    """Example 7: Monitor token usage and costs"""
    from langchain_openai import OpenAI
    from langchain.callbacks import get_openai_callback
    
    print("\n=== Example 7: Token Counting ===")
    
    llm = OpenAI()
    
    with get_openai_callback() as cb:
        result = llm.invoke("Explain quantum computing in simple terms")
        
        print(f"Result: {result}")
        print(f"\n--- Usage Stats ---")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost:.6f}")


def example_structured_output():
    """Example 8: Structured output parsing"""
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema
    from langchain.chains import LLMChain
    
    print("\n=== Example 8: Structured Output ===")
    
    # Define the structure we want
    response_schemas = [
        ResponseSchema(name="title", description="The title of the book"),
        ResponseSchema(name="author", description="The author of the book"),
        ResponseSchema(name="genre", description="The genre of the book"),
        ResponseSchema(name="summary", description="A brief summary")
    ]
    
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    
    # Create prompt
    template = """
    Given a book description, extract the following information:
    
    {format_instructions}
    
    Book description: {description}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["description"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Create chain
    llm = OpenAI()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run
    description = "1984 by George Orwell is a dystopian novel about totalitarianism"
    result = chain.invoke({"description": description})
    
    # Parse the output
    parsed = parser.parse(result["text"])
    print(f"Parsed output:\n{parsed}")


def example_list_output():
    """Example 9: Generate lists"""
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import CommaSeparatedListOutputParser
    from langchain.chains import LLMChain
    
    print("\n=== Example 9: List Output ===")
    
    parser = CommaSeparatedListOutputParser()
    format_instructions = parser.get_format_instructions()
    
    template = """
    List 5 {item_type}.
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["item_type"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    llm = OpenAI()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    result = chain.invoke({"item_type": "programming languages"})
    
    # Parse into list
    parsed_list = parser.parse(result["text"])
    print(f"Generated list: {parsed_list}")


def example_error_handling():
    """Example 10: Proper error handling"""
    from langchain_openai import OpenAI
    import time
    
    print("\n=== Example 10: Error Handling ===")
    
    llm = OpenAI()
    
    def safe_llm_call(prompt, max_retries=3):
        """Make LLM call with retry logic"""
        for attempt in range(max_retries):
            try:
                result = llm.invoke(prompt)
                return result
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Giving up.")
                    raise
    
    try:
        result = safe_llm_call("What is the meaning of life?")
        print(f"Success: {result}")
    except Exception as e:
        print(f"Final error: {e}")


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("LangChain Basic Examples")
    print("========================\n")
    
    # Uncomment the examples you want to run:
    
    example_basic_llm()
    # example_prompt_template()
    # example_simple_chain()
    # example_sequential_chain()
    # example_chat_model()
    # example_conversation_memory()
    # example_token_counting()
    # example_structured_output()
    # example_list_output()
    # example_error_handling()
    
    print("\n✓ Examples completed!")
