"""
Basic Text Summarizer
=====================
Simple summarization for short to medium texts.

Usage:
    python summarizer_basic.py
"""

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import sys

load_dotenv()


class TextSummarizer:
    """Simple text summarization tool"""
    
    def __init__(self, summary_style="bullet"):
        """
        Initialize summarizer
        
        Args:
            summary_style: Style of summary (bullet, paragraph, key_points)
        """
        self.llm = OpenAI(temperature=0.3)
        self.style = summary_style
        self.templates = {
            "bullet": """
            Summarize the following text in 5-7 bullet points.
            Focus on the most important information.
            
            Text: {text}
            
            Bullet Point Summary:
            """,
            "paragraph": """
            Summarize the following text in a single concise paragraph.
            Capture the main ideas and conclusions.
            
            Text: {text}
            
            Paragraph Summary:
            """,
            "key_points": """
            Extract the 3 most important key points from the following text.
            
            Text: {text}
            
            Key Points:
            1.
            2.
            3.
            """
        }
    
    def summarize(self, text: str) -> str:
        """Generate summary"""
        
        template = self.templates.get(self.style, self.templates["bullet"])
        
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.invoke({"text": text})
        
        return result["text"]
    
    def summarize_file(self, filepath: str) -> str:
        """Summarize a text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.summarize(text)
        except FileNotFoundError:
            return f"Error: File '{filepath}' not found"
        except Exception as e:
            return f"Error reading file: {e}"


def main():
    """CLI interface"""
    print("\n" + "="*60)
    print("Text Summarizer")
    print("="*60)
    
    # Get summary style
    print("\nChoose summary style:")
    print("1. Bullet points (default)")
    print("2. Paragraph")
    print("3. Key points")
    
    choice = input("\nEnter choice (1-3) or press Enter for default: ").strip()
    
    style_map = {
        "1": "bullet",
        "2": "paragraph",
        "3": "key_points",
        "": "bullet"
    }
    
    style = style_map.get(choice, "bullet")
    summarizer = TextSummarizer(summary_style=style)
    
    print("\nEnter your text (type 'END' on a new line when done):")
    print("-" * 60)
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        except EOFError:
            break
    
    text = "\n".join(lines)
    
    if not text.strip():
        print("No text provided. Exiting.")
        return
    
    print("\nGenerating summary...")
    summary = summarizer.summarize(text)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary)
    print("="*60)


if __name__ == "__main__":
    main()
