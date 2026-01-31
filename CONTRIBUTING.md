# Contributing to LangChain Learning Guide

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Contribution Guidelines](#contribution-guidelines)
- [Style Guidelines](#style-guidelines)
- [Submitting Contributions](#submitting-contributions)

---

## Code of Conduct

This project follows a Code of Conduct to ensure a welcoming environment for everyone:

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome newcomers and help them learn
- **Be constructive**: Provide helpful feedback
- **Be patient**: Remember that everyone is learning

---

## How Can I Contribute?

### 1. Report Issues

Found a bug or error in the documentation?

- Check if the issue already exists
- Create a new issue with:
  - Clear title and description
  - Steps to reproduce (if applicable)
  - Expected vs. actual behavior
  - Environment details (OS, Python version, etc.)

### 2. Improve Documentation

Documentation can always be better:

- Fix typos or grammatical errors
- Clarify confusing explanations
- Add missing examples
- Improve code comments
- Update outdated information

### 3. Add Examples

Share your LangChain code:

- Add new example scripts
- Share interesting use cases
- Provide real-world implementations
- Include clear explanations

### 4. Create Projects

Build and share learning projects:

- Beginner-friendly tutorials
- Intermediate applications
- Advanced implementations
- Must include:
  - README with clear instructions
  - Well-commented code
  - Example usage
  - Requirements/dependencies

### 5. Enhance Existing Content

Improve what's already here:

- Add more detailed explanations
- Include additional test cases
- Optimize code performance
- Add error handling
- Include best practices

---

## Getting Started

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LangChainScratch.git
   cd LangChainScratch
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/shfarhaan/LangChainScratch.git
   ```

### Set Up Environment

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` file:
   ```bash
   cp .env.example .env
   # Add your API keys
   ```

### Create a Branch

Create a feature branch for your work:

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features or enhancements
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `example/` - New examples or projects

---

## Contribution Guidelines

### Documentation

When contributing documentation:

1. **Use Markdown**: All docs should be in Markdown format
2. **Be Clear**: Write for beginners, explain concepts thoroughly
3. **Include Examples**: Provide code examples where applicable
4. **Check Links**: Ensure all links work correctly
5. **Follow Structure**: Match existing documentation style

Example structure:
```markdown
# Topic Title

## Overview
Brief introduction to the topic.

## Concept Explanation
Detailed explanation with examples.

## Code Examples
```python
# Well-commented code
```

## Best Practices
Key points to remember.

## Common Issues
Known problems and solutions.
```

### Code Contributions

#### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Include type hints where appropriate
- Keep functions focused and small

Example:
```python
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    Summarize the given text to specified length.
    
    Args:
        text: The input text to summarize
        max_length: Maximum length of summary
        
    Returns:
        str: The summarized text
        
    Raises:
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Text cannot be empty")
    
    # Implementation here
    return summary
```

#### Requirements

- Add comments explaining complex logic
- Include error handling
- Write reusable, modular code
- Test your code before submitting
- Update requirements.txt if adding dependencies

#### Testing

Before submitting:

1. **Test your code**:
   ```bash
   python your_script.py
   ```

2. **Check for errors**:
   - Syntax errors
   - Import errors
   - Runtime errors

3. **Verify examples**:
   - All examples should run successfully
   - Output should match documentation

### Project Contributions

When adding a new project:

1. **Create proper structure**:
   ```
   projects/[level]/[project_name]/
   ├── README.md          # Detailed guide
   ├── main.py            # Main implementation
   ├── requirements.txt   # Project-specific deps (if any)
   └── examples/          # Usage examples
   ```

2. **README must include**:
   - Project overview and goals
   - What you'll learn
   - Prerequisites
   - Step-by-step implementation
   - How to run the project
   - Example usage
   - Common issues and solutions
   - Next steps

3. **Code requirements**:
   - Well-commented
   - Error handling
   - Clear variable names
   - Modular structure

---

## Style Guidelines

### Markdown

- Use `#` for headers (not underlines)
- Use triple backticks for code blocks
- Specify language for syntax highlighting
- Use lists for multiple items
- Include blank lines between sections

### Python

```python
# Good
def calculate_cost(tokens: int, rate: float = 0.002) -> float:
    """Calculate API cost based on token usage."""
    return tokens * rate

# Bad
def calc(t,r=0.002):
    return t*r
```

### Comments

```python
# Good: Explain why, not what
# Use exponential backoff to handle rate limits
wait_time = 2 ** attempt

# Bad: Obvious comment
# Set x to 5
x = 5
```

---

## Submitting Contributions

### Commit Messages

Write clear, descriptive commit messages:

```bash
# Good
git commit -m "Add RAG implementation example with Chroma"
git commit -m "Fix token counting in summarization chain"
git commit -m "Update getting started guide with new setup steps"

# Bad
git commit -m "Update"
git commit -m "Fix bug"
git commit -m "Changes"
```

Format:
- Use imperative mood ("Add" not "Added")
- Keep first line under 72 characters
- Add details in body if needed

### Pull Request Process

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**:
   - Go to GitHub and create PR
   - Fill out the PR template
   - Link any related issues

4. **PR Title Format**:
   ```
   [Type] Brief description
   
   Examples:
   [Feature] Add chatbot with custom personality
   [Fix] Correct token counting in examples
   [Docs] Update installation instructions
   [Example] Add web scraping example
   ```

5. **PR Description Should Include**:
   - What changes were made
   - Why the changes were made
   - How to test the changes
   - Any breaking changes
   - Screenshots (if applicable)

### Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged
- Celebrate your contribution! 🎉

---

## Recognition

Contributors are recognized in:
- GitHub contributors list
- Project acknowledgments
- Special mentions for significant contributions

---

## Questions?

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: [If applicable]

---

## Thank You!

Every contribution, no matter how small, helps make this resource better for everyone learning LangChain.

**Happy Contributing!** 🚀
