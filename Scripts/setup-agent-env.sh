#!/bin/bash
echo " Setting up Agentic Environment..."

# Install agentic frameworks
pip install langchain langgraph chromadb

# Set up memory store
mkdir -p .memory/vector_store

# Configure API keys (from Codespaces secrets)
if [ -n "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY configured"
fi

echo " Agentic environment ready!"