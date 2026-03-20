# 🤖 HuggingFace Model Selector

Automatically find, evaluate, and deploy the best HuggingFace models for your specific task.

## 🚀 Live Demo

**Model Selector Interface:** http://16.59.36.61:7860

**Translation API Docs:** http://16.59.36.61:8000/docs

## ☁️ Deployment

Deployed on **AWS EC2** (t2.small, Ubuntu 24.04, us-east-2)
- Gradio web interface running as a systemd service on port 7860
- FastAPI translation API running as a systemd service on port 8000

## ✨ Features

- **Natural Language Understanding**: Describe your task in plain English
- **Intelligent Search**: Finds relevant models on HuggingFace Hub
- **Multi-criteria Scoring**: Evaluates models based on downloads, recency, license, size, and performance
- **Real Benchmarking**: Tests actual inference speed and memory usage
- **Deployment Generation**: Creates production-ready FastAPI/Gradio/Docker code
- **Comprehensive Documentation**: Auto-generated README and configuration files

## 🛠️ Installation
```bash
git clone https://github.com/honourjesus/agentic-model-selector.git
cd agentic-model-selector
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python gradio_app.py
```

## 👤 Author

**HonourJesus** — [github.com/honourjesus](https://github.com/honourjesus)
