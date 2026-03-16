# AI FAQ Chatbot with n8n Automation Pipeline

## Live Project
- HuggingFace Space: https://huggingface.co/spaces/srimanth23/ai-faq-chatbot-api
- Telegram Bot: @aifaq_srimanth_bot
- Model: https://huggingface.co/srimanth23/ai-faq-chatbot

## What it does
- Fine-tuned TinyLlama model on custom AI/ML FAQ dataset
- Deployed on HuggingFace Spaces as live API
- n8n workflow connects Telegram to AI model automatically
- Users chat with AI directly on Telegram

## Architecture
User (Telegram) → n8n Trigger → HuggingFace API → n8n → Telegram Reply

## Tech Stack
- n8n Cloud — workflow automation
- Python + HuggingFace — model training
- TinyLlama + LoRA — fine-tuned AI model
- Gradio — API serving
- Telegram Bot API — user interface
- Google Colab — free GPU training

## Project Structure
```
ai-chatbot-n8n-pipeline-/
├── data/
│   ├── faq_dataset.csv
│   └── faq_dataset.jsonl
├── scripts/
│   ├── generate_dataset.py
│   ├── train_model.py
│   └── app.py
├── n8n_workflows/
│   └── chatbot_workflow.json
├── requirements.txt
└── README.md
```

## Setup Instructions
1. Clone repo
   git clone https://github.com/Bhamidipatisrimanth/ai-chatbot-n8n-pipeline-

2. Generate dataset
   python scripts/generate_dataset.py

3. Train model on Google Colab
   - Open scripts/train_model.py in Colab
   - Set T4 GPU
   - Run all cells

4. Deploy API to HuggingFace Spaces
   - Upload scripts/app.py
   - Upload requirements.txt

5. Set up n8n workflow
   - Import n8n_workflows/chatbot_workflow.json
   - Configure Telegram Bot token

## Note
First response may take 30-60 seconds as free tier
services wake up from sleep. This is normal behavior.
