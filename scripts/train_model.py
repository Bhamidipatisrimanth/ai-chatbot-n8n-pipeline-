# ── Step 1: Install dependencies ──────────────────────────────────────
# Run this line first in Colab as a separate cell:
# !pip install -q unsloth transformers datasets peft trl accelerate bitsandbytes

# ── Step 2: Import libraries ──────────────────────────────────────────
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json, os

# ── Step 3: Load base model ───────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama-chat-bnb-4bit",
    max_seq_length = 512,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
)

# ── Step 4: Load dataset ──────────────────────────────────────────────
raw_data = [
    {"text": "### Question: What is machine learning?\n### Answer: Machine learning is a branch of AI where systems learn from data to make predictions without being explicitly programmed."},
    {"text": "### Question: What is deep learning?\n### Answer: Deep learning uses neural networks with many layers to learn complex patterns from large amounts of data."},
    {"text": "### Question: What is overfitting?\n### Answer: Overfitting happens when a model learns the training data too well and performs poorly on new, unseen data."},
    {"text": "### Question: What is a neural network?\n### Answer: A neural network is a system of algorithms modeled after the human brain that recognizes patterns in data."},
    {"text": "### Question: What is gradient descent?\n### Answer: Gradient descent is an optimization algorithm that minimizes a model's error by adjusting weights in the direction of steepest descent."},
    {"text": "### Question: What is transfer learning?\n### Answer: Transfer learning reuses a pre-trained model on a new but related task, saving time and computational resources."},
    {"text": "### Question: What is a training dataset?\n### Answer: A training dataset is the labeled data used to teach a machine learning model to make predictions."},
    {"text": "### Question: What is NLP?\n### Answer: Natural Language Processing (NLP) is a field of AI focused on enabling machines to understand and generate human language."},
    {"text": "### Question: What is a transformer model?\n### Answer: A transformer is a neural network architecture that uses self-attention mechanisms to process sequential data like text."},
    {"text": "### Question: What is fine-tuning?\n### Answer: Fine-tuning adapts a pre-trained model to a specific task by continuing to train it on a smaller, task-specific dataset."},
]

dataset = Dataset.from_list(raw_data)

# ── Step 5: Train ─────────────────────────────────────────────────────
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        output_dir = "outputs",
        logging_steps = 1,
        fp16 = True,
        optim = "adamw_8bit",
    ),
)
trainer.train()
print("✅ Training complete!")

# ── Step 6: Push to HuggingFace Hub ──────────────────────────────────
HF_USERNAME = "YOUR_HF_USERNAME"   # 👈 replace this
HF_TOKEN = "YOUR_HF_TOKEN"         # 👈 replace this (huggingface.co/settings/tokens)

model.push_to_hub(f"{HF_USERNAME}/ai-faq-chatbot", token=HF_TOKEN)
tokenizer.push_to_hub(f"{HF_USERNAME}/ai-faq-chatbot", token=HF_TOKEN)
print(f"✅ Model pushed to: huggingface.co/{HF_USERNAME}/ai-faq-chatbot")
