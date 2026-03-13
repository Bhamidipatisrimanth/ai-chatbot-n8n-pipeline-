import json, csv, os

faq_data = [
    {"question": "What is machine learning?", 
     "answer": "Machine learning is a branch of AI where systems learn from data to make predictions without being explicitly programmed."},
    {"question": "What is deep learning?", 
     "answer": "Deep learning uses neural networks with many layers to learn complex patterns from large amounts of data."},
    {"question": "What is overfitting?", 
     "answer": "Overfitting happens when a model learns the training data too well and performs poorly on new, unseen data."},
    {"question": "What is a neural network?", 
     "answer": "A neural network is a system of algorithms modeled after the human brain that recognizes patterns in data."},
    {"question": "What is gradient descent?", 
     "answer": "Gradient descent is an optimization algorithm that minimizes a model's error by adjusting weights in the direction of steepest descent."},
    {"question": "What is transfer learning?", 
     "answer": "Transfer learning reuses a pre-trained model on a new but related task, saving time and computational resources."},
    {"question": "What is a training dataset?", 
     "answer": "A training dataset is the labeled data used to teach a machine learning model to make predictions."},
    {"question": "What is NLP?", 
     "answer": "Natural Language Processing (NLP) is a field of AI focused on enabling machines to understand and generate human language."},
    {"question": "What is a transformer model?", 
     "answer": "A transformer is a neural network architecture that uses self-attention mechanisms to process sequential data like text."},
    {"question": "What is fine-tuning?", 
     "answer": "Fine-tuning adapts a pre-trained model to a specific task by continuing to train it on a smaller, task-specific dataset."},
]

os.makedirs("data", exist_ok=True)

with open("data/faq_dataset.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "answer"])
    writer.writeheader()
    writer.writerows(faq_data)

with open("data/faq_dataset.jsonl", "w") as f:
    for item in faq_data:
        prompt = f"### Question: {item['question']}\n### Answer:"
        f.write(json.dumps({"text": f"{prompt} {item['answer']}"}) + "\n")

print(f"✅ Dataset saved: {len(faq_data)} examples")
print("Files: data/faq_dataset.csv, data/faq_dataset.jsonl")
