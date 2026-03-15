import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "srimanth23/ai-faq-chatbot"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map="cpu"
)

print("Loading your trained adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
print("✅ Model ready!")

def answer_question(question):
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Answer:" in result:
        result = result.split("### Answer:")[-1].strip()
    return result

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question", placeholder="What is machine learning?"),
    outputs=gr.Textbox(label="Answer"),
    title="🤖 AI FAQ Chatbot",
    description="Ask any AI/ML question!"
)

demo.launch()
