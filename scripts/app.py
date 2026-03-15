from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, gradio as gr

MODEL_NAME = "srimanth23/ai-faq-chatbot"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)
print("✅ Model loaded")

def answer_question(question: str) -> str:
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
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Answer:" in response:
        response = response.split("### Answer:")[-1].strip()
    return response

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Answer"),
    title="AI FAQ Chatbot",
    description="Ask any AI/ML question!"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
