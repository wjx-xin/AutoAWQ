import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B-AWQ"

print("Loading quantized model...")
model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(quant_path)

prompt = "请用中文解释什么是量子计算，并说明它与经典计算的主要区别。"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.model.device)

print(f"\nPrompt: {prompt}")
print("\n--- Generating response ---\n")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

response = tokenizer.decode(
    output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
)
print(response)
print("\n--- Done ---")
