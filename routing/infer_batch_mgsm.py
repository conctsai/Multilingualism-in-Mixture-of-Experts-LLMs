from transformers import AutoTokenizer
from modeling_qwen3_moe import Qwen3MoeForCausalLM
import torch
from tqdm import tqdm
from dataloader import load_mgsm_dataset

model_path = "Qwen3-30B-A3B"

model = Qwen3MoeForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")


def format_dataset(dataset):
    formatted_data = []
    for item in dataset:
        question = item['question']
        
        prompt = f"{question}\n".strip()

        formatted_data.append(prompt)
    return formatted_data


for lang in ['bn', 'de', 'en', 'es', 'fr', 'ja', 'sw', 'zh']:
    model.moe_recorder.clear()

    dataset = load_mgsm_dataset(language=lang)
    print(f"Loaded {len(dataset)} samples for language: {lang}")
    prompts = format_dataset(dataset)

    print(f"First 3 prompts: {prompts[:3]}")

    messages = [{"role": "user", "content": prompt} for prompt in prompts]

    texts = [tokenizer.apply_chat_template(
        [message],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for message in messages]

    batch_size = 16


    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to("cuda")
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1
            )

    model.moe_recorder.dump(f"routing/mgsm_result/mgsm_{lang}.json")