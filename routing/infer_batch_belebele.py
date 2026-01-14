from transformers import AutoTokenizer
from modeling_qwen3_moe import Qwen3MoeForCausalLM
import torch
import json
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="../Qwen3-30B-A3B")
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    model_path = args.model_path
    langs = args.lang.split(',')

    model = Qwen3MoeForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    def format_dataset(dataset):
        formatted_data = []
        for item in dataset:
            passage = item['flores_passage']
            question = item['question']
            option_a = item['mc_answer1']
            option_b = item['mc_answer2']
            option_c = item['mc_answer3']
            option_d = item['mc_answer4']

            options = f"A. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}"
            prompt = f"{passage}\n{question}\n{options}\n".strip()

            formatted_data.append(prompt)
        return formatted_data


    for lang in langs:
        model.moe_recorder.clear()

        path = 'datasets/belebele/data/' + lang + '.jsonl'

        with open(path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

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
        model.moe_recorder.dump(f"{args.output_path}/{lang}.json")


if __name__ == "__main__":
    main()