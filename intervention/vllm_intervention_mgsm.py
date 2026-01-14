import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
import json
import argparse
import re
from langdetect import detect_langs, DetectorFactory
from vllm import LLM, SamplingParams
from dataloader import load_mgsm_dataset
from externel.PolyMath.instruction import query_dic
from externel.PolyMath.eval.scripts import math_equal
from transformers import AutoTokenizer

def extract_boxed_content(text):
    pattern = re.compile(r'boxed{')
    text = text.replace(' ', '')

    matches = pattern.finditer(text)
    results = []
    for match in matches:
        start_pos = match.end()
        brace_count = 1
        i = start_pos
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            results.append(text[start_pos:i-1])
    return results

def infer(lang, model_path, seed):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype=torch.bfloat16,
        seed=seed
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=4096
    )

    results = []

    dataset = load_mgsm_dataset(language=lang)

    messages = [
        {
            "role": "user",
            "content": data['question'] + '\n' + query_dic[lang]
        } for data in dataset
    ]

    print(f"Loaded {len(dataset)} samples for language: {lang}")
    print(f"First 3 messages: {messages[:3]}")

    inputs = [tokenizer.apply_chat_template(
        [msg],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for msg in messages]

    responses = llm.generate(inputs, sampling_params=sampling)

    for i, data in enumerate(dataset):
        output = responses[i].outputs[0].text.strip()
        results.append({
            'idx': i,
            'question': data['question'],
            'answer': data['answer'],
            'prediction': output
        })
        
    return results

def ablation_experts(args):
    L = 48
    E = 128

    input_dir = args.input_dir

    # get all json files under input_dir
    json_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.json')
    ]

    # get lang list by {lang}.json
    lang_list = [
        os.path.basename(f).split('.json')[0]
        for f in json_files
    ]

    lang_num = len(json_files)

    activation_tokens = torch.zeros(size=(lang_num, L, E), dtype=torch.int64)
    gate_values_sum = torch.zeros(size=(lang_num, L, E), dtype=torch.float32)
    total_tokens = torch.zeros(size=(lang_num,), dtype=torch.int64)

    for idx, item in enumerate(json_files):
        with open(item, 'r') as f:
            data = json.load(f)
        
        for layer_str, layer_data in data.items():
            layer = int(layer_str)
            for expert_str, expert_data in layer_data.items():
                if expert_str == 'total_tokens':
                    continue
                expert = int(expert_str)
                activation_tokens[idx, layer, expert] = expert_data['activation_tokens']
                gate_values_sum[idx, layer, expert] = expert_data['gate_values_sum']
                
            total_tokens[idx] = layer_data['total_tokens']
            
        print(f"Processed {lang_list[idx]}")

    activation_tokens = activation_tokens / total_tokens.unsqueeze(-1).unsqueeze(-1)

    # select top-k in expert wise and mask others
    k = args.k
    topk_values, topk_indices = torch.topk(activation_tokens, k=k, dim=-1)
    activation_tokens = torch.zeros_like(activation_tokens).scatter_(-1, topk_indices, topk_values)

    # normalize in the language dimension
    activation_tokens = activation_tokens / (activation_tokens.sum(dim=0, keepdim=True) + 1e-10)

    if args.shared:
        result = {}
        for layer in range(args.start_layer, args.end_layer + 1):
            for expert in range(E):
                value = activation_tokens[:, layer, expert]
                if torch.any(value > 0) and torch.all(value < args.thr):
                    result.setdefault(str(layer), []).append(expert)
        return result
    

    else:
        target_lang = args.target_lang.split(',')
        target_idxs = [lang_list.index(tl) for tl in target_lang]
        target = torch.sum(activation_tokens[target_idxs, :, :], dim=0)
        result = {}

        for layer in range(args.start_layer, args.end_layer + 1):
            for expert in range(E):
                value = target[layer, expert].item()
                if value > args.thr:
                    result.setdefault(str(layer), []).append(expert)

        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        default='mgsm_result'
    )
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        default=8
    )
    parser.add_argument(
        '--thr',
        type=float,
        required=True,
        default=0.4
    )
    parser.add_argument(
        '--lang',
        type=str,
        required=True
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../Qwen3-30B-A3B",
        help="Path to model (default ../Qwen3-30B-A3B)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)"
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=20,
        help="Start layer for steering (default 20)"
    )
    parser.add_argument(
        "--end_layer",
        type=int,
        default=29,
        help="End layer for steering (default 29)"
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="Whether to mask shared experts (default False)"
    )
    args = parser.parse_args()

    langs = [s.strip() for s in args.lang.split(",") if s.strip()]

    # stop steering
    steer_data = {}
    steer_data["lambda"] = 0.0
    steer_data["steer_dict"] = {}
    json.dump(steer_data, open("steer/steer_data.json", "w"), indent=4)

    print(f"Stop steering data saved to steer/steer_data.json")

    ablation_data = ablation_experts(args)
    json.dump(ablation_data, open("intervention/intervention.json", "w"), indent=4)

    print(f"Ablation data saved to intervention/intervention.json")

    for lang in langs:

        results = infer(lang, args.model_path, args.seed)

        json.dump(results, open(f"intervention/mgsm_output/mgsm_target_{args.target_lang}_lang_{lang}_seed{args.seed}_start{args.start_layer}_end{args.end_layer}_k{args.k}_thr{args.thr}_shared{args.shared}.json", "w"), indent=4, ensure_ascii=False)

        acc = 0
        lang_consistency = 0

        DetectorFactory.seed = 42

        for res in results:
            extracted_pred = extract_boxed_content(res['prediction'])
            extracted_pred = extracted_pred[0] if len(extracted_pred) > 0 else None
            acc_binary = math_equal(extracted_pred, res['answer'])
            acc += 1 if acc_binary else 0

            lang_detect_result = detect_langs(res['prediction'])
            should_be_lang = lang if lang != "zh" else "zh-cn"
            if len(lang_detect_result) >= 2:
                lang_consistency += 0
            elif lang_detect_result[0].lang.lower() == should_be_lang:
                lang_consistency += 1
            else:
                lang_consistency += 0

        accuracy = acc / len(results) if len(results) > 0 else 0
        consistency = lang_consistency / len(results) if len(results) > 0 else 0
        print(f"Accuracy for language {lang}: {accuracy:.4f}")
        print(f"Consistency for language {lang}: {consistency:.4f}")


        # save results to result.jsonl
        out_path = "intervention/mgsm_result.jsonl"
        result = {
            "language": lang,
            "target_lang": args.target_lang if not args.shared else "None",
            "accuracy": accuracy,
            "consistency": consistency,
            "seed": args.seed,
            "start_layer": args.start_layer,
            "end_layer": args.end_layer,
            "k": args.k,
            "thr": args.thr,
            "shared": args.shared
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()