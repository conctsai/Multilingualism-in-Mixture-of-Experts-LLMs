<h1>Understanding Multilingualism in Mixture-of-Experts LLMs: Routing Mechanism, Expert Specialization, and Layerwise Steering</h1>

<img src="https://img.shields.io/badge/License-MIT-blue" alt="license">

## 📖 Abstract

Mixture-of-Experts (MoE) architectures have shown strong multilingual capabilities, yet the internal mechanisms underlying performance gains and cross-language differences remain insufficiently understood. In this work, we conduct a systematic analysis of MoE models, examining routing behavior and expert specialization across languages and network depth.

Our analysis reveals that multilingual processing in MoE models is highly structured: routing aligns with linguistic families, expert utilization follows a clear layerwise pattern, and high-resource languages rely on shared experts while low-resource languages depend more on language-exclusive experts despite weaker performance. Layerwise interventions further show that early and late MoE layers support language-specific processing, whereas middle layers serve as language-agnostic capacity hubs. Building on these insights, we propose a routing-guided steering method that adaptively guides routing behavior in middle layers toward shared experts associated with dominant languages at inference time, leading to consistent multilingual performance improvements.

------

## 📋 Catalogue

* [Preparations](#preparations)
* [Routing Analysis](#routing-analysis)
* [Layerwise Intervention](#intervention-experiments)
* [Routing-guided Steering](#routing-guided-steering)

## ⚙️ Preparations

To set up the environment and download the necessary datasets, run the automated setup script.

```bash
bash scripts/setup.sh
```

### What this script does:

1. **Environment:** Creates a Conda environment with **Python 3.11**.

2. **Dependencies:** Installs required packages from `requirements.txt`.

3. **VLLM Patching:** Crucially, it modifies the `qwen3_moe` **model_executor** within the `vllm` library to support intervention and steering mechanisms.

4. **Datasets:** Downloads the following datasets into the `datasets/` folder:

   - **[facebook/belebele](https://huggingface.co/datasets/facebook/belebele)**: Multilingual reading comprehension.

   - **[openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)**: Evaluation of translation and multilingual capabilities.

   - **[juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm)**: Multilingual Grade School Math.

   - **[google/xquad](https://huggingface.co/datasets/google/xquad)**: Multilingual Question Answering.

   - **[Qwen/PolyMath](https://huggingface.co/datasets/Qwen/PolyMath)**: Mathematics evaluation across diverse languages.

## 🔍 Routing Analysis

To get the routing information from the `belebele` dataset, run:

```bash
bash scripts/run_routing.sh
```

> Before running, ensure you have specified `MODEL_PATH` and `CUDA_VISIBLE_DEVICES` inside the script.

The routing information will be saved in the `routing/` directory.

## 🧪 Intervention Experiments

We perform layerwise interventions to understand the role of specific layers in language processing. These experiments can be executed for `flores`, `mgsm`, or `xquad`.

```bash
# Usage: bash scripts/run_intervention_{dataset_name}.sh
bash scripts/run_intervention_flores.sh
```

- **Outputs:** Model generations are stored in `intervention/{dataset_name}_output/`.
- **Results:** Quantitative results are saved in `intervention/{dataset_name}_result.jsonl`.

### 🛠️ Intervention Hyperparameters

| **Parameter** | **Default** | **Description**                                              |
| ------------- | ----------- | ------------------------------------------------------------ |
| `k`           | `15`        | Number of top experts selected from the target language distribution. |
| `thr`         | `0.4`       | Threshold to filter exclusive experts.                       |
| `seed`        | `2025`      | Random seed for reproducibility across different simulation runs. |
| `input_dir`   | `"routing"` | Path to the directory containing pre-computed expert activation frequencies. |
| `lang`        | -           | Evaluation language.                                         |
| `target_lang` | -           | The language whose expert distribution guides intervention.  |
| `start_layer` | -           | The index of the first layer in the MoE block where the intervention begins. |
| `end_layer`   | -           | The index of the last layer in the MoE block where the intervention ends. |

## 🧭 Routing-Guided Steering

To run the steering experiments on the PolyMath dataset:

```bash
bash scripts/run_steer_polymath.sh
```

- **Outputs:** Model generations are stored in `steer/polymath_output/`.
- **Results:** Quantitative results are saved in `steer/polymath_result.jsonl`.

### 🛠️ Steer Hyperparameters

| **Parameter** | **Default** | **Description**                                              |
| ------------- | ----------- | ------------------------------------------------------------ |
| `k`           | `15`        | Number of top experts selected from the target language distribution. |
| `thr`         | `0.4`       | Threshold to filter exclusive experts.                       |
| `seed`        | `2025`      | Random seed for reproducibility across different simulation runs. |
| `input_dir`   | `"routing"` | Path to the directory containing pre-computed expert activation frequencies. |
| `lang`        | -           | Evaluation language.                                         |
| `target_lang` | -           | Comma-separated languages whose expert distribution guides steering, the first language is the source of coefficient. |
| `lambda`      | -           | Steer lambda.                                                |
| `start_layer` | -           | The index of the first layer in the MoE block where the steering begins. |
| `end_layer`   | -           | The index of the last layer in the MoE block where the steering ends. |