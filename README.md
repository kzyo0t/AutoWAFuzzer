# AutoWAFuzzer: An Adaptive Framework for Web Application Firewall Penetration Testing with a Multi-Agent System and RAG-Augmented Reinforcement Learning
<a href="https://github.com/kzyo0t/AutoWAFuzzer" target="_blank" style="text-decoration:none;">
  <button style="background-color:#24292e; color:white; border:none; padding:8px 16px; font-size:14px; border-radius:6px; display:flex; align-items:center; gap:8px; cursor:pointer;">
    <svg height="16" viewBox="0 0 16 16" width="16" aria-hidden="true" fill="white">
      <path d="M8 0C3.58 0 0 3.58 0 8a8.004 8.004 0 005.47 7.59c.4.07.55-.17.55-.38v-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.91-.88-1.15-.88-1.15-.72-.5.06-.49.06-.49.79.06 1.21.82 1.21.82.71 1.21 1.87.86 2.33.66.07-.52.28-.86.5-1.06-1.78-.2-3.64-.89-3.64-3.96 0-.87.31-1.59.82-2.15-.08-.2-.36-1.01.08-2.1 0 0 .67-.21 2.2.82a7.5 7.5 0 012-.27c.68 0 1.36.09 2 .27 1.52-1.03 2.2-.82 2.2-.82.44 1.09.16 1.9.08 2.1.51.56.82 1.28.82 2.15 0 3.08-1.87 3.76-3.65 3.96.29.25.54.73.54 1.48v2.2c0 .21.15.45.55.38A8.003 8.003 0 0016 8c0-4.42-3.58-8-8-8z"/>
    </svg>
    <span>View on GitHub</span>
  </button>
</a>


## 🔍 Introduction

**AutoWAFuzzer** is a novel multi-agent framework designed for automated Web Application Firewall (WAF) penetration testing. It integrates Large Language Models (LLMs), Reinforcement Learning (A2C), and Retrieval-Augmented Generation (RAG) using threat intelligence from MISP, enabling context-aware and adaptive payload generation against both rule-based and machine learning-based WAFs.

## 📊 Data

We provide curated and pre-processed datasets for reproducible evaluation:

| Source             | Usage                               | Types Covered | Size (SQLi / XSS)   |
|--------------------|--------------------------------------|----------------|----------------------|
| **MISP**           | Threat-informed retrieval (RAG)      | SQLi, XSS      | 1,500 / 26,000       |
| **Attack Grammars**| RL/LLM fine-tuning                   | SQLi, XSS      | 1,000,000 each       |



> Synthetic dataset created by Attack Grammars is available in the `data/` folder.
> MISP data collection can be found in the `RAG-Agent/` folder.

## ⚙️ Requirements

* Python >= 3.2.4
* PyTorch >= 2.4.0
* Transformer >= 4.44.1
* LangChain >= 0.3.26
* tqdm

Install all dependencies via:

```bash
pip install -r requirements.txt
```

## ▶️ Run Training Code with Published Data

AutoWAFuzzer is trained and executed using modular agent notebooks and scripts. Follow the steps below to reproduce results using the provided datasets:

Each core agent has its own training notebook:

| Folder            | Notebook                          | Description                              |
|-------------------|------------------------------------|------------------------------------------|
| `LLM-Agent/`       | `GPT_Neo_1M.ipynb`         | Pretrains or loads base language model       |
| `RL-Agent/`        | `A2C-GPT-Neo.ipynb`            | Trains the A2C-based language model agent |
| `RAG-Agent/`       | `vectorizeMISP.ipynb`            | Builds the vector database from MISP data |
| `Reward-Agent/`      | `Reward_Model_GPT_Neo_XSS_WAF_Brain.ipynb`         | Trains the reward model for WAF feedback |


## 🔎 Inference

After training, you can use the pretrained models to generate attack payloads with dynamic threat-informed retrieval using the following commands:

### 🚀 XSS Payload Generation

```bash
python Inference/inference.py \
  --model_dir "./gpt-neo-xss-a2c-generator" \
  --vector_db_path "./RAG-Agent/vectorDB/vectorize_xss_26k_MISP" \
  --output_csv "./Inference/Output/AutoWAFuzzer_XSS.csv" \
  --num_payloads 100 \
  --max_length 100 \
  --top_k 50 \
  --top_p 0.95 \
  --dynamic_retrieval \
  --retrieval_interval 10 \
  --payload_mode xss
```

### 🚀 SQLi Payload Generation

```bash
python Inference/inference.py \
  --model_dir "./gpt-neo-sqli-a2c-generator" \
  --vector_db_path "./RAG-Agent/vectorDB/vectorize_sqli_1k5_MISP" \
  --output_csv "./Inference/Output/AutoWAFuzzer_SQLI.csv" \
  --num_payloads 100 \
  --max_length 100 \
  --top_k 50 \
  --top_p 0.95 \
  --dynamic_retrieval \
  --retrieval_interval 10 \
  --payload_mode sqli
```

> 📝 Note: Ensure the vector DB paths point to pre-built MISP-embedded ChromaDB indexes and the model directory contains the trained A2C fine-tuned checkpoints.


## 📄 Publication

The AutoWAFuzzer framework is under active development and submitted for publication. Citation information will be updated here upon acceptance.

## 📬 Support or Contact
