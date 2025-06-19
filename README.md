# AutoWAFuzzer: An Adaptive Framework for Web Application Firewall Penetration Testing with a Multi-Agent System and RAG-Augmented Reinforcement Learning

## 🔍 Introduction

**AutoWAFuzzer** is a novel multi-agent framework designed for automated Web Application Firewall (WAF) penetration testing. It integrates Large Language Models (LLMs), Reinforcement Learning (A2C), and Retrieval-Augmented Generation (RAG) using threat intelligence from MISP, enabling context-aware and adaptive payload generation against both rule-based and machine learning-based WAFs.

## 📊 Data

We provide curated and pre-processed datasets for reproducible evaluation:

<!--* **xxxx** -->
<!--* **xxxx** -->
<!--* **xxxxx** -->

| Source         | Use                                | Type  | Size            |
|----------------|-------------------------------------|-------|------------------|
| **MISP**       | Threat-informed retrieval (RAG)     | SQLi  | 1,500 samples    |
|                |                                     | XSS   | 26,000 samples   |
| **Attack Grammars** ([11]) | RL/LLM fine-tuning              | SQLi  | 1 million samples |
|                |                                     | XSS   | 1 million samples |

> Download links and formats are available in the `data/` folder.

## ⚙️ Requirements

* Python >= 3.8
* PyTorch >= 1.11
* PyG (PyTorch Geometric)
* NetworkX
* tqdm, scikit-learn, matplotlib

Install all dependencies via:

```bash
pip install -r requirements.txt
```

## ▶️ Run Code with Published Data

You can run AutoWAFuzzer using the provided datasets with the following command:

```bash
python main.py --dataset xxx --model xxxxh --epochs 100
```

Training logs will be saved to `logs/`. To visualize:

```bash
tensorboard --logdir logs/
```

Make sure the corresponding dataset is placed inside the `data/` directory.

## 📄 Publication

The AutoWAFuzzer framework is under active development and submitted for publication. Citation information will be updated here upon acceptance.

## 📬 Support or Contact

For any query, issue, or collaboration request, feel free to contact:

* [duypt@uit.edu.vn](mailto:duypt@uit.edu.vn)
* [truonghuu.tram@singaporetech.edu.sg](mailto:truonghuu.tram@singaporetech.edu.sg)
