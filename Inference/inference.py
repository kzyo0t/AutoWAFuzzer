import torch
import csv
import pandas as pd
import argparse
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import random
import html

def initialize_chromadb(vector_db_path):
    embedding = HuggingFaceEmbeddings(model_name="/content/drive/MyDrive/NCKH/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embedding)
    return vector_db

def retrieve_relevant_context_static(vector_db, num_queries=3, top_k=5, mode="xss"):
    xss_keywords = [
        "script", "img", "onerror", "svg", "iframe", "input", "form", "body", "a", "div", "object",
        "embed", "video", "math", "meta", "style", "base", "link", "plaintext", "xss"
    ]

    sqli_keywords = [
        "select", "union", "insert", "update", "delete", "drop", "and", "or", "not", "where", "from",
        "join", "table", "column", "--", "/*", "*/", "'", "\"", "`", "sleep", "benchmark", "information_schema"
    ]

    if mode == "xss":
        payload_keywords = xss_keywords
    elif mode == "sqli":
        payload_keywords = sqli_keywords
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    selected_queries = random.sample(payload_keywords, min(num_queries, len(payload_keywords)))
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    retrieved_contents = set()

    for query in selected_queries:
        docs = retriever.invoke(query)
        for doc in docs:
            retrieved_contents.add(doc.page_content.strip())

    return "\n".join(retrieved_contents)

def retrieve_relevant_context_dynamic(vector_db, partial_payload, top_k=5):
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(partial_payload)
    retrieved_contents = set(doc.page_content.strip() for doc in docs)
    return "\n".join(retrieved_contents)

def generate_payloads_with_rag(
    tokenizer,
    model,
    device,
    num_payloads,
    max_length,
    top_k,
    top_p,
    vector_db_path,
    dynamic_retrieval=False,
    retrieval_interval=10,
    rag_boost_weight=2.0,
    payload_mode="xss"
):
    total_payloads = []
    vector_db = initialize_chromadb(vector_db_path)

    for _ in range(num_payloads):
        input_ids = tokenizer.encode("<|endoftext|>", return_tensors='pt').to(device)
        generated_ids = input_ids
        rag_tokens = set()

        if not dynamic_retrieval:
            rag_context = retrieve_relevant_context_static(vector_db, mode=payload_mode)
        else:
            rag_context = ""

        rag_tokens = set(tokenizer.encode(rag_context, add_special_tokens=False))

        for step in range(max_length):
            if dynamic_retrieval and step % retrieval_interval == 0 and step > 0:
                partial_payload = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                rag_context = retrieve_relevant_context_dynamic(vector_db, partial_payload)
                rag_tokens = set(tokenizer.encode(rag_context, add_special_tokens=False))

            with torch.no_grad():
                outputs = model(generated_ids)
                logits = outputs.logits[:, -1, :]

                if rag_tokens:
                    boost_mask = torch.zeros_like(logits)
                    boost_mask[:, list(rag_tokens)] = rag_boost_weight
                    logits = logits + boost_mask

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        payload = tokenizer.decode(generated_ids[0][input_ids.size(1):], skip_special_tokens=True)
        payload = html.unescape(payload)
        payload = "".join(payload.split())[:max_length]
        total_payloads.append(payload)

    return total_payloads

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPTNeoForCausalLM.from_pretrained(args.model_dir).to(device)
    model.config.pad_token_id = tokenizer.eos_token_id

    generated_sequences = generate_payloads_with_rag(
        tokenizer=tokenizer,
        model=model,
        device=device,
        num_payloads=args.num_payloads,
        max_length=args.max_length,
        top_k=args.top_k,
        top_p=args.top_p,
        vector_db_path=args.vector_db_path,
        dynamic_retrieval=args.dynamic_retrieval,
        retrieval_interval=args.retrieval_interval,
        payload_mode=args.payload_mode
    )

    df = pd.DataFrame(generated_sequences, columns=["payloads"])
    df = df.drop_duplicates()
    print(f"Generated: {len(generated_sequences)}, After removing duplicates: {len(df)}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved deduplicated payloads to {args.output_csv}")

    print("\nSample Payloads:")
    for i, p in enumerate(df["payloads"].head(5), 1):
        print(f"{i}: {p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Payloads using GPT-Neo + RAG")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to pretrained model directory")
    parser.add_argument("--vector_db_path", type=str, required=True, help="Path to Chroma vector database")
    parser.add_argument("--num_payloads", type=int, default=1000, help="Number of payloads to generate")
    parser.add_argument("--max_length", type=int, default=75, help="Maximum length of each payload")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path for payloads")
    parser.add_argument("--dynamic_retrieval", action='store_true', help="Enable dynamic retrieval mode")
    parser.add_argument("--retrieval_interval", type=int, default=10, help="Retrieval interval (only for dynamic mode)")
    parser.add_argument("--payload_mode", type=str, choices=["xss", "sqli"], default="xss",
                        help="Type of payload to guide RAG context (xss or sqli)")
    args = parser.parse_args()

    main(args)
