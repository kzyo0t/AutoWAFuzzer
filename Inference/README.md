### XSS

```
python inference.py \
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

### SQLI

```
python inference.py \
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
