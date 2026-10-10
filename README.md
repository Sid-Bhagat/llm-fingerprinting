# llm-fingerprinting
## Reproduce

```bash
pip install -r requirements.txt
echo "HF_TOKEN=<your token>" > .env   # dataset is gated

# 1. Build data/processed/{train,val,test}.parquet
python src/data.py
```
