# llm-fingerprinting
## Reproduce

```bash
pip install -r requirements.txt
echo "HF_TOKEN=<your token>" > .env   # dataset is gated

# 1. Build data/processed/{train,val,test}.parquet
python src/data.py

# 2. Train one model (the first run also builds the shared vocab at data/processed/vocab.json)
#    --model cnn|lstm  --mode input|output|both  --seed N
#    --ablation none|strip_markdown|truncate100  --train_domains ...  --test_domains ...
python src/train.py --model cnn --mode output --seed 0

# 3. RQ1/RQ2 grid: {cnn,lstm} x {input,output,both} x seeds {0,1,2} (18 runs, resumable:
#    runs whose results/<run>.json exists are skipped)
bash scripts/run_grid.sh

# 4. results/summary.csv (mean ± std over seeds) + seed-0 confusion matrices in results/figures/
python src/aggregate.py

# Smoke test (1 epoch, 20% of train; results kept out of results/)
for m in cnn lstm; do for mode in input output both; do
  python src/train.py --model $m --mode $mode --seed 0 --epochs 1 --train_frac 0.2 --log_every 20 --results_dir /tmp/smoke
done; done
```
