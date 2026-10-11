# llm-fingerprinting
## Reproduce

```bash
pip install -r requirements.txt
echo "HF_TOKEN=<your token>" > .env   # dataset is gated

# 1. Build data/processed/{train,val,test}.parquet
python src/data.py

# 2. Train one model (the first run also builds the shared vocab at data/processed/vocab.json)
#    --model cnn|lstm  --mode input|output|both  --seed N
#    --ablation none|strip_markdown|truncate100|drop_first_sentence  --train_domains ...  --test_domains ...
#    --max_out_len N (default 256 output / 192 both)  --patience N (default 2)
#    --save_preds [--preds_dir DIR] (test predictions CSV, default results/preds/)
python src/train.py --model cnn --mode output --seed 0

# 3. RQ1/RQ2 grid: {cnn,lstm} x {input,output,both} x seeds {0,1,2} (18 runs, resumable:
#    runs whose results/<run>.json exists are skipped)
bash scripts/run_grid.sh

# 4. RQ4 stylistic analysis -> results/rq4/ (feature_stats.csv, feature_separation.csv,
#    top4_features_boxplot.png, openings.csv), then logistic regression on those features
#    (C tuned on val) -> results/rq4/feature_lr.json
python src/features.py
python src/feature_lr.py

# 5. RQ4 runs (18, resumable): strip_markdown and truncate100 ablations (cnn+lstm), output-only with
#    --max_out_len 192 (cnn, RQ2 control), --patience 4 (lstm). DRY_RUN=1 prints commands only.
bash scripts/run_rq4.sh

# 6. RQ3 runs (12, resumable): train on general+math, test on code (cross-domain) and on
#    general+math (in-domain baseline); cnn, and lstm with --patience 4. When train and test
#    domains are disjoint, the test set pools train+val+test for the test domains and balances
#    families (seed 42). DRY_RUN=1 prints commands only.
bash scripts/run_rq3.sh

# 7. results/summary.csv (mean ± std over seeds, grouped by model/mode/ablation/domains/
#    max_out_len/patience), results/rq3_per_family.csv (per-family F1 for domain-restricted runs)
#    + seed-0 confusion matrices in results/figures/
python src/aggregate.py

# 8. Report figures -> results/report_figs/ (rq1_rq2_bars, confusion_side_by_side, ablations, rq3_per_family,
#    training_curves; 300 dpi PNG). Also prints the seed-averaged row-normalized confusion matrices
#    (cnn output, lstm output patience 4) and trainable-parameter counts. Reads results/*.json only.
python src/make_figures.py

# 9. Saved test predictions (6 reruns, resumable): cnn output and lstm output --patience 4, seeds 0-2, with
#    --save_preds -> results/preds/{run}.csv (question_id, model, family, domain, pred, correct). Their JSONs go
#    to results/preds/runs/ (results/*.json untouched); src/verify_preds.py then checks each rerun's test
#    metrics match results/{run}.json exactly. DRY_RUN=1 prints commands only.
#    This one command runs both run_preds.sh and run_extra.sh (step 12), one after the other, into a single extras.log.
nohup bash -c 'bash scripts/run_preds.sh && bash scripts/run_extra.sh' > extras.log 2>&1 &

# 10. Per-underlying-model error analysis from the saved predictions -> results/error_analysis/
#     (per_model.csv, per_model_heatmap.png; examples.md is gitignored because it contains dataset text)
python src/error_analysis.py

# 11. TF-IDF + logistic regression on outputs (word 1-2 grams, char 3-5 grams; 50k features, case kept;
#     C tuned on val) -> results/tfidf/ (word_1-2.json, char_3-5.json, summary.csv, top_ngrams.csv)
python src/tfidf_baseline.py

# 12. drop_first_sentence ablation (3 runs, resumable): cnn output, seeds 0-2. Removes the output up to and
#     including the first . ! ? or newline; outputs with no boundary or nothing after it are kept as-is
#     (counts are printed and stored under "ablation_stats" in the JSON). DRY_RUN=1 prints commands only.
#     Already run by the step-9 command (log in extras.log); to run it alone:
nohup bash scripts/run_extra.sh >> extras.log 2>&1 &
#     then rerun step 7 (and 8) to include them in results/summary.csv

# Smoke test (1 epoch, 20% of train; results kept out of results/)
for m in cnn lstm; do for mode in input output both; do
  python src/train.py --model $m --mode $mode --seed 0 --epochs 1 --train_frac 0.2 --log_every 20 --results_dir /tmp/smoke
done; done
```
