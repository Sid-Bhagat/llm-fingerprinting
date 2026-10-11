# llm-fingerprinting

Classifies which LLM family (GPT, Claude, Llama, PaLM) wrote a chatbot response, using `lmsys/chatbot_arena_conversations`.
Models: a TextCNN and a BiLSTM in PyTorch, trained on the prompt, the response, or both, with ablations, a cross-domain test, and TF-IDF / handcrafted-feature baselines.
All runs were done on CPU; every number in the report comes from the files in `results/`.

## What's in results/

- `results/*.json`: one file per training run (test metrics, confusion matrix, training history); 51 runs.
- `results/summary.csv`, `results/rq3_per_family.csv`: mean ± std over seeds, built by `src/aggregate.py`.
- `results/figures/`: seed-0 confusion matrix per configuration (`src/aggregate.py`).
- `results/report_figs/`: report figures (`src/make_figures.py`).
- `results/rq4/`: stylistic feature tables, box plot, openings, feature logistic regression (`src/features.py`, `src/feature_lr.py`).
- `results/tfidf/`: TF-IDF + logistic regression baseline (`src/tfidf_baseline.py`).
- `results/preds/`: test predictions from 6 reruns (no dataset text); `results/preds/runs/` holds their JSONs.
- `results/error_analysis/`: per-underlying-model error analysis (`src/error_analysis.py`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1   # CPU build; skip on a GPU machine
pip install -r requirements.txt
# Dataset is gated: request access at https://huggingface.co/datasets/lmsys/chatbot_arena_conversations,
# create a read token at https://huggingface.co/settings/tokens, then:
echo "HF_TOKEN=<your token>" > .env
# Check without data or training: python src/aggregate.py && python src/make_figures.py
```

## Reproduce

Run the steps in order; steps b–e run in the background, so wait for each to finish (`tail -f <log>`) before
starting the next. The run scripts are resumable (runs whose result JSON exists are skipped) and accept
`DRY_RUN=1` to print the commands without training (except `run_grid.sh`). `grid.log` only shows skip lines
because the original grid output was overwritten by a resume run.

```bash
# a. Build data/processed/{train,val,test}.parquet
python src/data.py

#    Train one model by hand (the first run also builds the shared vocab at data/processed/vocab.json):
#    --model cnn|lstm  --mode input|output|both  --seed N
#    --ablation none|strip_markdown|truncate100|drop_first_sentence  --train_domains ...  --test_domains ...
#    --max_out_len N (default 256 output / 192 both)  --patience N (default 2)
#    --save_preds [--preds_dir DIR] (test predictions CSV, default results/preds/)
#    e.g. python src/train.py --model cnn --mode output --seed 0

# b. RQ1/RQ2 grid: {cnn,lstm} x {input,output,both} x seeds {0,1,2} (18 runs)
nohup bash scripts/run_grid.sh > grid.log 2>&1 &

# c. RQ4 runs (18): strip_markdown and truncate100 ablations (cnn+lstm), output-only with
#    --max_out_len 192 (cnn, RQ2 control), --patience 4 (lstm)
nohup bash scripts/run_rq4.sh > rq4.log 2>&1 &

# d. RQ3 runs (12): train on general+math, test on code (cross-domain) and on general+math (in-domain
#    baseline); cnn, and lstm with --patience 4. When train and test domains are disjoint, the test set
#    pools train+val+test for the test domains and balances families (seed 42).
nohup bash scripts/run_rq3.sh > rq3.log 2>&1 &

# e. Saved test predictions + drop_first_sentence ablation, one after the other, into extras.log:
#    run_preds.sh: 6 reruns (cnn output and lstm output --patience 4, seeds 0-2) with --save_preds ->
#    results/preds/{run}.csv (question_id, model, family, domain, pred, correct); their JSONs go to
#    results/preds/runs/ (results/*.json untouched), then src/verify_preds.py checks each rerun's test
#    metrics match results/{run}.json exactly.
#    run_extra.sh: drop_first_sentence ablation (3 runs, cnn output, seeds 0-2). Removes the output up to and
#    including the first . ! ? or newline; outputs with no boundary or nothing after it are kept as-is
#    (counts are printed and stored under "ablation_stats" in the JSON).
nohup bash -c 'bash scripts/run_preds.sh && bash scripts/run_extra.sh' > extras.log 2>&1 &

# f. RQ4 stylistic analysis -> results/rq4/ (feature_stats.csv, feature_separation.csv,
#    top4_features_boxplot.png, openings.csv), then logistic regression on those features
#    (C tuned on val) -> results/rq4/feature_lr.json
python src/features.py
python src/feature_lr.py

# g. TF-IDF + logistic regression on outputs (word 1-2 grams, char 3-5 grams; 50k features, case kept;
#    C tuned on val) -> results/tfidf/ (word_1-2.json, char_3-5.json, summary.csv, top_ngrams.csv)
python src/tfidf_baseline.py

# h. Per-underlying-model error analysis from the saved predictions -> results/error_analysis/
#    (per_model.csv, per_model_heatmap.png; examples.md is gitignored because it contains dataset text)
python src/error_analysis.py

# i. results/summary.csv (mean ± std over seeds, grouped by model/mode/ablation/domains/
#    max_out_len/patience), results/rq3_per_family.csv (per-family F1 for domain-restricted runs)
#    + seed-0 confusion matrices in results/figures/
python src/aggregate.py

# j. Report figures -> results/report_figs/ (rq1_rq2_bars, confusion_side_by_side, ablations, rq3_per_family,
#    training_curves; 300 dpi PNG). Also prints the seed-averaged row-normalized confusion matrices
#    (cnn output, lstm output patience 4) and trainable-parameter counts. Reads results/*.json only.
python src/make_figures.py

# Smoke test (1 epoch, 20% of train; results kept out of results/)
for m in cnn lstm; do for mode in input output both; do
  python src/train.py --model $m --mode $mode --seed 0 --epochs 1 --train_frac 0.2 --log_every 20 --results_dir /tmp/smoke
done; done
```
