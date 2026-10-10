# LLM Fingerprinting Project

Classify which LLM family wrote a response. Course project with a 2-day deadline. Required models: CNN and RNN (BiLSTM) in PyTorch. Keep things simple and working; nothing fancy.

## Data
- Source: `lmsys/chatbot_arena_conversations` (Hugging Face, gated; read HF_TOKEN from env).
- Each row has two models answering one prompt. Turn every row into two examples: (model_a, conversation_a) and (model_b, conversation_b).
- Keep English, single-turn conversations only.
- Family map: gpt-4, gpt-3.5-turbo -> GPT; claude-v1, claude-instant-v1 -> Claude; vicuna-13b, vicuna-7b, koala-13b, alpaca-13b, wizardlm-13b, gpt4all-13b-snoozy, guanaco-33b -> Llama; palm-2 -> PaLM. Drop all other models (including llama-13b).
- Balance classes by downsampling every family to the size of the smallest one.
- Drop any response whose output mentions Claude, Anthropic, OpenAI, ChatGPT, GPT-4, GPT-3, Vicuna, LMSYS, Koala, Alpaca, PaLM, Bard, Google, or "trained by" (case-insensitive). Log how many were dropped per family.
- Split 70/15/15 BY question_id, never by row, so one prompt can't land in two splits.
- Domain label from prompt keywords: code fences or code keywords -> code; math words, arithmetic expressions (e.g. "2+2", dates excluded), "how many/much/long/old/far" with 2+ numbers, or mostly digits -> math; everything else -> general.
- Save to `data/processed/{train,val,test}.parquet` with columns: question_id, model, family, input, output, domain.
- Never modify `data/raw/`.

## Tokenization
- Word-level, built on train only. KEEP case, punctuation, and newlines (use a `<nl>` token) because formatting is part of the signal.
- Input-only mode: max 64 tokens. Output-only mode: max 256 tokens. Both mode: input (64) + `<sep>` + output (192).

## Models (`src/models.py`)
- TextCNN: embedding 128, filter sizes 3/4/5 with 100 filters each, max-pool, dropout 0.5.
- BiLSTM: embedding 128, hidden 128, 1 layer, max-pool over time, dropout 0.5.

## Training (`src/train.py`)
- CLI: `--model cnn|lstm --mode input|output|both --seed N --ablation none|strip_markdown|truncate100 --train_domains ... --test_domains ...`
- Adam with lr 1e-3, batch size 64, up to 10 epochs, early stopping on val macro-F1 (patience 2).
- Write `results/{run_name}.json` with accuracy, macro-F1, per-class F1, and the confusion matrix.
- Set all seeds. Each run should finish in under 10 minutes on a Colab GPU.

## Analysis
- `src/features.py`: per-family length, type-token ratio, bullet/header/code-block rates, and punctuation rates. Save tables and plots to `results/rq4/`.
- `src/aggregate.py`: build `results/summary.csv` (mean ± std over seeds) and confusion-matrix plots.

## Rules for Claude
- Work one phase at a time. After writing a file, explain in 2–3 sentences what it does.
- Never invent or estimate results. Report numbers only from `results/`.
- Keep `README.md` updated with the exact commands that reproduce every result.
