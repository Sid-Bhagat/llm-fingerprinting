# Report Notes: LLM Fingerprinting

Running notes for the final report. The report draft (sections 1–4) is in the Claude Doc; these notes hold the raw numbers and the Lessons Learned material. [TODO] = still waiting on results.

---

## 1. Problem & Dataset

**Task:** Given an LLM response (and optionally the user prompt), predict which LLM family produced it. Four classes: GPT, Claude, Llama, PaLM.

**Source:** `lmsys/chatbot_arena_conversations` (Hugging Face, gated, CC license, arXiv:2306.05685).
- 33,000 rows, 20 models. Each row has one user prompt answered by two randomly assigned models.
- Access was granted on accepting the terms. Loaded with a read-only HF token in `.env` (gitignored).

**Filtering pipeline (src/data.py):**
1. Single-turn English rows: 25,938 rows, 51,876 responses (25,928 unique question_ids, so a few repeat).
2. Each row becomes two examples: (model, input = user turn, output = assistant turn).
3. Map to families, drop other models, empty outputs, and exact duplicate (input, output) pairs.
4. Drop self-identifying responses (Claude, Anthropic, OpenAI, ChatGPT, GPT-4, GPT-3, Vicuna, LMSYS, Koala, Alpaca, PaLM, Bard, Google, "trained by"). Dropped rather than masked, since a placeholder token would itself be a signal.
5. Domain labels from prompt keywords.
6. Split 70/15/15 by question_id (GroupShuffleSplit, seed 42).
7. Balance within each split by downsampling to the smallest family (seed 42).

**Family map:**
| Family | Models |
|---|---|
| GPT | gpt-4, gpt-3.5-turbo |
| Claude | claude-v1, claude-instant-v1 |
| Llama | vicuna-13b, vicuna-7b, koala-13b, alpaca-13b, wizardlm-13b, gpt4all-13b-snoozy, guanaco-33b |
| PaLM | palm-2 |

Dropped (different base models): RWKV-4-Raven-14B, chatglm-6b, stablelm-tuned-alpha-7b, dolly-v2-12b, oasst-pythia-12b, mpt-7b-chat, fastchat-t5-3b.

**Counts after filtering:** GPT 6,944 · Claude 5,142 · Llama 16,895 · PaLM 2,305 · total 31,286

**Self-ID drops:** GPT 22 · Claude 28 · Llama 35 · PaLM 7 (92 total, under 0.3%; not a major leakage source)

**Final splits:** train 6,332 · val 1,416 · test 1,444 · total 9,192 (≈2,298 per family, balanced in every split, 0 question_id overlap)

**Domain counts (final data):**
| Family | code | general | math |
|---|---|---|---|
| GPT | 291 | 1,847 | 160 |
| Claude | 290 | 1,848 | 160 |
| Llama | 298 | 1,841 | 159 |
| PaLM | 321 | 1,794 | 183 |

**Domain rules:** code first (fences, code keywords); math = clear math terms (equation, integral, prime, divisible, solve for, percent...), arithmetic like `2+2` with dates removed first, "how many/much/long/old/far" + 2 numbers, or mostly digits/operators; else general. Widening the math rules raised math from 516 to 662 examples (~7%).

**Limitations:**
- PaLM sets the class size (~9.2k total).
- The Llama class = community LLaMA-1 fine-tunes, several trained on ChatGPT outputs (Vicuna on ShareGPT), so expect GPT–Llama confusion.
- Heuristic domain labels: ~16/20 sampled math labels correct; false positives like "Compare 9/11 and January 6th."
- Math too small (~160/family) to be a held-out test domain.
- 2023-era models.
- Vocab is built on all of train including code prompts (no labels), a minor caveat for RQ3.

---

## 2. Models & Training

**Tokenization:** words, single punctuation marks, `<nl>`; case kept. Vocab from train only, capped at 30k (the cap was hit, so min-freq 2 didn't matter). Saved to `data/processed/vocab.json` and shared by all runs (not committed; it rebuilds identically from the seeded split).

**Lengths:** input 64 · output 256 · both = 64 + `<sep>` + 192

**TextCNN:** emb 128 → filters 3/4/5 × 100 → max-pool → dropout 0.5 → linear(4)
**BiLSTM:** emb 128 → 1-layer BiLSTM hidden 128 → max-pool over time → dropout 0.5 → linear(4)
Padding is excluded from the max-pool in both models.

**Training:** Adam lr 1e-3, batch 64, ≤10 epochs, early stopping on val macro-F1 (patience 2), best checkpoint evaluated on test, seeds 0/1/2.

**Ablation implementation:** applied to the output text in every split. strip_markdown removes code fences, headers, bullets, `*`/`_`/backticks, repeated blank lines. truncate100 keeps the first 100 output tokens.
**Domain filtering:** val is filtered by --train_domains (it drives early stopping); test by --test_domains.

**Smoke test (1 epoch, 20% of train, seed 0):**
| Run | val macro-F1 | sec/epoch |
|---|---|---|
| cnn input | 0.209 | 0.7 |
| cnn output | 0.303 | 1.7 |
| cnn both | 0.265 | 1.3 |
| lstm input | 0.227 | 1.4 |
| lstm output | ~0.35 (output cut off) | — |
| lstm both | ~0.31 (output cut off) | — |

**Runtime:** estimated ≤68 min for the full grid on CPU (16 threads); ran locally, no GPU needed. LSTM output/both took ~85% of the time. The first real CNN output run took ~52 s.

**Environment:** CPU torch 2.14.1, pandas 3.0.6, datasets 5.1.0; requirements pinned. CPU torch install: `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1`.

---

## 3. Results (RQ1, RQ2): DONE, drafted in the doc

| Mode | CNN macro-F1 | CNN acc | LSTM macro-F1 | LSTM acc |
|---|---|---|---|---|
| Output only | 0.688 ± 0.005 | 0.685 ± 0.006 | 0.646 ± 0.060 | 0.643 ± 0.058 |
| Input + output | 0.675 ± 0.012 | 0.672 ± 0.013 | 0.648 ± 0.010 | 0.650 ± 0.007 |
| Input only | 0.244 ± 0.012 | 0.261 ± 0.008 | 0.252 ± 0.009 | 0.259 ± 0.005 |

**LSTM output per seed:**
| Seed | Test macro-F1 | Best epoch | Stopped |
|---|---|---|---|
| 0 | 0.664 | 8 | 10 |
| 1 | 0.694 | 7 | 9 |
| 2 | 0.579 | 2 | 4 |
Seed 2 dipped at epoch 3 and patience 2 ended it. Without seed 2, mean is 0.679. Kept in the reported mean (dropping it would be cherry-picking).

**Patience-4 sensitivity check (LSTM output, seeds 0–2): DONE**
| Seed | patience 2 | patience 4 |
|---|---|---|
| 0 | 0.664 | 0.664 (identical; best epoch already reached) |
| 1 | 0.694 | 0.693 (best epoch 10) |
| 2 | 0.579 | **0.700** (best of all six LSTM runs) |
| Mean | 0.646 ± 0.060 | **0.686 ± 0.019** |
- With patience 4 the LSTM ties the CNN (0.688 ± 0.005). Same accuracy; the CNN is just more consistent and less sensitive to the stopping rule.
- Patience 2 stays the main reported result (chosen before seeing results); patience 4 is reported as a sensitivity check.
- The original seed-1 run and its patience-4 rerun had identical val curves (0.5338, 0.5661, ... 0.6873), which shows runs are reproducible.
- Seeds 1 and 2 both had a val dip that patience 2 treated as the end of training; with patience 4 both kept improving. The dips were noise, not overfitting.

**CNN output seed-0 confusion (row-normalized recall):** Claude 0.83 · PaLM 0.74 · Llama 0.67 · GPT 0.52. GPT → Llama = 0.37 (largest error).
[TODO: LSTM output confusion matrix; does it show the same GPT→Llama pattern?]

**RQ1 takeaways:**
- Both models ≈2.7× chance from the output alone.
- The CNN is more reliable (std 0.005); the LSTM's lower mean comes from one early-stopped seed. With patience 4 the LSTM ties the CNN (0.686 vs 0.688).
- Hypothesis: the cues are local n-grams (openings, stock phrases, formatting), which CNN filters catch directly. To be tested in RQ4.
- GPT–Llama confusion matches the data: Llama fine-tunes imitate ChatGPT.

**RQ2 takeaways:**
- Input only ≈ chance for both models. Train loss fell 1.40 → 1.20 while val F1 stayed ~0.26: memorizing prompts without generalizing.
- Why: Arena assigns models to prompts at random, so prompts are independent of the family. In datasets where users choose models per task, input-only could beat chance through selection bias, not a fingerprint.
- Adding the input doesn't help (CNN 0.688 → 0.675; LSTM flat).
- Confound: "both" mode shows only 192 output tokens vs 256. **Control DONE:** CNN output-only at 192 tokens = 0.681 ± 0.010. About half of the 0.013 drop comes from the shorter output; the remaining 0.006 attributed to the prompt is within seed noise. The prompt neither helps nor meaningfully hurts.

---

## 4. Analyses

### RQ4: DONE, drafted in the doc

**Handcrafted features (src/features.py, all 9,192 outputs):**
| Feature (mean) | GPT | Claude | Llama | PaLM |
|---|---|---|---|---|
| Words (mean / median) | 134.9 / 92 | 170.4 / 186 | 128.8 / 102 | 175.5 / 169 |
| Chars (mean) | 831 | 1,043 | 762 | 1,009 |
| TTR, first 100 words | 0.718 | 0.684 | 0.669 | 0.642 |
| Sentences (mean) | 12.3 | 16.6 | 10.0 | 14.7 |
| Words per sentence | 14.2 | 11.2 | 16.3 | 13.4 |
| Newlines per 100 words | 6.34 | 9.68 | 5.60 | 7.43 |
| List share of lines | 0.182 | 0.364 | 0.160 | 0.207 |
| Bold per 100 words | 0.017 | 0.017 | 0.010 | 0.373 |
| Share of outputs with any bold | ~0.5–1.2% | ~0.5–1.2% | ~0.5–1.2% | 13.3% |
| '!' per 100 words | 0.271 | 0.365 | 0.252 | 0.200 |
| '?' per 100 words | 0.286 | 0.224 | 0.282 | 0.206 |

**Kruskal-Wallis H (how much a feature differs across families):** bold 710 · newline 692 · list 618 · sentence count 585 · avg sentence length 525 · TTR 418 · words 320 · chars 293 · code blocks 32 · '?' 12 · headers 3.2

**Box plot:** results/rq4/top4_features_boxplot.png (bold shown as share of outputs with any bold, since most outputs have none).

**Family habits:**
- Claude: longest, most list-heavy, shortest sentences.
- PaLM: the only family that regularly uses bold.
- GPT: highest lexical diversity, shortest typical response.
- Llama: longest sentences.
- GPT and Llama are the closest pair on almost every feature (matches the RQ1 confusion).

**Top openings (share of responses):**
- Claude: "Here is a" 13.5%, "Here are some" 9.2%, "I apologize, but" 5.7%
- PaLM: "I'm not able" 5.0%, "Sure, here is" 4.7%, "I am trained" 1.3%
- GPT: "As an AI" 3.6%, "I'm sorry, but" 2.4%, "As a helpful" 1.1%
- Llama: "I'm sorry, but" 2.6%, "As an AI" 1.9%
- GPT and Llama share the same top openings.

**Decision on disclaimer phrases:** "As an AI", "I am trained", and "I'm not able" don't name a model, so they're habitual style, not identity leakage. Kept in the data, and discussed in the report as part of the fingerprint. Rebuilding would have cost ~2 hours of retraining.

**Feature-only logistic regression (src/feature_lr.py):**
- Test macro-F1 0.435, accuracy 0.443 (C tuned on val; flat from 0.1 to 100).
- Per-class F1: GPT 0.304 · Claude 0.508 · Llama 0.437 · PaLM 0.492 (GPT hardest again; 95/361 correct).
- Top coefficients: GPT char_count, avg_sentence_len, sentence_count · Claude char_count, list_rate, ttr_100 · Llama avg_sentence_len, char_count · PaLM word_count, bold_rate, code_block_rate.
- Caveat: word_count and char_count are nearly collinear, so their individual signs aren't meaningful. Bold for PaLM and lists for Claude match the descriptive stats and are the trustworthy ones.

**Ablations (output only, macro-F1, 3 seeds, patience 2):**
| Setting | CNN | LSTM |
|---|---|---|
| Full response, 256 tokens | 0.688 ± 0.005 | 0.646 ± 0.060 |
| First 192 tokens | 0.681 ± 0.010 | — |
| Markdown stripped | 0.652 ± 0.018 | 0.650 ± 0.016 |
| First 100 tokens | 0.640 ± 0.032 | 0.629 ± 0.007 |

**RQ4 takeaways:**
- Signals are both linguistic and structural, but mostly wording, especially openings and refusal phrases.
- Structure features alone = 0.435, yet stripping markdown costs the CNN only 0.036, so the same information is largely also present in the wording (redundant signals).
- Truncating to 100 tokens costs 0.048 and removes most of the length signal, yet the first 100 tokens alone still give 0.640, so the fingerprint is front-loaded.
- Openings being 3-word phrases supports the RQ1 explanation that CNN filters (width 3–5) suit this task.
- LSTM ablation comparisons are noisier because its patience-2 baseline includes the early-stopped seed.

### RQ3: NEXT

**Design:** train on general + math, test on ALL code examples from all three splits (~290–320 per family, downsampled to equal counts); compare with an in-domain baseline (train and test on general + math). CNN at default patience; LSTM with patience 4 (patience 2 is shown to be unstable). Per-family F1.
- train.py fix needed: when test domains don't overlap train domains, build the test set from all splits.
- Vocab was built on all of train, including code prompts (no labels): minor caveat.
- [TODO: results]

## 5. Lessons Learned (raw notes; rewrite in my own voice)

**Setup**
- Literal `<you>` placeholder in a git URL: bash read `<` as a redirect.
- `python` vs `python3` on Ubuntu; venv needed `python3-venv`. Failed `cd`s twice created a venv in the home folder.
- GitHub no longer accepts passwords for git; fixed with `gh auth login`. GitHub usernames can't contain underscores (Sid_Bhagat → Sid-Bhagat).
- Git needed user.name/email set before the first commit.

**Secrets**
- Pasted an HF token into a chat and had to revoke it. Switched to `read -s` so the token never appears on screen or in shell history.

**Data**
- Inspecting raw data before building paid off: 20 models, and my initial family map missed several LLaMA fine-tunes.
- Leakage is the main validity threat: self-identification and the same prompt answered by two models. Fixed with self-ID dropping and question_id-grouped splits. Self-ID turned out to be rare (92 responses).
- Split first, then balance within splits.
- Heuristic labels need spot checks: the first math rules caught dates and ranges.

**Training**
- Early-stopping patience 2 was too aggressive for the LSTM: one noisy epoch ended a run at epoch 4 that likely would have kept improving.
- The CNN trains far faster and more stably than the LSTM on CPU.
- Input-only runs showed memorization: training loss fell, validation stayed at chance.
- A comparison can hide a confound: "both" mode silently cut the output from 256 to 192 tokens.

**Tooling / workflow**
- Python buffers stdout when redirected to a file, so the log looked frozen during long runs. Fixed with `PYTHONUNBUFFERED=1`; checked progress with `ps` (≈1600% CPU = training).
- Resumable grid scripts (skip runs whose JSON exists) made pausing and restarting safe.
- Claude Code accidentally started the grid by piping it to `head -1`, then caught and deleted the extra result. Lesson: review what the agent runs, especially in auto mode.
- Working in phases with `/clear` between them and a CLAUDE.md spec kept Claude Code on track.
- **Patience 2 stopped the LSTM's best seed at epoch 4 with 0.579; with patience 4 the same seed reached 0.700.** The clearest example of a hyperparameter choice changing a conclusion (it made the LSTM look worse than the CNN).
- Reporting the bad seed instead of dropping it, then testing the explanation with a controlled rerun, was more convincing than either hiding it or just asserting why it happened.
- A confound control (192-token output-only) turned a vague result into a clear one: half of the "input hurts" effect was really "less output."
- Box plots can hide a feature that's zero for most examples (bold collapsed to a flat line); switched to "share of outputs with any bold."
- Correlated features (word vs char count) make individual logistic regression coefficients unreliable.
- Deciding what counts as leakage: names are leakage; habitual phrases ("As an AI") are the fingerprint itself.
- [TODO: add RQ3 lessons]

---

## AI use disclosure

Claude (claude.ai) for planning, debugging, and drafting the report; Claude Code for writing and running pipeline code. All code reviewed and all numbers taken from the result files.
