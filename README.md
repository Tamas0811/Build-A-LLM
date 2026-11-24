# From internet text to an AI assistant

Practical exercises for the course case based on Andrej Karpathy's video
*Deep Dive into LLMs like ChatGPT*. Each step below lists the files it produced and
what I observed. All model outputs in this repository come from my own runs.

## Repository contents

| Step | Files |
|---|---|
| Assignment | `requirements.html` (the assignment page) |
| 1. Training data | `fineweb_sample.json`, `fineweb_corpus.txt` |
| 2. Text representation | `fineweb_bits.txt`, `fineweb_bytes.txt`, `fineweb_tokens.txt` |
| 3. Train a small model | `requirements.txt`, `model.py`, `train.py`, `outputs/` (`training_log.txt`, `training_log.csv`, `loss_plot.png`, `samples.txt`, `model.pt`), `sky_completions.py`, `sky_completions.txt`, `corpus_completions.py`, `corpus_completions.txt` |
| 4. Base model and assistant | `step4_outputs.md` |
| 5. Supervised fine-tuning data | `conversations.json` |
| 6. Hallucinations, thinking and tools | `step6_outputs.md` |
| 7. Run a model locally | `Local_model.png` |

## 1. Training data

Downloaded the first 10 documents of FineWeb (`sample-10BT`) and joined their text
fields into one continuous block (`fineweb_corpus.txt`, 22,044 characters). The sample
is typical raw web text: forum posts, a blog, news articles, a college sports report,
a food inspection report and a sign-up page.

## 2. Text representation

- `fineweb_bits.txt` / `fineweb_bytes.txt`: the first "sentence" of the corpus, the forum
  header `|Viewing Single Post From: Spoilers for the Week of February 11th|`, as binary
  (one 8-bit group per byte) and as decimal byte values (CyberChef).
- `fineweb_tokens.txt`: the whole corpus as GPT-4o token IDs (Tiktokenizer, `o200k_base`),
  **4,731 tokens**.

## 3. Train a small model

`train.py` checks the token IDs against the corpus, maps the distinct token IDs to a
compact vocabulary, and trains a tiny randomly initialised causal transformer on CPU for
60 seconds. `model.py` contains the model.

To reproduce:

```
python -m pip install -r requirements.txt
python train.py
python sky_completions.py
python corpus_completions.py
```

Results from my run (`outputs/training_log.txt`):

- **Validation:** the 4,731 token IDs in `fineweb_tokens.txt` match the corpus exactly.
- **Model:** 2 layers, 4 heads, 64 dimensions, context of 64 tokens, 336,384 parameters.
  Compact vocabulary: 1,814 distinct tokens out of 200,019 in `o200k_base`.
- **Training:** 1,986 updates in 60 s (about 34,000 tokens/s). The loss fell from 7.67
  (random guessing would be ln(1,814) = 7.50) to **0.064**.
- **Samples** (`outputs/samples.txt`, same beginning "If you have set yourself" and seed
  every time): random tokens before training; at the end, fluent-looking text stitched
  together from phrases of different documents.
- **"The sky is …"** (`sky_completions.txt`): failed. The token `' sky'` never occurs in
  the corpus, so the model has no embedding for it and cannot read the prompt.
- **Corpus sentence** (`corpus_completions.txt`): beginning "The exchange's retail".
  **81 of 100** completions reproduce the original sentence word for word; the others
  start correctly and then jump into other documents.

**Interpretation:** a near-zero loss on 4,731 tokens means **memorisation, not
understanding**. The model can only process text it has seen, and mostly recites it.
Real LLMs avoid this by training on trillions of tokens.

## 4. Base model and assistant

Tool: SLM Testing Space (CPU). Full outputs in `step4_outputs.md`.

**Limitation:** the Space did not offer the base model `HuggingFaceTB/SmolLM2-135M`, only
the Instruct versions. The prompts were therefore run on `SmolLM2-135M-Instruct`, and the
base-model behaviour was approximated by using the Instruct model in Completion mode
(raw text, no chat template).

- **"What is 2+2?"** (Chat mode, 3 runs): every run answers in an assistant style
  (steps, explanations), but none gives a clear, correct "4". Run 1 describes a nonsense
  procedure and ends with 4 + 4 = 8 as "the sum of 3 and 4"; run 3 says "two and two is equal to four … which equals
  five in total". Post-training gives the model the *format* of a helpful answer; a 135M
  model cannot reliably supply the content.
- **"Here is my top 10 list of landmarks to see in Paris:"** (Chat mode, 3 runs): the
  model treats the line as a request and writes a list, but the lists repeat items (the Arc
  de Triomphe three times in one list), invent details ("Palais Garnier (Palace of Versailles!)")
  and contradict themselves (the Eiffel Tower is "267 meters high" and "335 meters" in the
  same entry).
- **"Why is the sky blue?"** in Completion mode: **no new text was generated**. The same
  question at the end of a short `Human:` / `Assistant:` conversation: the model answers as
  the Assistant (scattering of light, shorter wavelengths of blue light). **The model did not
  change, only the document it was continuing.**
- **Knowledge cutoff:**
  - 2020 Nobel Prize in Economic Sciences: the model says "the economics department of
    Harvard University". Wrong: it went to Paul Milgrom and Robert Wilson for auction
    theory. A hallucination (the prize is never awarded to a department).
  - 2026 prize: the model invents a "fiscal deficit / Budgetary Gap" story. Wrong: it went
    to Robert E. Hall and Michael Woodford "for microeconomic foundations of macroeconomic
    stabilisation" (announced 12 October 2026, nobelprize.org). The event is after the
    model's training data, so any answer is invented.

## 5. Supervised fine-tuning

`conversations.json`: three English question/answer pairs from OpenAssistant `oasst1`
(opening questions with the reviewer-ranked best answer, `rank = 0`), selected with the
Hugging Face SQL console: monopsony in labour markets, contrastive learning, and
astrophotography.

Observation: the "ideal" monopsony answer cites Bivens & Mishel (2013), a paper on
executive pay and rents rather than labour-market monopsony. Human-written fine-tuning
data can contain unsupported citations, and a model learns the pattern "a good answer ends
with a reference".

## 6. Hallucinations, thinking and tools

Assistant: ChatGPT, free plan, web interface, web search not switched on.
Full outputs in `step6_outputs.md`.

### 6.1 Five questions about an obscure article

Article: the Augustana College cross-country report from 2010 (one of the FineWeb
documents). Correct answers: 4th place, 25:10, Wisconsin-Stevens Point, 55 points, art
history.

**Without the article**, ChatGPT first declined to answer. When asked for its best guess:

| Question | Best guess | Correct? |
|---|---|---|
| Finishing place | about 10th–20th | ❌ (4th) |
| Time | about 25:30–26:30 | ❌ (25:10) |
| Team winner | Augustana College | ❌ (Wisconsin-Stevens Point) |
| Augustana's points | about 30–60 | Range includes 55, but not an answer |
| Major | "I can't make a reliable estimate" | Declined |

**With the article pasted**, all five answers were correct and matched the article exactly.

Additional observations: in one chat ChatGPT searched the web on its
own and answered all five correctly with sources; when given links to the college's pages,
it named the wrong team winner (North Central College, which beat Augustana at a different
meet).

**Takeaway:** the model does not know obscure facts and guesses plausibly but wrongly when
pushed; putting the evidence in the context gives correct answers, and sources still need
checking.

### 6.2 "Who created you?"

Answer: "I was created by OpenAI … I'm ChatGPT, powered by GPT-6." The creator is
correct. The model name cannot be verified from the answer: OpenAI rolled out GPT-6 Luna
to free users from 8 October 2026, but reports differ on whether that applies in the web
interface or only in the desktop app. A model knows its identity only from its system
prompt and post-training, not by inspecting itself.

### 6.3 Retailer profit

Correct answer, checked with executed code: **€930.46**
(1,035 sold, 72 returned, 963 kept, 212 unsold; revenue €54,998.856 + €4,536.80;
costs €58,605.20).

| | "Give only the final answer" | "Show the calculation before answering" |
|---|---|---|
| Non-thinking | €930.46 ✅ | €913.66 ❌ (963 × 57.112 written as 54,982.056 instead of 54,998.856) |
| Thinking mode on | €930.46 ✅ | €1,317.96 ❌ (1,247 × 38.75 written as 47,933.75 instead of 48,321.25) |

Both wrong answers used the correct method and failed in a single multiplication, with
neat, convincing working. A generated explanation predicts text; it does not execute the
arithmetic. One run per condition is too few to rank the models.

### 6.4 Tokens vs characters

| Question | ChatGPT, no code | ChatGPT, executed Python |
|---|---|---|
| Larger: 9.11 or 9.9 | 9.9 ✅ | 9.9 ✅ |
| r's in "strawberry" | 3 ✅ | 3 ✅ |
| Every third character of "Ubiquitous" | "Uio" ❌ | "Uqts" ✅ |

The well-known examples are now answered correctly; the less familiar one fails because
the model sees tokens, not letters. The executed code (`text[::3]`) works on the actual
characters.

## 7. Run a model locally

LM Studio 0.4.28 on Windows with `Llama-3.2-3B-Instruct` (GGUF, Q4_K_S, 1.93 GB).
`Local_model.png` shows the model name and its pelican joke: "Why did the pelican go to
therapy? Because it had a fowl temper!"

## Notes

- `.venv/` and `__pycache__/` are excluded via `.gitignore`.
- Generation uses random sampling, so re-running the scripts or prompts can give
  different outputs.
