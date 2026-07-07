# Exercise 2 — Token Counting and Context Window Management

**Difficulty:** Beginner–Intermediate | **Time:** 15–20 min | **Theory link:** 2.6 Embeddings, Context Windows, and Tokenisation in LLMs

---

## 🎯 Goal

Every LLM has a hard limit on how much text it can "see" at once — the **context window** — and every character you send or receive is billed and constrained in **tokens**, not words or characters. This exercise builds the practical intuition you need before you design any RAG pipeline (Module 4) or long-running agent (Module 5), both of which live or die by careful token budgeting.

By the end, you will be able to:
- Use `tiktoken` to count tokens in any piece of text
- Explain why token count isn't the same as word count
- Track how a prompt's token budget grows as you add a system prompt, few-shot examples, and a long document
- Explain what happens when a prompt exceeds the model's context window, and why that's dangerous in a clinical setting specifically

**No Azure OpenAI needed for this exercise** — token counting happens entirely locally using the same tokeniser the model uses internally.

---

## 🧠 Concept Primer (read before coding)

### What is a token?

A token is **not** a word. Tokenisers break text into sub-word pieces based on how frequently character sequences appear in the tokeniser's training data. As a rough rule of thumb for English text: **1 token ≈ 4 characters ≈ ¾ of a word.** So 100 words of English is typically **~130–150 tokens**, not 100.

```
"The patient was started on atorvastatin."
   ↓ tokenised (approximate — exact split depends on the tokeniser)
["The", " patient", " was", " started", " on", " ator", "vastat", "in", "."]
```

Notice "atorvastatin" — an uncommon medical term — gets split into multiple sub-word pieces (`ator`, `vastat`, `in`), while common words like "The" or "was" stay whole. **This is exactly why medical, legal, and other jargon-heavy domains often burn through token budgets faster than everyday text** — the tokeniser wasn't trained primarily on that vocabulary, so it can't represent those words as efficiently.

### What is the context window?

The context window is the **maximum total number of tokens** a model can process in a single call — and critically, this total includes **everything**: your system prompt, conversation history, any documents you've pasted in, few-shot examples, AND the space reserved for the model's response. If your input tokens plus your requested output tokens exceed the limit, something has to give.

```
[ system prompt ] + [ few-shot examples ] + [ user's document ] + [ response space ]
                              all counted against ONE shared budget
```

### What actually happens when you exceed the limit?

Depending on how your application is built, one of a few things happens:
1. **The API call fails outright** with an error, if you send more input tokens than the model allows.
2. **Silent truncation** — if your application code (not the API) trims the input to "fit," you may lose exactly the part of the document that mattered, without any warning.
3. **In a RAG pipeline** (Module 4), if too many retrieved chunks are stuffed into the prompt, the least relevant ones (or the last ones added) may get cut — silently changing what the model can actually answer about.

**Why this matters clinically:** if a long radiology report gets silently truncated before reaching the model, and the truncated portion happened to contain the "Impression" section with the critical finding, the model will confidently answer questions using only what it *did* see — with no way to know it's missing the most important part. This exercise is designed to make that failure mode visible and concrete, not just theoretical.

---

## Step 1 — The datasets

- `data/system_prompt.txt` — a realistic system prompt for a clinical documentation assistant
- `data/few_shot_examples.csv` — a few worked question/answer examples
- `data/radiology_report_long.txt` — a longer synthetic radiology report, the "document" being analysed

---

## Step 2 — Set up

```bash
pip install tiktoken
```

No Azure OpenAI account is required for this exercise.

---

## Step 3 — Code (in stages, with explanation after each stage)

### Stage 1 — Count tokens in a simple string

```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(encoding.encode(text))

sample = "The patient was started on atorvastatin."
print(f"Text: {sample}")
print(f"Token count: {count_tokens(sample)}")
print(f"Word count: {len(sample.split())}")
```

**What this does:** `tiktoken.get_encoding("cl100k_base")` loads the same tokeniser family used by GPT-4-class models (including the `gpt-4o` family used elsewhere in this program). `encoding.encode(text)` converts a string into a list of integer token IDs — we only need the *count*, so we take `len(...)` of that list. Comparing token count to word count directly on the same sentence is the fastest way to build intuition for the "1 token ≈ ¾ word" rule of thumb.

---

### Stage 2 — Progressively build up a real prompt and track token growth

```python
import pandas as pd

with open("data/system_prompt.txt") as f:
    system_prompt = f.read()

few_shot_df = pd.read_csv("data/few_shot_examples.csv")

with open("data/radiology_report_long.txt") as f:
    radiology_report = f.read()

# Build the few-shot block as it would actually appear in a prompt
few_shot_block = "\n\n".join(
    f"Q: {row.question}\nA: {row.answer}" for row in few_shot_df.itertuples()
)

stages = {
    "1. System prompt only": system_prompt,
    "2. System prompt + few-shot examples": system_prompt + "\n\n" + few_shot_block,
    "3. System prompt + few-shot + full radiology report": (
        system_prompt + "\n\n" + few_shot_block + "\n\n" + radiology_report
    ),
}

for label, text in stages.items():
    print(f"{label}: {count_tokens(text)} tokens")
```

**What this does:** builds the prompt incrementally, exactly the way a real application accumulates context — start with a fixed system prompt, add reusable few-shot examples, then add the variable, potentially large document. Printing the running token count after each stage shows you **where your budget is actually being spent** — a habit directly useful when you're debugging why a RAG prompt in Module 4 is unexpectedly expensive or hitting limits.

---

### Stage 3 — Simulate hitting a context window limit

```python
SIMULATED_CONTEXT_LIMIT = 800   # artificially small, to make the failure visible in this exercise

full_prompt = stages["3. System prompt + few-shot + full radiology report"]
full_token_count = count_tokens(full_prompt)

print(f"\nFull prompt: {full_token_count} tokens")
print(f"Simulated limit: {SIMULATED_CONTEXT_LIMIT} tokens")

if full_token_count > SIMULATED_CONTEXT_LIMIT:
    print(f"\n⚠️  Prompt EXCEEDS the simulated limit by {full_token_count - SIMULATED_CONTEXT_LIMIT} tokens.")

    # Naive truncation: just cut from the end — this is what "silent truncation" looks like
    tokens = encoding.encode(full_prompt)
    truncated_tokens = tokens[:SIMULATED_CONTEXT_LIMIT]
    truncated_text = encoding.decode(truncated_tokens)

    print(f"\nTruncated prompt now ends with:\n...{truncated_text[-300:]}")

    # Check specifically whether the critical IMPRESSION section survived truncation
    if "IMPRESSION" in truncated_text:
        print("\n✅ The IMPRESSION section (the critical clinical summary) survived truncation.")
    else:
        print("\n❌ The IMPRESSION section did NOT survive truncation — "
              "a model given only this truncated text could not correctly answer "
              "questions about the report's key findings.")
else:
    print("\nPrompt fits within the simulated limit.")
```

**What this does, and why it matters:**
- We deliberately set `SIMULATED_CONTEXT_LIMIT = 800` — far smaller than any real model's actual context window — purely so the exercise reliably demonstrates truncation within a short dataset, without needing an enormous document.
- `encoding.encode(full_prompt)` turns the full prompt into its token ID list; `tokens[:SIMULATED_CONTEXT_LIMIT]` keeps only the first N tokens — this simulates the crude "just cut it off" truncation strategy that a naive application might apply.
- `encoding.decode(truncated_tokens)` converts the truncated token IDs back into readable text, so we can inspect exactly what survived.
- The final check — whether the `IMPRESSION` section (the report's most clinically important part, appearing near the end of the document) survived — is the punchline of the exercise: **naive front-to-back truncation systematically loses the most important part of a document if that part happens to come last**, which is a very common real-world document structure (executive summaries and impressions are often placed at the end, not the beginning).

---

## Expected Output (verified against the files in this package)

```
Text: The patient was started on atorvastatin.
Token count: 9
Word count: 7

1. System prompt only: ~148 tokens
2. System prompt + few-shot examples: ~258 tokens
3. System prompt + few-shot + full radiology report: ~1386 tokens

Full prompt: ~1386 tokens
Simulated limit: 800 tokens

⚠️  Prompt EXCEEDS the simulated limit by ~586 tokens.

Truncated prompt now ends with:
...(text cut off partway through the FINDINGS section)

❌ The IMPRESSION section did NOT survive truncation — a model given only this
truncated text could not correctly answer questions about the report's key findings.
```
*(Note: this sandbox's network policy blocks the one-time download `tiktoken` needs for its encoding file, so none of the exact numbers above were verified by actually running the code here — treat them as realistic illustrations, not guaranteed output. What we *did* verify directly against the files in this package: the "IMPRESSION" section starts roughly two-thirds of the way through the combined document by character count, so any truncation limit well below the full document length will reliably cut it off — that structural fact holds regardless of the exact tokeniser or token counts you get. Run the script yourself with a working internet connection to see your exact numbers.)*

---

## 🛠 Common Pitfalls

- **Assuming 1 token = 1 word:** as shown in Stage 1, this is reliably wrong, especially for jargon-heavy or non-English text. Always measure, don't estimate.
- **Forgetting that response tokens count too:** this exercise only measures *input* tokens. In a real API call, you must also reserve budget for the model's *output* — if you set `max_tokens=1000` for the response, that 1000 is subtracted from your available context budget too, on top of your input.
- **Assuming truncation always happens at a "safe" boundary:** naive truncation (as simulated in Stage 3) cuts mid-sentence, mid-word, or even mid-token — it has no awareness of document structure. This is exactly why Module 4 introduces smarter chunking strategies instead of relying on blind cutoffs.

---

## 🏠 Homework Exercise

1. Change `SIMULATED_CONTEXT_LIMIT` to a value large enough that the IMPRESSION section *does* survive truncation. What's the smallest limit at which this becomes true?
2. Instead of truncating from the **end** (`tokens[:SIMULATED_CONTEXT_LIMIT]`), try truncating from the **start** (`tokens[-SIMULATED_CONTEXT_LIMIT:]`). Does the IMPRESSION section survive now? What important information from the *beginning* of the report (hint: the clinical indication for the scan) is lost instead?
3. Write 2–3 sentences on what this tells you about why "smart" chunking (covered in Module 4) beats naive truncation for any document where important information could appear anywhere in the text.

**Hints:**
- For question 1, you can binary-search the limit manually, or write a small loop that tries increasing limits until the check passes.
- For question 2, remember Python's negative slicing (`list[-n:]`) takes the *last* n elements — so `tokens[-SIMULATED_CONTEXT_LIMIT:]` keeps the tail of the document instead of the head.
