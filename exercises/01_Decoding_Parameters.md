# Exercise 1 — Decoding Parameters Exploration

**Difficulty:** Beginner | **Time:** 15–20 min | **Theory link:** 2.4 Decoding Strategies

---

## 🎯 Goal

An LLM doesn't just produce "the" answer to a prompt — at every step, it's choosing the next word from a probability distribution, and *how* it chooses is controlled by decoding parameters you set in your API call. This exercise makes that choice visible and tangible.

By the end, you will be able to:
- Explain what `temperature` actually controls during text generation
- Predict, for a given use case, whether you want low or high temperature
- Explain why the same clinical prompt can look "safe and boring" at one setting and "creative but risky" at another — and why that risk profile matters enormously in regulated domains

**Why this matters:** Clinical use cases demand low-temperature, deterministic outputs — a referral letter shouldn't read differently every time it's generated from the same input. Marketing copy generation, by contrast, benefits from higher temperature. Getting this wrong in production leads to inconsistent or unsafe outputs — this is one of the simplest, highest-leverage settings you'll configure in every LLM application you build for the rest of this program.

---

## 🧠 Concept Primer (read before coding)

### What is temperature, mechanically?

At each step of generating text, the model computes a probability for *every possible next word* in its vocabulary. Before temperature is applied, imagine the model is fairly confident: word A has 70% probability, word B has 20%, and hundreds of other words share the remaining 10%.

**Temperature reshapes that probability distribution before a word is sampled:**

```
Temperature 0.0  →  Always pick the single highest-probability word.
                     Fully deterministic — same input always gives the same output.

Temperature 0.5  →  Mostly favours high-probability words, but occasionally
                     picks a slightly less likely one. Some variation run-to-run.

Temperature 1.0  →  Closer to the model's raw, unmodified probability distribution.
                     More willing to pick lower-probability words — more varied,
                     more "creative," but also more likely to wander off-topic
                     or introduce inconsistency.
```

Think of it like this: temperature doesn't add randomness *for its own sake* — it controls how much the model is willing to deviate from its own most-confident prediction at each word.

### Why does this matter clinically?

If you ask a model to summarise a patient's history for a referral letter at **temperature 0**, you get a stable, repeatable, conservative summary — critical when the same output needs to be reproducible and auditable. At **temperature 1.0**, you might get a more fluent, varied summary — but you also increase the chance of the model rephrasing a clinical detail in a way that subtly shifts its meaning, which is precisely the kind of risk that has no place in a clinical document.

### A note on top_p and frequency_penalty

Two other decoding parameters you'll see in production code:
- **`top_p` (nucleus sampling):** instead of a temperature-based reshaping, `top_p` restricts the model to only sample from the smallest set of words whose cumulative probability adds up to `p` (e.g., `top_p=0.9` means "only consider words from the top 90% of probability mass"). It's an alternative lever for controlling randomness, often used *instead of* temperature rather than alongside it.
- **`frequency_penalty`:** discourages the model from repeating the same words/phrases too often within one response — useful for longer creative outputs, less relevant for short factual summaries.

This exercise focuses on `temperature` since it's the most commonly tuned parameter and the clearest to reason about; the homework asks you to explore `top_p` on your own.

---

## Step 1 — The dataset

`data/patient_histories.csv` contains 4 longer, synthetic patient history narratives — the kind of raw clinical detail a referral-letter-generation prompt would need to condense.

---

## Step 2 — Azure OpenAI setup

This exercise needs one deployment: `gpt-4o-mini` (variable `AZURE_OPENAI_DEPLOYMENT_MINI` in your `.env`). Full setup instructions are in the package `README.md` — complete that once before this exercise.

---

## Step 3 — Code (in stages, with explanation after each stage)

### Stage 1 — Load data and set up the Azure OpenAI client

```python
import os
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

df = pd.read_csv("data/patient_histories.csv")
print(f"Loaded {len(df)} patient histories")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI")
```

**What this does:** standard setup you'll reuse in every Azure OpenAI exercise from here on — load credentials from `.env`, build the client, and note which deployment name to call.

---

### Stage 2 — Define the summarisation prompt and the temperatures to test

```python
PROMPT_TEMPLATE = """Summarise the following patient medical history in 3 concise sentences for a specialist referral letter.

Patient history:
{history}
"""

temperatures_to_test = [0.0, 0.5, 1.0]
```

**Why this specific prompt:** it's a realistic, single, well-defined task — summarise for a specific downstream reader (a specialist) in a specific length constraint (3 sentences). Keeping the task fixed while only changing temperature isolates temperature as the one variable under test — good experimental hygiene, and a habit worth carrying into your own prompt evaluation work in Module 3.

---

### Stage 3 — Call the API once per temperature, for one patient history

```python
def summarise_at_temperature(history_text, temperature):
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(history=history_text)}],
        temperature=temperature,
    )
    return response.choices[0].message.content

sample_history = df.loc[0, "history_text"]

results = []
for temp in temperatures_to_test:
    summary = summarise_at_temperature(sample_history, temp)
    results.append({"temperature": temp, "summary": summary})
    print(f"\n{'='*80}\nTEMPERATURE = {temp}\n{'='*80}\n{summary}")
```

**What this does:** loops through `[0.0, 0.5, 1.0]`, calling the same prompt against the same patient history each time, changing only the `temperature` parameter. Everything else in the API call is held constant — this is the controlled comparison the exercise is built around.

---

### Stage 4 — Build a structured comparison table across all patients

```python
comparison_rows = []
for _, row in df.iterrows():
    for temp in temperatures_to_test:
        summary = summarise_at_temperature(row["history_text"], temp)
        comparison_rows.append({
            "patient_id": row["patient_id"],
            "temperature": temp,
            "summary": summary,
        })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))
comparison_df.to_csv("decoding_comparison_output.csv", index=False)
```

**What this does:** extends the single-patient demo across the whole dataset, building a tidy table (`patient_id`, `temperature`, `summary`) you can scan for patterns — e.g., "does temperature 1.0 introduce clinical detail not present in temperature 0's summary?" `to_csv(...)` saves this table so you can review it later or bring it into the group discussion.

**⚠️ Note on cost/calls:** this stage makes `4 patients × 3 temperatures = 12` API calls. That's intentional and cheap on `gpt-4o-mini`, but it's worth pointing out explicitly — every loop like this has a real cost, a habit you'll need once you're designing evaluation loops in Module 6.

---

## Expected Output (illustrative — exact wording will vary; that variation *is* the point)

```
================================================================================
TEMPERATURE = 0.0
================================================================================
A 58-year-old male with an 8-year history of type 2 diabetes presented with two
days of worsening chest tightness radiating to the left arm and mild exertional
breathlessness. ECG showed non-specific ST-T changes with a rising troponin trend.
He was admitted for cardiac workup, started on dual antiplatelet therapy, a
statin, and a beta-blocker, and referred for coronary angiography.

================================================================================
TEMPERATURE = 1.0
================================================================================
This 58-year-old gentleman, a known diabetic for eight years, arrived with a
concerning two-day history of chest tightness and breathlessness that had been
steadily worsening. His ECG and rising troponin pointed toward an acute coronary
event, prompting admission, cardioprotective medication, and urgent referral for
angiography — a case that clearly warrants close specialist follow-up.
```

At temperature 0, running this twice on the same input should give you the **same or nearly identical** output. At temperature 1.0, running it twice will likely give you **noticeably different phrasing** each time — try it and confirm this yourself as part of the exercise.

---

## 🛠 Common Pitfalls

- **Assuming temperature 0 means "always identical, forever":** it's very close to deterministic, but Azure OpenAI does not formally guarantee bit-for-bit identical output at temperature 0 across all conditions. Treat it as "highly consistent," not "mathematically guaranteed identical."
- **Reading too much into one sample:** a single run at temperature 1.0 might happen to look fine. Run each temperature 2–3 times before concluding anything about consistency — this is why Stage 4's loop across the whole dataset matters more than Stage 3's single example.
- **Confusing "more detailed" with "more accurate":** a higher-temperature summary can sound more polished and confident while actually being *less* faithful to the source text. Always check generated content against the original, especially in clinical or compliance-sensitive contexts.

---

## 🏠 Homework Exercise

1. Add `top_p=0.9` to the API call (keep `temperature=1.0`) and compare the output to plain `temperature=1.0` with no `top_p` set.
2. Write 2–3 sentences: did adding `top_p` change the *style* of variation, or just reduce *how much* variation you saw?
3. Add one more patient history of your own to `data/patient_histories.csv`, written in a different clinical specialty (e.g., dermatology, psychiatry), and run it through all three temperatures.

**Hints:**
- `top_p` and `temperature` are usually not combined in production — most teams pick one lever, not both, to keep behaviour easier to reason about. This homework is specifically to help you *feel* why that convention exists.
- When judging "faithfulness," check specifically: did any temperature setting introduce a detail (a number, a diagnosis, a drug name) that isn't actually in the original patient history? That would be a small-scale hallucination — a preview of Exercise 4.
