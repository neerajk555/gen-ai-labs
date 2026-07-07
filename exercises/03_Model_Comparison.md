# Exercise 3 — Model Comparison for Enterprise Decision Making

**Difficulty:** Intermediate | **Time:** 20 min | **Theory link:** 2.8 Model Selection and Enterprise Considerations

---

## 🎯 Goal

Choosing between a smaller, cheaper model and a larger, more capable one isn't a one-time technical decision — it's an ongoing business trade-off between quality, latency, and cost at scale. This exercise gives you a repeatable, structured way to make that call with evidence instead of guesswork.

By the end, you will be able to:
- Call two different Azure OpenAI deployments with the identical prompt and compare them fairly
- Design a simple scoring rubric for output quality
- Reason about the cost/latency/quality trade-off at production scale, not just for one API call

**Real-world case — Banking:** A digital bank evaluating model tiers for customer complaint response automation needed to justify the cost of `gpt-4o` vs `gpt-4o-mini` at 50,000 interactions per day. This exercise builds the exact analytical framework needed for that kind of decision.

---

## 🧠 Concept Primer (read before coding)

### Why not just always use the biggest, best model?

At the scale of a single test prompt, cost differences look trivial — fractions of a cent. But enterprise systems operate at volume. At 50,000 interactions/day, even a small per-call cost difference compounds into a large monthly bill, and latency differences compound into real user-experience impact across thousands of concurrent conversations.

The decision isn't "which model is better" in the abstract — it's **"is the quality improvement from the larger model worth its added cost and latency, for this specific task?"** For some tasks (simple classification, short factual lookups), a smaller model performs indistinguishably from a larger one, making the larger model's extra cost pure waste. For others (nuanced tone, complex reasoning, high-stakes accuracy), the quality gap is real and worth paying for.

### What should a fair comparison actually control for?

To compare two models fairly, you must hold **everything except the model itself** constant:
- Same prompt, word for word
- Same temperature and other decoding parameters
- Same evaluation criteria, ideally scored by the same method (a rubric, or a consistent human/LLM judge — covered formally in Module 6)

If you let *any* of these vary between the two calls, you can no longer attribute the output difference to the model itself — this is the same "controlled experiment" discipline from Exercise 1, now applied to comparing models instead of comparing temperatures.

### What are the four dimensions this exercise scores?

| Dimension | What it captures |
|---|---|
| **Output quality** | Is the response accurate, well-structured, and complete? |
| **Tone appropriateness** | Does it strike the right emotional register for a sensitive banking complaint (empathetic but professional)? |
| **Latency** | How long did the call take, end to end? |
| **Token cost** | How many tokens did the call consume (a direct proxy for $ cost)? |

A model can win on quality but lose on latency and cost — the whole point of scoring all four is to make that trade-off explicit rather than picking a "winner" on gut feel.

---

## Step 1 — The dataset

`data/customer_complaints.csv` — 4 realistic banking customer complaints about unauthorised transactions, the exact scenario type used in the case study above.

---

## Step 2 — Azure OpenAI setup

This exercise needs **two deployments**: `gpt-4o-mini` and `gpt-4o` (variables `AZURE_OPENAI_DEPLOYMENT_MINI` and `AZURE_OPENAI_DEPLOYMENT_GPT4O`). See the package `README.md` for the full walkthrough — make sure both deployments show "Succeeded" before starting this exercise.

---

## Step 3 — Code (in stages, with explanation after each stage)

### Stage 1 — Load data and set up both deployments

```python
import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

df = pd.read_csv("data/customer_complaints.csv")
print(f"Loaded {len(df)} customer complaints")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

deployments = {
    "gpt-4o-mini": os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI"),
    "gpt-4o": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O"),
}
```

**What this does:** one client object can call *any* deployment on the same Azure OpenAI resource — you just pass a different `model=` (deployment name) per call. The `deployments` dictionary lets us loop cleanly over both tiers using human-readable labels ("gpt-4o-mini", "gpt-4o") mapped to their actual deployment names.

---

### Stage 2 — Define the prompt and a call function that also measures latency and tokens

```python
PROMPT_TEMPLATE = """Draft a response to a customer complaint about an unauthorised transaction on their account. The response must be empathetic, professional, and include clear next steps.

Customer complaint:
{complaint}
"""

def call_model(deployment_name, complaint_text):
    start = time.time()
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(complaint=complaint_text)}],
        temperature=0.3,
    )
    latency_seconds = round(time.time() - start, 2)

    return {
        "response_text": response.choices[0].message.content,
        "latency_seconds": latency_seconds,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
```

**What this does:**
- `time.time()` before and after the call gives us wall-clock latency — a real, user-facing metric, not just a theoretical one.
- `temperature=0.3` is deliberately low but not zero — for a customer-facing response you want consistency (Exercise 1's lesson) but a small amount of natural variation is fine here since this isn't a clinical document.
- **`response.usage`** is a field Azure OpenAI returns on every call, containing the exact token counts consumed — `prompt_tokens` (your input), `completion_tokens` (the model's output), and `total_tokens` (their sum). This is the *authoritative* token count for billing purposes — more reliable here than manually running `tiktoken` yourself, since it reflects exactly what you were charged for.

---

### Stage 3 — Run both models on every complaint and build a comparison table

```python
comparison_rows = []
for _, row in df.iterrows():
    for label, deployment_name in deployments.items():
        result = call_model(deployment_name, row["complaint_text"])
        comparison_rows.append({
            "complaint_id": row["complaint_id"],
            "model": label,
            "latency_seconds": result["latency_seconds"],
            "total_tokens": result["total_tokens"],
            "response_text": result["response_text"],
        })
        print(f"\n{'='*80}\n{row['complaint_id']} — {label} "
              f"(latency={result['latency_seconds']}s, tokens={result['total_tokens']})\n{'='*80}")
        print(result["response_text"])

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("model_comparison_output.csv", index=False)
```

**What this does:** the same nested-loop pattern as Exercise 1's Stage 4 — every complaint through every model — printing each response alongside its measured latency and token cost as it runs, then saving the full table to CSV for later scoring.

---

### Stage 4 — Aggregate the numbers to support a decision

```python
summary = comparison_df.groupby("model").agg(
    avg_latency_seconds=("latency_seconds", "mean"),
    avg_total_tokens=("total_tokens", "mean"),
).round(2)

print("\n" + "=" * 50)
print("AGGREGATE COMPARISON")
print("=" * 50)
print(summary)
```

**What this does:** `groupby("model")` splits the comparison table back into the two model groups, and `.agg(...)` computes the average latency and average token usage per model — exactly the kind of aggregate view a real cost/latency decision would be based on, not a single anecdotal example. Multiply `avg_total_tokens` by your Azure OpenAI per-token pricing and by your expected daily call volume (like the case study's 50,000/day) to translate this straight into a cost projection.

---

## Expected Output (illustrative — exact latency/tokens depend on your Azure region, load, and API version)

```
CMP001 — gpt-4o-mini (latency=1.1s, tokens=187)
Dear Mr. Verma, thank you for bringing this to our attention immediately...
[response continues, empathetic and structured]

CMP001 — gpt-4o (latency=2.3s, tokens=203)
Dear Mr. Verma, I completely understand how distressing it must be to see
an unfamiliar charge on your account...
[response continues, generally more nuanced in tone]

==================================================
AGGREGATE COMPARISON
==================================================
             avg_latency_seconds  avg_total_tokens
model
gpt-4o             2.1                  198.5
gpt-4o-mini         1.0                  183.0
```

**What to look for:** does `gpt-4o`'s response actually read as meaningfully more empathetic or well-structured, or is the difference marginal? If a human reader can't reliably tell the two apart in a blind read, that's strong evidence `gpt-4o-mini` is the better production choice for this specific task — the extra latency and cost of `gpt-4o` would be paying for a quality difference that doesn't actually reach the customer.

---

## 🛠 Common Pitfalls

- **Only looking at one example and generalising:** always compare across the full dataset (Stage 3–4), not just the first complaint — a single example can mislead you either way.
- **Ignoring latency because "it's just 1 second":** at scale, and especially in a synchronous customer-facing chat interface, latency differences compound into a real perceived responsiveness gap — don't dismiss it just because it looks small in a single test.
- **Confusing `response.usage.total_tokens` with the token counts from Exercise 2's `tiktoken`:** they should be very close but not always bit-for-bit identical — `response.usage` is the authoritative source since it reflects what actually happened server-side, including the exact system message formatting Azure adds internally.

---

## 🏠 Homework Exercise

1. Design a simple 1–5 scoring rubric covering: relevance, tone appropriateness, and completeness of next steps. Manually score all 8 responses (4 complaints × 2 models) from your `model_comparison_output.csv`.
2. Compute the average rubric score per model, and put it next to the average latency and token numbers from Stage 4.
3. Write a short recommendation (3–5 sentences) addressed to a fictional engineering lead: which model would you deploy for this use case at 50,000 interactions/day, and why? Justify using your actual numbers, not general assumptions.

**Hints:**
- This manual scoring step is a simplified preview of the LLM-as-a-Judge technique you'll formalise in Module 6 — for now, do it by eye, but notice how quickly it becomes tedious at only 8 responses. That tedium is exactly the motivation for automating evaluation later in the program.
- A good recommendation usually isn't "always use the bigger model" or "always use the cheaper one" — it's conditional, e.g., "use gpt-4o-mini as the default, but route to gpt-4o for complaints flagged as high-value or high-risk." Try to arrive at a similarly nuanced recommendation rather than a blanket one.
