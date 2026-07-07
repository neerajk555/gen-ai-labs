# Module 2 — Generative AI Fundamentals (Revision)
## Hands-On Lab Package

**Days 2–3 | Beginner–Intermediate | 4 exercises, ~15–20 min each**

Unlike Module 1, every exercise in this module calls a real LLM via **Azure OpenAI** — this module is about observing and reasoning about live model behaviour, not offline NLP processing. Set up Azure OpenAI **before** the session; it's the one step that can't be rushed on the day.

```
module2_labs/
├── README.md                          ← you are here — setup + index
├── requirements.txt
├── .env.example
├── data/
│   ├── patient_histories.csv          (Exercise 1)
│   ├── system_prompt.txt              (Exercise 2)
│   ├── few_shot_examples.csv          (Exercise 2)
│   ├── radiology_report_long.txt      (Exercise 2)
│   ├── customer_complaints.csv        (Exercise 3)
│   ├── hallucination_probe_prompts.csv (Exercise 4)
│   └── grounding_context.csv          (Exercise 4)
└── exercises/
    ├── 01_Decoding_Parameters.md       + 01_decoding_parameters.py
    ├── 02_Token_Context_Window.md      + 02_token_context_window.py
    ├── 03_Model_Comparison.md          + 03_model_comparison.py
    └── 04_Hallucination_Grounding.md   + 04_hallucination_grounding.py
```

Each exercise is a **pair of files** in `exercises/`: a `.md` lab guide (concept primer, line-by-line code walkthrough, expected output, troubleshooting, homework) and the matching `.py` script that actually runs. **Read the `.md` first, then run the `.py`** — the guide gives you the "why," the script gives you the "go."

| # | Exercise | Theory Topic | Needs | Runtime |
|---|----------|--------------|-------|---------|
| 1 | Decoding Parameters Exploration | 2.4 Decoding Strategies | 1 deployment (`gpt-4o-mini`) | ~2 min + 12 API calls |
| 2 | Token Counting & Context Window Management | 2.6 Embeddings, Context Windows, Tokenisation | `tiktoken` only — **no API calls** | <1 min |
| 3 | Model Comparison for Enterprise Decisions | 2.8 Model Selection and Enterprise Considerations | 2 deployments (`gpt-4o-mini` **and** `gpt-4o`) | ~2 min + 8 API calls |
| 4 | Hallucination Detection and Grounding | 2.7 Hallucination, Grounding, Bias | 1 deployment (`gpt-4o-mini`) | ~2 min + 12 API calls |

---

## Quick Start (TL;DR)

```bash
python -m venv module2-env && source module2-env/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your Azure OpenAI credentials — see Step 3 below
python exercises/02_token_context_window.py   # safe first run — needs no Azure setup at all
```

---

## One-Time Environment Setup

### 1. Create and activate a virtual environment
```bash
python -m venv module2-env
source module2-env/bin/activate        # Windows: module2-env\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Azure OpenAI

If your organisation admin already provisioned this for the training subscription (likely, if you did Module 1's Exercise 4), you may already have a working `.env` — **but Exercise 3 needs a second deployment** (`gpt-4o`) that Module 1 didn't require, so read step 3.3 either way.

**3.1 — Confirm access is approved.** Azure OpenAI requires a one-time subscription eligibility approval. Check with your Azure admin if unsure.

**3.2 — Create the resource** (skip if you already have one from Module 1):
- [Azure Portal](https://portal.azure.com) → search **"Azure OpenAI"** → **Create**
- Subscription / Resource Group: your training subscription, e.g. `rg-genai-training`
- Region: one where both `gpt-4o-mini` **and** `gpt-4o` are available — `East US` and `Sweden Central` are safe choices as of this writing
- Name: e.g. `genai-training-openai-<yourinitials>`
- Pricing tier: Standard S0
- **Review + Create** → **Create**

**3.3 — Deploy TWO models** (this module needs both, for the Exercise 3 model comparison):
1. Open the resource → **Go to Azure AI Foundry portal** → **Create new deployment**
2. First deployment — **Model:** `gpt-4o-mini`, **Deployment name:** `gpt-4o-mini-deploy`
3. Repeat — **Model:** `gpt-4o`, **Deployment name:** `gpt-4o-deploy`
4. Wait for both to show status "Succeeded"

**3.4 — Get credentials:** resource → **Keys and Endpoint** → copy **KEY 1** and the **Endpoint**.

**3.5 — Configure `.env`:**
```bash
cp .env.example .env
```
Fill in:
```
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT_MINI=gpt-4o-mini-deploy
AZURE_OPENAI_DEPLOYMENT_GPT4O=gpt-4o-deploy
```

Exercises 1, 3, and 4 read these variables automatically — no code changes needed. Exercise 2 needs no Azure setup at all (it only counts tokens locally).

---

## Running the Exercises

Run from the **project root** (`module2_labs/`), not from inside `exercises/`:

```bash
python exercises/01_decoding_parameters.py
python exercises/02_token_context_window.py
python exercises/03_model_comparison.py
python exercises/04_hallucination_grounding.py
```

If Azure OpenAI isn't configured yet, Exercises 1, 3, and 4 will print a clear warning and exit that specific script cleanly rather than crashing — you can still read through the code and run Exercise 2 while setup is in progress.

---

## Editing the Datasets

Every script reads its data from `data/` — nothing is hardcoded in the Python files. Add a new patient history, a new complaint, or a new hallucination probe question by editing the relevant CSV/TXT file directly, then re-run the script. No code changes needed — the same "pull a report" pattern used in Module 1.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `[WARNING] Azure OpenAI not configured — missing: ...` | Expected until `.env` is filled in. Complete Step 3 above. Exercise 3 specifically needs **both** deployment variables set, not just one. |
| `AuthenticationError` / `401` | Check `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are copied exactly from the Azure Portal's **Keys and Endpoint** page — no extra spaces, quotes, or trailing slashes that don't match the original. |
| `NotFoundError` / `404` referencing the model name | You're likely passing the **model name** instead of the **deployment name**. `AZURE_OPENAI_DEPLOYMENT_MINI` / `_GPT4O` must match the deployment name you chose in Step 3.3 (e.g. `gpt-4o-mini-deploy`), not `gpt-4o-mini` itself. |
| `RateLimitError` / `429` | You've hit your deployment's tokens-per-minute or requests-per-minute quota — common if running Exercise 1, 3, or 4 back-to-back with a whole class on a shared resource. Wait a minute and retry, or check your quota under the resource's **Quotas** page in Azure AI Foundry. |
| Exercise 2 fails to download `tiktoken`'s encoding file | `tiktoken` needs a one-time internet connection to fetch its tokeniser data on first run. If you're behind a restrictive corporate firewall/proxy, this specific download may be blocked even though the rest of your internet works — check with your network admin if this persists. |
| Region shows model `gpt-4o` or `gpt-4o-mini` as unavailable | Not every Azure region has every model. Check the model availability table in Azure AI Foundry, or switch to a region such as `East US` or `Sweden Central` when creating the resource. |
| `ModuleNotFoundError` for any package | Confirm your virtual environment is activated, then re-run `pip install -r requirements.txt`. |

---

## Suggested Pacing (fits inside the two Day 2–3 sessions)

| Time | Activity |
|------|----------|
| 0–5 min | Environment + `.env` check |
| 5–20 min | Exercise 1 — Decoding Parameters |
| 20–35 min | Exercise 2 — Token Counting & Context Window |
| 35–55 min | Exercise 3 — Model Comparison |
| 55–75 min | Exercise 4 — Hallucination Detection & Grounding |

Instructors: Exercise 2 requires no API calls and no waiting on model responses, so it's the safest one to run first if the group's Azure access is still being approved — it buys time without blocking on infrastructure.
