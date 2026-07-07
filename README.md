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
| 1 | Decoding Parameters Exploration | 2.4 Decoding Strategies | 1 deployment (smaller/cheaper tier — see Step 3 for current model guidance) | ~2 min + 12 API calls |
| 2 | Token Counting & Context Window Management | 2.6 Embeddings, Context Windows, Tokenisation | `tiktoken` only — **no API calls** | <1 min |
| 3 | Model Comparison for Enterprise Decisions | 2.8 Model Selection and Enterprise Considerations | 2 deployments (smaller/cheaper **and** larger/more capable tier) | ~2 min + 8 API calls |
| 4 | Hallucination Detection and Grounding | 2.7 Hallucination, Grounding, Bias | 1 deployment (smaller/cheaper tier) | ~2 min + 12 API calls |

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

If your organisation admin already provisioned this for the training subscription (likely, if you did Module 1's Exercise 4), you may already have a working `.env` — **but Exercise 3 needs a second deployment** that Module 1 didn't require, so read step 3.3 either way.

**3.1 — Confirm access is approved.** Azure OpenAI requires a one-time subscription eligibility approval. Check with your Azure admin if unsure.

**3.2 — Create the resource** (skip if you already have one from Module 1):
- [Azure Portal](https://portal.azure.com) → search **"Azure OpenAI"** → **Create**
- Subscription / Resource Group: your training subscription, e.g. `rg-genai-training`
- Region: `East US` and `Sweden Central` have generally had the broadest model selection as of this writing — but see the model-availability warning in Step 3.3 below before assuming any specific model is deployable
- Name: e.g. `genai-training-openai-<yourinitials>`
- Pricing tier: Standard S0
- **Review + Create** → **Create**

**3.3 — Deploy two models** (this module needs two deployments, for the Exercise 3 model comparison):

> ⚠️ **Model names on Azure OpenAI have been changing rapidly through 2026** — `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, and `gpt-4.1-mini` have all moved into "deprecating" status faster than their originally published timelines, and can return a `ServiceModelDeprecating` error if you try to deploy them new. **Don't follow a hardcoded model name from this guide or anywhere else without checking first** — instead:
> 1. In Azure AI Foundry → **Deployments** → **+ Create new deployment**, filter **Deployment type** to **Standard**
> 2. See what's actually selectable (not greyed out) in your region right now
> 3. Pick one **smaller/cheaper** model and one **larger/more capable** model from what's available — these become your two deployments below
>
> **A note on reasoning vs non-reasoning models:** if only the `gpt-5` family is available to you, note that most of it (`gpt-5`, `gpt-5-mini`, `gpt-5.1`, etc.) are *reasoning* models and don't support the `temperature` parameter. `gpt-5-chat` (if available) is *not* a reasoning model and behaves more like the original `gpt-4o`. **This is fine either way** — the exercises in this package (as of this version) automatically detect which kind of model you have and adapt: Exercise 1 runs a `reasoning_effort` sweep instead of a `temperature` sweep if needed, and Exercises 3–4 silently drop `temperature` and fall back to the model's default behaviour. You don't need to do anything extra — just deploy whatever's actually available.

1. First deployment — pick your smaller/cheaper option, e.g. **Model:** `gpt-5-mini` (or whatever's available), **Deployment name:** `mini-deploy`
2. Second deployment — pick your larger option, e.g. **Model:** `gpt-5` (or whatever's available), **Deployment name:** `large-deploy`

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
AZURE_OPENAI_DEPLOYMENT_MINI=mini-deploy
AZURE_OPENAI_DEPLOYMENT_GPT4O=large-deploy
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
| `ServiceModelDeprecating` when creating a deployment | The model you picked has moved into "deprecating" status and can't be used for *new* deployments (existing ones keep working until full retirement). This has been happening faster than Microsoft's published timelines throughout 2026 — go back to Azure AI Foundry's deployment screen, filter to **Standard** deployment type, and pick from whatever's actually selectable right now instead of a specific name from this guide. |
| `NotFoundError` / `404` referencing the model name | You're likely passing the **model name** instead of the **deployment name**. `AZURE_OPENAI_DEPLOYMENT_MINI` / `_GPT4O` must match the deployment name *you chose* in Step 3.3 (e.g. `mini-deploy`), not the underlying model name. |
| `RateLimitError` / `429` | You've hit your deployment's tokens-per-minute or requests-per-minute quota — common if running Exercise 1, 3, or 4 back-to-back with a whole class on a shared resource. Wait a minute and retry, or check your quota under the resource's **Quotas** page in Azure AI Foundry. |
| Exercise 2 fails to download `tiktoken`'s encoding file | `tiktoken` needs a one-time internet connection to fetch its tokeniser data on first run. If you're behind a restrictive corporate firewall/proxy, this specific download may be blocked even though the rest of your internet works — check with your network admin if this persists. |
| A model you expected is greyed out / unavailable when creating a deployment | Not every Azure region has every model, and availability changes frequently — check the model catalog in Azure AI Foundry directly rather than assuming a name from documentation is currently deployable. |
| `BadRequestError` mentioning `temperature` outside of Exercises 1/3/4's built-in fallback | Should be handled automatically — Exercises 1, 3, and 4 all detect and retry without `temperature` if your deployment is a reasoning model. If you still see this error, you may be running an older copy of these scripts; re-download the latest version. |
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
