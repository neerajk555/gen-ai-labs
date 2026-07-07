"""
Exercise 1 — Decoding Parameters Exploration
Module 2: Generative AI Fundamentals (Revision)

Goal: Call the Azure OpenAI API with the same summarisation prompt across
varying settings and observe how output shifts from deterministic and
clinical to varied and creative.

Run from the project root: python exercises/01_decoding_parameters.py

IMPLEMENTATION NOTE (added mid-2026): Azure has been rapidly retiring
non-reasoning "Standard" tier models (gpt-4o, gpt-4o-mini, gpt-4.1,
gpt-4.1-mini) in favour of the reasoning-model family (gpt-5 series,
gpt-chat-latest, etc). Reasoning models do NOT support the `temperature`
parameter at all — they use `reasoning_effort` instead, which controls how
much internal reasoning the model does before answering, rather than how
"creative" its sampling is.

This script auto-detects, by making small test calls, both:
  1. Whether your deployment supports `temperature` at all
  2. If not, exactly WHICH `reasoning_effort` levels it supports — this
     varies by model. Some support the full low/medium/high range; others
     (observed in practice, e.g. some `gpt-chat-latest` deployments) only
     support a single fixed level such as 'medium' and reject the rest with
     a 400 error. The script probes each candidate individually and only
     uses the ones that actually work, rather than assuming.

Either way, the pedagogical point is the same: this one parameter trades
consistency/determinism for something else (creativity, for temperature;
depth of internal reasoning, for reasoning_effort) — and clinical use cases
generally want the "low variation" end of whichever knob your model has.
If your deployment only supports a single reasoning_effort level, the
exercise still runs — it just demonstrates the concept with one setting
instead of a full sweep, and says so explicitly.

Full concept explanation, line-by-line walkthrough, expected output, and homework
are in 01_Decoding_Parameters.md
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """Summarise the following patient medical history in 3 concise sentences for a specialist referral letter.

Patient history:
{history}
"""

TEMPERATURES_TO_TEST = [0.0, 0.5, 1.0]
REASONING_EFFORT_CANDIDATES = ["low", "medium", "high"]


def load_data(path="data/patient_histories.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} patient histories")
    return df


def get_client_and_deployment():
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_MINI"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(
            f"\n[WARNING] Azure OpenAI not configured — missing: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill it in (see README.md). Exiting.\n"
        )
        return None, None

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI")
    return client, deployment


def _probe_call(client, deployment, temperature=None, reasoning_effort=None):
    """Fires one minimal test call and returns (True, None) on success or
    (False, error_message) on failure — never raises for expected
    unsupported-parameter errors, so callers can probe safely."""
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None:
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    try:
        client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Reply with just the word OK."}],
            **kwargs,
        )
        return True, None
    except Exception as e:
        return False, str(e)


def detect_temperature_support(client, deployment):
    """Returns True if this deployment accepts `temperature`, False if it's
    a reasoning model that rejects it. Any error NOT about temperature
    (auth failure, wrong deployment name, etc.) is re-raised immediately so
    it surfaces clearly instead of being silently swallowed."""
    ok, error_message = _probe_call(client, deployment, temperature=0.5)
    if ok:
        return True
    if error_message and "temperature" in error_message.lower():
        return False
    raise RuntimeError(f"Unexpected error while probing deployment '{deployment}':\n{error_message}")


def detect_supported_reasoning_efforts(client, deployment, candidates=REASONING_EFFORT_CANDIDATES):
    """Probes each candidate reasoning_effort level individually and returns
    only the ones this specific deployment actually accepts. Different
    reasoning models support different subsets — some support the full
    low/medium/high range, others only a single fixed level."""
    supported = []
    for effort in candidates:
        ok, error_message = _probe_call(client, deployment, reasoning_effort=effort)
        if ok:
            supported.append(effort)
        elif error_message and "reasoning_effort" not in error_message.lower():
            # Some other, unrelated failure — surface it rather than hiding it.
            raise RuntimeError(f"Unexpected error while probing deployment '{deployment}':\n{error_message}")
    return supported


def summarise(client, deployment, history_text, temperature=None, reasoning_effort=None):
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None:
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(history=history_text)}],
        **kwargs,
    )
    return response.choices[0].message.content


def demo_single_patient(client, deployment, df, use_temperature, settings):
    sample_history = df.loc[0, "history_text"]
    label = "TEMPERATURE" if use_temperature else "REASONING_EFFORT"

    for setting in settings:
        display_value = setting if setting is not None else "default (fixed by model)"
        if use_temperature:
            summary = summarise(client, deployment, sample_history, temperature=setting)
        else:
            summary = summarise(client, deployment, sample_history, reasoning_effort=setting)
        print(f"\n{'=' * 80}\n{label} = {display_value}\n{'=' * 80}\n{summary}")


def build_full_comparison_table(client, deployment, df, use_temperature, settings, output_path="decoding_comparison_output.csv"):
    setting_column = "temperature" if use_temperature else "reasoning_effort"

    comparison_rows = []
    for _, row in df.iterrows():
        for setting in settings:
            if use_temperature:
                summary = summarise(client, deployment, row["history_text"], temperature=setting)
            else:
                summary = summarise(client, deployment, row["history_text"], reasoning_effort=setting)
            comparison_rows.append(
                {"patient_id": row["patient_id"], setting_column: setting if setting is not None else "default", "summary": summary}
            )

    comparison_df = pd.DataFrame(comparison_rows)
    print("\n" + comparison_df.to_string(index=False))
    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved full comparison table to {output_path}")
    return comparison_df


def main():
    df = load_data()
    client, deployment = get_client_and_deployment()
    if client is None:
        return

    print("\nDetecting whether this deployment supports `temperature`...")
    use_temperature = detect_temperature_support(client, deployment)

    if use_temperature:
        print("-> This is a non-reasoning model. Running the TEMPERATURE sweep (0.0 / 0.5 / 1.0).")
        settings = TEMPERATURES_TO_TEST
    else:
        print("-> `temperature` is not supported — it's a reasoning model. Checking which reasoning_effort levels it accepts...")
        settings = detect_supported_reasoning_efforts(client, deployment)

        if len(settings) >= 2:
            print(f"-> Supported reasoning_effort levels: {settings}. Running the sweep across all of them.")
        elif len(settings) == 1:
            print(
                f"-> This deployment only supports a single reasoning_effort level ({settings[0]!r}) — "
                "the rest were rejected. Running the exercise with just that one setting; "
                "you won't see variation across levels, but you'll still see how a reasoning "
                "model's output compares to what temperature-based sampling would have produced."
            )
        else:
            print(
                "-> No reasoning_effort level could be set on this deployment — it only runs at its "
                "fixed default. Running the exercise with no reasoning_effort parameter at all."
            )
            settings = [None]

    print("\n" + "=" * 80)
    print("STAGE 1: Single-patient demo")
    print("=" * 80)
    demo_single_patient(client, deployment, df, use_temperature, settings)

    print("\n" + "=" * 80)
    print(f"STAGE 2: Full comparison table across all patients ({len(df) * len(settings)} API calls)")
    print("=" * 80)
    build_full_comparison_table(client, deployment, df, use_temperature, settings)


if __name__ == "__main__":
    main()
