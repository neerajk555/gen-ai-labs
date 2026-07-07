"""
Exercise 1 — Decoding Parameters Exploration
Module 2: Generative AI Fundamentals (Revision)

Goal: Call the Azure OpenAI API with the same summarisation prompt across
varying settings and observe how output shifts from deterministic and
clinical to varied and creative.

Run from the project root: python exercises/01_decoding_parameters.py

IMPLEMENTATION NOTE (added mid-2026): Azure has been rapidly retiring
non-reasoning "Standard" tier models (gpt-4o, gpt-4o-mini, gpt-4.1,
gpt-4.1-mini) in favour of the reasoning-model family (gpt-5, gpt-5-mini,
gpt-5.1, etc). Reasoning models do NOT support the `temperature` parameter
at all — they use `reasoning_effort` (low/medium/high) instead, which
controls how much internal reasoning the model does before answering,
rather than how "creative" its sampling is.

This script auto-detects which kind of model your deployment actually is,
by making one small test call, and runs the appropriate version of the
exercise:
  - Non-reasoning model available -> original temperature sweep (0.0/0.5/1.0)
  - Reasoning model only          -> reasoning_effort sweep (low/medium/high)

Either way, the pedagogical point is the same: this one parameter trades
consistency/determinism for something else (creativity, for temperature;
depth of internal reasoning, for reasoning_effort) — and clinical use cases
generally want the "low variation" end of whichever knob your model has.

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
REASONING_EFFORTS_TO_TEST = ["low", "medium", "high"]


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


def detect_temperature_support(client, deployment):
    """Makes one small test call to find out whether this deployment accepts
    `temperature`. Reasoning models (gpt-5 family and similar) reject it with
    a 400 error mentioning 'temperature' in the message — we use that as the
    signal. Any other kind of error (auth, wrong deployment name, etc.) is
    re-raised so it surfaces clearly rather than being silently swallowed."""
    try:
        client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Reply with just the word OK."}],
            temperature=0.5,
        )
        return True
    except Exception as e:
        if "temperature" in str(e).lower():
            return False
        raise


def summarise(client, deployment, history_text, temperature=None, reasoning_effort=None):
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None:
        # Passed via extra_body so this works regardless of installed openai
        # SDK version's typed support for the reasoning_effort parameter.
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(history=history_text)}],
        **kwargs,
    )
    return response.choices[0].message.content


def demo_single_patient(client, deployment, df, use_temperature):
    sample_history = df.loc[0, "history_text"]
    settings = TEMPERATURES_TO_TEST if use_temperature else REASONING_EFFORTS_TO_TEST
    label = "TEMPERATURE" if use_temperature else "REASONING_EFFORT"

    for setting in settings:
        if use_temperature:
            summary = summarise(client, deployment, sample_history, temperature=setting)
        else:
            summary = summarise(client, deployment, sample_history, reasoning_effort=setting)
        print(f"\n{'=' * 80}\n{label} = {setting}\n{'=' * 80}\n{summary}")


def build_full_comparison_table(client, deployment, df, use_temperature, output_path="decoding_comparison_output.csv"):
    settings = TEMPERATURES_TO_TEST if use_temperature else REASONING_EFFORTS_TO_TEST
    setting_column = "temperature" if use_temperature else "reasoning_effort"

    comparison_rows = []
    for _, row in df.iterrows():
        for setting in settings:
            if use_temperature:
                summary = summarise(client, deployment, row["history_text"], temperature=setting)
            else:
                summary = summarise(client, deployment, row["history_text"], reasoning_effort=setting)
            comparison_rows.append(
                {"patient_id": row["patient_id"], setting_column: setting, "summary": summary}
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
    else:
        print(
            "-> `temperature` is not supported on this deployment — it's a reasoning model.\n"
            "   Running the REASONING_EFFORT sweep instead (low / medium / high).\n"
            "   Same underlying lesson: this parameter trades consistency for something else —\n"
            "   creativity for temperature, depth of internal reasoning for reasoning_effort."
        )

    print("\n" + "=" * 80)
    print("STAGE 1: Single-patient demo")
    print("=" * 80)
    demo_single_patient(client, deployment, df, use_temperature)

    print("\n" + "=" * 80)
    print("STAGE 2: Full comparison table across all patients (12 API calls)")
    print("=" * 80)
    build_full_comparison_table(client, deployment, df, use_temperature)


if __name__ == "__main__":
    main()
