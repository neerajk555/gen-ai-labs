"""
Exercise 1 — Decoding Parameters Exploration
Module 2: Generative AI Fundamentals (Revision)

Goal: Call the Azure OpenAI API with the same summarisation prompt across
varying temperature settings and observe how output shifts from deterministic
and clinical to varied and creative.

Run from the project root: python exercises/01_decoding_parameters.py

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


def summarise_at_temperature(client, deployment, history_text, temperature):
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(history=history_text)}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def demo_single_patient(client, deployment, df):
    sample_history = df.loc[0, "history_text"]
    for temp in TEMPERATURES_TO_TEST:
        summary = summarise_at_temperature(client, deployment, sample_history, temp)
        print(f"\n{'=' * 80}\nTEMPERATURE = {temp}\n{'=' * 80}\n{summary}")


def build_full_comparison_table(client, deployment, df, output_path="decoding_comparison_output.csv"):
    comparison_rows = []
    for _, row in df.iterrows():
        for temp in TEMPERATURES_TO_TEST:
            summary = summarise_at_temperature(client, deployment, row["history_text"], temp)
            comparison_rows.append(
                {"patient_id": row["patient_id"], "temperature": temp, "summary": summary}
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

    print("\n" + "=" * 80)
    print("STAGE 1: Single-patient demo across three temperatures")
    print("=" * 80)
    demo_single_patient(client, deployment, df)

    print("\n" + "=" * 80)
    print("STAGE 2: Full comparison table across all patients (12 API calls)")
    print("=" * 80)
    build_full_comparison_table(client, deployment, df)


if __name__ == "__main__":
    main()
