"""
Exercise 3 — Model Comparison for Enterprise Decision Making
Module 2: Generative AI Fundamentals (Revision)

Goal: Send identical prompts to two Azure OpenAI deployments (whatever your
"smaller/cheaper" and "larger/more capable" tiers currently are — model names
have been changing rapidly on Azure through 2026, see README.md for current
guidance) and compare output quality, tone, latency, and token cost to build
an evidence-based model-tier recommendation.

Run from the project root: python exercises/03_model_comparison.py

IMPLEMENTATION NOTE: this script tries each call with temperature=0.3 first,
and automatically retries without temperature if the deployment rejects it
(reasoning models like the gpt-5 family don't support temperature at all).
This exercise isn't fundamentally about temperature, so dropping it for
reasoning-model deployments doesn't change what the exercise teaches.

Full concept explanation, line-by-line walkthrough, expected output, and homework
are in 03_Model_Comparison.md
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """Draft a response to a customer complaint about an unauthorised transaction on their account. The response must be empathetic, professional, and include clear next steps.

Customer complaint:
{complaint}
"""


def load_data(path="data/customer_complaints.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} customer complaints")
    return df


def get_client_and_deployments():
    required = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_MINI",
        "AZURE_OPENAI_DEPLOYMENT_GPT4O",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(
            f"\n[WARNING] Azure OpenAI not fully configured — missing: {', '.join(missing)}\n"
            "This exercise needs BOTH deployments (gpt-4o-mini and gpt-4o). "
            "See README.md. Exiting.\n"
        )
        return None, None

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    deployments = {
        "gpt-4o-mini": os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI"),
        "gpt-4o": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O"),
    }
    return client, deployments


def call_model(client, deployment_name, complaint_text):
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(complaint=complaint_text)}]
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=deployment_name, messages=messages, temperature=0.3,
        )
    except Exception as e:
        if "temperature" not in str(e).lower():
            raise
        # This deployment is a reasoning model that doesn't accept temperature — retry without it.
        response = client.chat.completions.create(model=deployment_name, messages=messages)
    latency_seconds = round(time.time() - start, 2)

    return {
        "response_text": response.choices[0].message.content,
        "latency_seconds": latency_seconds,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }


def run_comparison(client, deployments, df, output_path="model_comparison_output.csv"):
    comparison_rows = []
    for _, row in df.iterrows():
        for label, deployment_name in deployments.items():
            result = call_model(client, deployment_name, row["complaint_text"])
            comparison_rows.append({
                "complaint_id": row["complaint_id"],
                "model": label,
                "latency_seconds": result["latency_seconds"],
                "total_tokens": result["total_tokens"],
                "response_text": result["response_text"],
            })
            print(
                f"\n{'=' * 80}\n{row['complaint_id']} — {label} "
                f"(latency={result['latency_seconds']}s, tokens={result['total_tokens']})\n{'=' * 80}"
            )
            print(result["response_text"])

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved full comparison table to {output_path}")
    return comparison_df


def print_aggregate_summary(comparison_df):
    summary = comparison_df.groupby("model").agg(
        avg_latency_seconds=("latency_seconds", "mean"),
        avg_total_tokens=("total_tokens", "mean"),
    ).round(2)

    print("\n" + "=" * 50)
    print("AGGREGATE COMPARISON")
    print("=" * 50)
    print(summary)


def main():
    df = load_data()
    client, deployments = get_client_and_deployments()
    if client is None:
        return

    comparison_df = run_comparison(client, deployments, df)
    print_aggregate_summary(comparison_df)


if __name__ == "__main__":
    main()
