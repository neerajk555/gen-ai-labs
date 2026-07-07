"""
Exercise 4 — Hallucination Detection and Grounding
Module 2: Generative AI Fundamentals (Revision)

Goal: Deliberately elicit hallucinations using fictional drug/disease names
(so any specific fabricated detail is unambiguous), then apply three grounding
techniques in sequence — authoritative context, an abstain instruction, and a
citation requirement — and compare ungrounded vs grounded answers side by side.

Run from the project root: python exercises/04_hallucination_grounding.py

Full concept explanation, line-by-line walkthrough, expected output, and homework
are in 04_Hallucination_Grounding.md
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

UNGROUNDED_SYSTEM_PROMPT = "You are a helpful clinical information assistant."

GROUNDED_SYSTEM_PROMPT_TEMPLATE = """You are a helpful clinical information assistant.
Use ONLY the following reference material to answer the user's question:

{context}
"""

FULLY_GROUNDED_TEMPLATE = """You are a helpful clinical information assistant.
Use ONLY the following reference material to answer the user's question:

{context}

Rules:
- If the reference material does not contain a specific answer to the question, you MUST say so explicitly instead of guessing or estimating.
- Do not state any numeric value, threshold, or protocol detail that is not explicitly present in the reference material above.
- After your answer, add a line starting with "Source:" that quotes the specific part of the reference material your answer is based on.
"""


def load_data():
    probes_df = pd.read_csv("data/hallucination_probe_prompts.csv")
    context_df = pd.read_csv("data/grounding_context.csv")
    return probes_df, context_df


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


def call(client, deployment, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def get_context_for_topic(context_df, topic):
    match = context_df[context_df["topic"] == topic]
    return match.iloc[0]["context_text"] if len(match) > 0 else None


def run_ungrounded(client, deployment, probes_df):
    print("=" * 80)
    print("STAGE: UNGROUNDED — no context, no abstain instruction")
    print("=" * 80)
    results = []
    for _, row in probes_df.iterrows():
        answer = call(client, deployment, UNGROUNDED_SYSTEM_PROMPT, row["question"])
        results.append({"probe_id": row["probe_id"], "question": row["question"], "answer": answer})
        print(f"\n[{row['probe_id']}] {row['question']}\n-> {answer}")
    return results


def run_context_only(client, deployment, probes_df, context_df):
    print("\n" + "=" * 80)
    print("STAGE: GROUNDED — context supplied, no abstain instruction yet")
    print("=" * 80)
    for _, row in probes_df.iterrows():
        context_text = get_context_for_topic(context_df, row["topic"])
        system_prompt = GROUNDED_SYSTEM_PROMPT_TEMPLATE.format(context=context_text)
        answer = call(client, deployment, system_prompt, row["question"])
        print(f"\n[{row['probe_id']}] {row['question']}\n-> {answer}")


def run_fully_grounded(client, deployment, probes_df, context_df):
    print("\n" + "=" * 80)
    print("STAGE: FULLY GROUNDED — context + abstain instruction + citation requirement")
    print("=" * 80)
    results = []
    for _, row in probes_df.iterrows():
        context_text = get_context_for_topic(context_df, row["topic"])
        system_prompt = FULLY_GROUNDED_TEMPLATE.format(context=context_text)
        answer = call(client, deployment, system_prompt, row["question"])
        results.append({"probe_id": row["probe_id"], "answer": answer})
        print(f"\n[{row['probe_id']}] {row['question']}\n-> {answer}")
    return results


def print_side_by_side(ungrounded_results, fully_grounded_results):
    print("\n" + "=" * 80)
    print("STAGE: SIDE-BY-SIDE — ungrounded vs fully grounded")
    print("=" * 80)
    for ungrounded, grounded in zip(ungrounded_results, fully_grounded_results):
        print(f"\n{'-' * 80}\nQUESTION: {ungrounded['question']}")
        print(f"\nUNGROUNDED:\n{ungrounded['answer']}")
        print(f"\nFULLY GROUNDED:\n{grounded['answer']}")


def main():
    probes_df, context_df = load_data()
    client, deployment = get_client_and_deployment()
    if client is None:
        return

    ungrounded_results = run_ungrounded(client, deployment, probes_df)
    run_context_only(client, deployment, probes_df, context_df)
    fully_grounded_results = run_fully_grounded(client, deployment, probes_df, context_df)
    print_side_by_side(ungrounded_results, fully_grounded_results)


if __name__ == "__main__":
    main()
