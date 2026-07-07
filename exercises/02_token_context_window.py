"""
Exercise 2 — Token Counting and Context Window Management
Module 2: Generative AI Fundamentals (Revision)

Goal: Use tiktoken to count tokens across progressively larger prompt
structures, then simulate hitting a context window limit and observe what
gets lost to naive truncation.

Run from the project root: python exercises/02_token_context_window.py

No Azure OpenAI account needed — this exercise runs entirely locally.

Full concept explanation, line-by-line walkthrough, expected output, and homework
are in 02_Token_Context_Window.md
"""

import pandas as pd
import tiktoken

SIMULATED_CONTEXT_LIMIT = 800  # artificially small, to make truncation visible

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(encoding.encode(text))


def load_data():
    with open("data/system_prompt.txt") as f:
        system_prompt = f.read()

    few_shot_df = pd.read_csv("data/few_shot_examples.csv")

    with open("data/radiology_report_long.txt") as f:
        radiology_report = f.read()

    return system_prompt, few_shot_df, radiology_report


def demo_simple_token_count():
    sample = "The patient was started on atorvastatin."
    print(f"Text: {sample}")
    print(f"Token count: {count_tokens(sample)}")
    print(f"Word count: {len(sample.split())}")


def build_prompt_stages(system_prompt, few_shot_df, radiology_report):
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

    return stages


def simulate_context_limit(full_prompt):
    full_token_count = count_tokens(full_prompt)

    print(f"\nFull prompt: {full_token_count} tokens")
    print(f"Simulated limit: {SIMULATED_CONTEXT_LIMIT} tokens")

    if full_token_count <= SIMULATED_CONTEXT_LIMIT:
        print("\nPrompt fits within the simulated limit.")
        return

    print(f"\n⚠️  Prompt EXCEEDS the simulated limit by {full_token_count - SIMULATED_CONTEXT_LIMIT} tokens.")

    tokens = encoding.encode(full_prompt)
    truncated_tokens = tokens[:SIMULATED_CONTEXT_LIMIT]
    truncated_text = encoding.decode(truncated_tokens)

    print(f"\nTruncated prompt now ends with:\n...{truncated_text[-300:]}")

    if "IMPRESSION" in truncated_text:
        print("\n✅ The IMPRESSION section (the critical clinical summary) survived truncation.")
    else:
        print(
            "\n❌ The IMPRESSION section did NOT survive truncation — "
            "a model given only this truncated text could not correctly answer "
            "questions about the report's key findings."
        )


def main():
    print("=" * 80)
    print("STAGE 1: Token count vs word count on a simple sentence")
    print("=" * 80)
    demo_simple_token_count()

    print("\n" + "=" * 80)
    print("STAGE 2: Progressive prompt build-up — tracking token growth")
    print("=" * 80)
    system_prompt, few_shot_df, radiology_report = load_data()
    stages = build_prompt_stages(system_prompt, few_shot_df, radiology_report)

    print("\n" + "=" * 80)
    print("STAGE 3: Simulating a context window limit and naive truncation")
    print("=" * 80)
    simulate_context_limit(stages["3. System prompt + few-shot + full radiology report"])


if __name__ == "__main__":
    main()
