"""CLI entrypoint for GSM8K math finetuning."""

from modern_llm.hf import finetune_math_gsm8k


def main() -> None:
    finetune_math_gsm8k.main()


if __name__ == "__main__":
    main()

