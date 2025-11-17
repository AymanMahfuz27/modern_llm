"""CLI entrypoint for SAMSum summarization finetuning."""

from modern_llm.hf import finetune_t5_samsum


def main() -> None:
    finetune_t5_samsum.main()


if __name__ == "__main__":
    main()

