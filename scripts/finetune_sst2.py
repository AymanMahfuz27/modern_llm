"""CLI entrypoint for SST-2 LoRA finetuning."""

from modern_llm.hf import finetune_gpt2_sst2


def main() -> None:
    finetune_gpt2_sst2.main()


if __name__ == "__main__":
    main()

