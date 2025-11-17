"""CLI entrypoint for from-scratch LM training."""

from modern_llm.training import train_lm


def main() -> None:
    train_lm.main()


if __name__ == "__main__":
    main()

