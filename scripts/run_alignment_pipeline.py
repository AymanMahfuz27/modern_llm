"""Run the Base → SFT → DPO → Verifier pipeline."""

from modern_llm.alignment import alignment_pipeline


def main() -> None:
    alignment_pipeline.run_alignment_pipeline()


if __name__ == "__main__":
    main()

