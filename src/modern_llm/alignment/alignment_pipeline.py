"""Orchestrates Base → SFT → DPO → Verifier pipeline (RLHF-inspired).

References:
    - SFT: Ouyang et al., 2022 (InstructGPT).
    - DPO: Rafailov et al., 2023.
    - Verifier reranking: Lightman et al., 2023.
"""

from __future__ import annotations


def run_alignment_pipeline() -> None:
    """Execute Base → SFT → DPO → Verifier with shared evaluation tables.

    Pre:
        - Base, SFT, DPO checkpoints exist on disk.
    Post:
        - Logs metric deltas after each stage and verifier impact.
    """

    raise NotImplementedError("Alignment pipeline orchestration will be implemented in Phase 3.")

