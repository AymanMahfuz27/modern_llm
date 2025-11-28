# Modern LLM Training Report (v2)

**Generated:** 2025-11-28T17:45:52.859962

## Model Architecture

- **d_model:** 1024
- **n_layers:** 12
- **n_heads:** 16
- **max_seq_len:** 1024
- **Parameters:** ~253M

## Training Data

- **Pretrain:** WikiText-103 + TinyStories (~600M tokens, 30K steps)
- **SFT:** tatsu-lab/alpaca (52K examples, 5K steps)
- **DPO:** Anthropic/hh-rlhf (161K pairs, 3K steps)
- **Verifier:** GSM8K (15K examples, 3K steps)

## Evaluation Results

### Perplexity (WikiText-2 Validation)

| Model | Parameters | Perplexity |
|-------|------------|------------|
| GPT-2 (baseline) | 124M | 40.64 |
| Your pretrain | 253M | 27.03 |
| Your sft | 253M | 34.14 |
| Your dpo | 253M | 34.32 |

### Sample Generations

**Prompt:** The history of artificial intelligence began

- **Your Model:** The history of artificial intelligence began to spread. It was a powerful and powerful machine that could do amazing things. It could make people think of new ideas and ideas, and it could even help p...
- **GPT-2:** The history of artificial intelligence began in the late 1960s, when the computer was first developed. The first computer was designed by a group of researchers at the University of California, Berkel...

**Prompt:** In a groundbreaking scientific discovery,

- **Your Model:** In a groundbreaking scientific discovery, the scientists discovered that the most important part of the universe is the universe. This is because the universe is composed of thousands of stars, which ...
- **GPT-2:** In a groundbreaking scientific discovery, the researchers found that the brain's ability to process information is not limited to the brain's ability to process information.

"The brain is a very comp...

**Prompt:** The economic implications of

- **Your Model:** The economic implications of the day were high. The sun was shining brightly, and the birds were singing. The air was crisp and the air was crisp. The air was crisp and the air was crisp. The air was ...
- **GPT-2:** The economic implications of the new law are unclear.

"The law is not going to be a big deal," said David B. Smith, a professor of economics at the University of California, Berkeley. "It's going to ...

## Checkpoints

- **pretrain:** `/home1/09999/aymanmahfuz/modern_llm/checkpoints/pretrain_best.pt`
- **sft:** `/home1/09999/aymanmahfuz/modern_llm/checkpoints/sft_final.pt`
- **dpo:** `/home1/09999/aymanmahfuz/modern_llm/checkpoints/dpo_final.pt`
