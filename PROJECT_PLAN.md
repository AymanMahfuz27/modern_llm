
# Building a Modern LLM from Scratch – Project Plan

## Introduction & Goals

Develop a **miniature large language model (LLM)** from scratch that incorporates several *cutting-edge techniques* from recent research. The goal is to simulate, on a small scale, features found in the latest GPT-style models (GPT-4 era and beyond). This will be a hands-on project (\~2 weeks, \~30 hours) focused on coding a transformer-based model with modern improvements. By the end, you will have a working toy LLM architecture on your GitHub portfolio that demonstrates knowledge of frontier LLM ideas. The project emphasizes learning core concepts (not just fine-tuning existing models) and showcasing implementation of novel components inspired by recent papers and industry practices.

**Key Objectives:**

* **Implement a Transformer Decoder** from scratch (in PyTorch or similar) as the backbone (analogous to GPT-2/GPT-3 architecture).
* **Integrate advanced LLM techniques**: including *Mixture-of-Experts layers*, *Rotary positional embeddings*, *attention sink* tokens for infinite context, and a *“universal verifier”* style feedback mechanism.
* **Work within limited compute** (CPU-only): use efficient design, small model size, and possibly optimize certain parts (e.g. lower precision or limited data) to train/evaluate in reasonable time.
* **Learn modern practices** in LLM training and architecture (e.g. improved normalization and activation functions, efficient attention methods) to get up-to-speed with “frontier lab” models.

## Frontier Techniques to Incorporate

To ensure the project is as **novel and impressive** as possible, we will incorporate several state-of-the-art techniques that recent LLM research has introduced:

### 1. Mixture-of-Experts (MoE) Layers

MoE is a neural network architecture that uses *multiple expert sub-networks* (for example, multiple feed-forward networks) and a gating mechanism to route each input token to a subset of these experts. In other words, instead of a single dense feed-forward layer, the model has many parallel feed-forward “experts,” and for each token the model activates only a few of them based on the token’s content:

* **Conditional Computation:** Only a *subset* of experts is active per input, so the model’s capacity scales (many parameters overall) while computation per token remains manageable. This sparsely gated approach lets us simulate a larger model without needing to compute every parameter for every token.
* **Gating Mechanism:** A small trainable router network scores each expert for a given token and selects the top-k experts to use. The selected experts process the token, and their outputs are combined (e.g. averaged or weighted) to produce the layer’s output. An auxiliary loss often helps keep the load balanced across experts.
* **Why it’s cutting-edge:** MoE allows scaling to very high parameter counts with relatively lower computation per token. Research has shown MoE can achieve *faster training and competitive performance* versus dense models on multi-domain tasks. Notably, Google’s Switch Transformer and GLaM used MoE to reach trillions of parameters by activating only one or two experts per token.
* **How to implement in our project:** Replace one or more feed-forward (MLP) layers in the Transformer with an MoE layer. For example, implement an MoE module with *N* small feed-forward sub-networks (experts) and a gating function (a lightweight fully-connected layer that produces an N-dimensional softmax for each token to choose experts). Use Top-1 or Top-2 gating (to keep it simple, you might start with Top-1 like Switch Transformer). During training, you’ll update all experts, but during inference each token only runs through its selected expert(s). This adds complexity to your implementation, but it’s doable on CPU for a small model (e.g. 4 experts each with reduced size). You’ll gain experience in designing dynamic computation graphs and understanding how modern LLMs scale efficiently.

&#x20;*Figure: Overview of a Transformer block incorporating an MoE layer. A gating network (“Gate/router”) selects a subset of expert feed-forward networks (Expert 1–4) based on each input, and their outputs are aggregated as the MoE layer’s output. In our project, we can replace the standard feed-forward in some Transformer layers with this MoE structure to experiment with conditional computation.*

### 2. Rotary Positional Embeddings (RoPE)

Modern LLMs often use **rotary positional embeddings** instead of classic position encodings to represent token positions. RoPE encodes positions by rotating the query/key vectors in multi-head attention by a certain angle dependent on the token index. This method was introduced in *Su et al. (2021)* and adopted in models like GPT-NeoX and LLaMA:

* **Why RoPE?** It provides *relative positional information* implicitly. By rotating vectors, the model can infer how far apart tokens are based on phase differences. RoPE doesn’t add learned parameters for positions and preserves the scale (norm) of vectors, aiding numerical stability.
* **Better generalization for long sequences:** A key benefit is that RoPE allows models to extrapolate to sequences longer than those seen in training. Since positions are encoded by continuous rotations, the model can in principle handle longer context (unlike fixed index embeddings which are limited to a maximum and often struggle beyond training length).
* **How to implement:** Instead of adding a positional vector to token embeddings, implement RoPE by applying a rotation to the Q and K vectors in each attention layer. This involves splitting the vector into even-dimensional pairs and rotating each pair by a position-dependent angle (the formula uses geometric series of frequencies; see Su et al., 2021). There are open-source examples – essentially you multiply the Q and K by a predefined rotation matrix (or use sin/cos formula) per position. Many implementations exist (including in LLaMA code) that you can reference. Integrate this in your Transformer forward pass (it’s a few lines of code in PyTorch). This will give your model a positional encoding scheme on par with current LLMs like LLaMA.
* **Verification:** After implementing, test that for a simple sequence you can encode/decode positions correctly (perhaps write a tiny script to confirm that relative distances affect dot products as expected). While you likely won’t *train to very long sequences*, having RoPE in place means your model *conceptually* supports longer context lengths than the training set – a very “current” feature (important as models are now extending context windows to 100k+ tokens).

### 3. Attention Sinks for Infinite Context

A very recent innovation is the concept of **attention sinks** to enable *streaming or infinite-context LLMs*. Normally, a Transformer has a fixed context window (e.g. 2048 tokens), and generating beyond that either becomes infeasible or requires dropping old context (which can cause the model to lose track of the conversation). Research by Xiao et al. (2023) found that autoregressive LLMs tend to put disproportionately large attention on the first few tokens in a sequence (those tokens act as a “sink” for attention scores). Removing those initial tokens (as happens when a context window slides) causes instability and gibberish output. The solution: **never drop the first token(s)** from the attention cache, and/or introduce dedicated *sink tokens*:

* **Attention Sink Tokens:** By always keeping a small number of initial tokens in the context (or adding special non-semantic tokens that serve as anchors), the model has a stable place to offload excess attention. This prevents the pathological collapse when old context is evicted. Essentially, the first few tokens (even if meaningless or just a special “\[SINK]” token) remain throughout generation and absorb attention, letting the model continue fluently.
* **StreamingLLM/Infinite Context:** Using attention sinks, researchers enabled Llama-2 and other models to generate coherently for millions of tokens without retraining. The memory is managed by a rolling window that *always retains the sink tokens* and the most recent N tokens, discarding only the oldest non-sink tokens as new ones come in. This achieves **constant memory** usage with stable perplexity even as the text grows very long.
* **How to implement:** In your toy model’s generation loop, implement a *sliding window attention* cache. For example, decide on a window size (maybe 256 tokens for experimentation) and a number of sink tokens (say 4). During generation, if the sequence exceeds the window, discard the oldest tokens *except* the first few. You might prepend a special token at the start of every sequence to serve as an anchor (or simply use the actual first token as anchor). Adjust how you feed the `past_key_values` to ensure the sink tokens’ key/value vectors are kept. **Note:** You don’t necessarily need to train the model differently for this to work (Xiao et al. showed even pretrained models benefit without fine-tuning). However, for full effect, one could train with a placeholder token designated as sink – you may not have time or data for that, but you can still demonstrate the mechanism with an out-of-the-box approach.
* **Demonstration:** After implementing, test your model in a looped generation (even on random or repeating data) to see that it can go beyond the normal context length. You can measure that it doesn’t break down into repetition or junk immediately when surpassing the window, thanks to the sink token. This feature is very *frontier*: it shows you’re aware of current research tackling unlimited context and memory efficiency in LLMs.

&#x20;*Illustration of using attention sinks in a sliding window context. The first four tokens (yellow) act as persistent **attention sinks** that are never evicted from the Transformer’s key/value cache, while older tokens (gray) drop off as new tokens (blue/red) are generated. This method allows the model to maintain fluency and context over infinite streams.*

In summary, attention sinks upgrade your model from a fixed-context GPT-2 style to a *streaming-capable* LLM, a capability at the forefront of current research.

### 4. “Universal Verifier” Feedback Loop

Recently, OpenAI insiders revealed a training technique called the **“Universal Verifier,”** essentially using one model to *verify or critique* the outputs of another. This is akin to having a built-in expert that checks the work of the main LLM. During training (often in Reinforcement Learning with Human Feedback, RLHF), this automated verifier grades the answers, enabling improvements without constant human labels:

* **Concept:** Assign an LLM the role of a **critic/judge** that can research or use tools to check another model’s answer. For example, if the main model answers a math problem, the verifier model can try to solve it independently or use a calculator and then compare answers, giving a reward if correct. In a sense, it’s like the discriminator in a GAN, but here it’s a smart evaluator using knowledge and reasoning to assess outputs.
* **Why it’s novel:** This approach was highlighted as a *“secret weapon”* in OpenAI’s progress toward GPT-4 and beyond. It generalizes the idea of using code execution or unit tests (which work for math/code tasks) into a broader, learned verification system that can handle subjective answers too. Essentially, it’s an AI safety and quality technique to ensure the model’s answers are not just fluent but *correct and high-quality across domains*.
* **In our project:** We won’t be able to fully replicate RLHF, but we can incorporate the **spirit of a verifier** in a couple of ways:

  * *Architecturally*, you could design a two-stage generation: first the model produces an answer, then it gets fed back in (along with the question) for a “verification pass.” This could be the same model used in a different mode or a smaller separate model. The verifier can output a score or classification (correct/incorrect) or even suggest an improvement. For instance, you might train a tiny classifier head on top of the model’s embeddings to judge correctness for a specific task (if you have synthetic labeled data).
  * *Tool integration*, if coding a separate model is too heavy: For certain domains like math or code, you can integrate a tool as the verifier. E.g., after the model generates a solution, write a Python function to actually compute the result or run the code and see if it errors. This result can be used to inform the user or even fed back to refine the model’s output (in an interactive setting).
* **Example demonstration:** Suppose you train your model on a small math QA dataset (e.g. addition problems) or logical puzzles. You can then implement a loop where for each answer generated, a verifier function checks it (using ground-truth or calculation) and the system either flags it or has the model attempt a “second try” if the first was wrong. This showcases an *automated validation pipeline*. It’s exactly the kind of approach OpenAI hinted at: *an LLM that checks another LLM’s answers using external knowledge or computations*. Even a toy demonstration of this (on simple tasks) will be impressive, underlining that you understand how to incorporate feedback and self-correction mechanisms into LLMs.
* **Learning aspect:** Implementing a verifier will teach you about framing one model’s output as input to another, and possibly about reinforcement or at least iterative refinement. You’ll also confront challenges like: what if the verifier is not always correct? How to train it or utilize it without human labels? These are open problems at the frontier, so even a basic implementation along these lines is a strong portfolio piece.

### 5. Additional Modern Enhancements (Normalization, etc.)

To round out the project with modern best practices, consider including some *architectural enhancements* that nearly all new LLMs use:

* **RMSNorm & Pre-Normalization:** Instead of LayerNorm after sublayers, use *RMSNorm* (root mean square norm) applied *before* the attention and MLP sublayers. LLaMA, for example, adopted RMSNorm pre-normalization to improve training stability for deep networks. This change is minor to implement (PyTorch has RMSNorm or you can code it easily) but shows you’re following current design patterns.
* **SwiGLU activation:** Use *Swish-Gated Linear Units* (SwiGLU) in the feed-forward network instead of the old GELU or ReLU. This activation (introduced by Shazeer 2020) was used in PaLM and LLaMA, and tends to slightly improve performance by a better use of network capacity. It’s basically two linear projections where one is multiplied by a sigmoid gating of the other. Implementing it is straightforward (it’s like `out = (W1·x) * sigmoid(W2·x)` instead of `W·x` followed by GELU).
* **Efficient attention computation:** If you were using a GPU, you’d want to integrate *FlashAttention* or a similar memory-efficient attention algorithm. On CPU, you might not have those libraries available or it might not help as much, so this is optional. But be aware of it: FlashAttention is an optimization that computes attention in a fused, GPU-friendly way to avoid storing huge matrices. Including mention of it (or using it if running on a smaller sequence) could be a bonus point.
* **Evaluation with tokenizer & decoding:** Don’t forget to include a BPE tokenizer (you can use a simple byte-level tokenizer like GPT-2’s or even character-level for simplicity) to encode text for your model and decode outputs. Showcase that your model can generate sample text. While the focus is architecture, demonstrating end-to-end text generation (even if the model’s output is babble due to its small size) is valuable for a portfolio.

These additional tweaks align your project with what “the big guys” are doing in 2024-2025. For instance, combining **Rotary embeddings, RMSNorm, SwiGLU** basically gives you the architectural recipe of LLaMA, one of the best open-source models. By implementing these, you’ll understand *why* such choices matter.

## Project Roadmap

Following is a proposed step-by-step roadmap for the project, combining the above components into an organized implementation plan. Each step builds on the previous, and approximate time allocations (given \~30 hours total) are noted:

1. **Plan & Setup (Day 1):** Set up your development environment (Python, PyTorch, etc.). Gather reference materials – e.g., Karpathy’s nanoGPT code and documentation for any unfamiliar components. Define the scope: decide on model size (e.g. **small Transformer** with maybe 4-6 layers, 8 heads, 256-dimension embeddings – tune this to what can train on CPU in a few hours per epoch). Also prepare or choose a **dataset** for training – something modest like a subset of WikiText, Shakespeare, or a synthetic dataset (to demonstrate capabilities like math or code). Ensure you have a tokenizer ready (you can reuse an existing one like GPT-2’s BPE to save time). *Outcome:* a clear project skeleton and needed libraries installed.

2. **Implement the Base Transformer (Days 2-4):** Code the core model without the fancy stuff first – just to get it running:

   * Create the `TransformerBlock` with self-attention, feed-forward, and residual connections. Use PyTorch modules for Linear layers, etc. Apply *pre-norm* (RMSNorm) before attention and MLP as decided.
   * Test the forward pass with random data to ensure dimensions line up. This is also where you add **Rotary Positional Embedding**: implement a function to apply RoPE to Q,K vectors each forward pass. Verify that after adding RoPE, the model still produces output of correct shape.
   * Keep the model small at first for testing (maybe 2 layers) and ensure you can run a forward and backward pass. No MoE yet, no attention sink logic – just a standard transformer decoder with causal masking. *Outcome:* A minimal GPT-like model that you can feed some data and get an output (logits).
   * **Tip:** You can borrow ideas from Karpathy’s [makemore](https://github.com/karpathy/makemore) or nanoGPT code for things like masking and batching. Given you have limited compute, also implement an efficient training loop (use PyTorch DataLoader, etc., and maybe start with very small batch size to see it train).

3. **Integrate Mixture-of-Experts Layer (Days 5-6):** Now replace the feed-forward network in **at least one** of the Transformer layers with an MoE layer:

   * Implement an `MoEFeedForward` module: it should contain *N* expert linear->activation->linear sub-networks (start with e.g. 4 experts, each could have smaller hidden size than the original to keep total params in check). Also implement a gating network that takes the input (the token’s representation) and produces a score for each expert. Use a softmax to get probabilities and optionally choose top-1 or top-2 experts for routing.
   * For simplicity, you might do *Top-1 routing*: pick the single expert with highest score for each token (this avoids blending outputs and is easier to implement – just gather the output of that one expert). This is like the Switch Transformer approach. Alternatively, try weighting two experts’ outputs if you’re comfortable with that.
   * Ensure gradient flows through the gating (you may need to multiply expert outputs by the gate probability if using more than one, and maybe stop gradient on the selection if using hard top-1 – research *Straight-Through Estimator* if needed, or simply implement a soft mix of all experts weighted by softmax probabilities as a first attempt).
   * **Testing:** After integrating, run a forward pass to ensure that for a given input, the gating selects an expert and you get an output of the same dimension. Print which experts are firing for some test inputs (just to see it working). This is also a good time to overfit a tiny batch – e.g., make sure the model can fit a very small dataset (like 10 examples) to verify training behaves normally with the MoE in place.
   * *Outcome:* Your transformer now has conditional computation. This is a big accomplishment – you’ve effectively built a simplified version of a Switch Transformer layer. You can highlight in your portfolio that you implemented gating and experts from scratch.

4. **Implement Attention Sink Mechanism (Day 7):** Add the capability for a rolling attention window with fixed sink tokens:

   * Decide how to realize a “sink token.” One easy way: define a special token in your vocabulary (e.g. `<sink>`) and ensure every input sequence starts with it repeated a few times (e.g. 4 sink tokens). During generation, you will keep these in the cache.
   * Modify your inference code (the code that generates autoregressively) such that if the number of tokens exceeds your window (e.g. >256), you will slice the `past_key_values`. Typically `past_key_values` is a list for each layer of tensors \[3] of shape (batch, heads, seq\_len, head\_dim). You need to retain the first `sink_count` entries and the last `window_size - sink_count` entries, concatenating them appropriately. You might write a helper to do this bookkeeping.
   * This step doesn’t change the model’s *training* (you can still train on fixed sequences of some length, possibly including the sink tokens at start if you want the model to be aware of them). It mainly affects *inference*. You could simulate a long text generation by feeding a sequence, truncating with the sink rule, then continuing generation.
   * *Outcome:* A test script where you generate, say, a few hundred tokens from your model (they can be random if the model isn’t well-trained yet) but confirm that the process doesn’t error out and that the sink tokens remain at the start. If your model is trained on some repetitive data (e.g. alphabet as in Xiao’s example), you could test that with sinks it continues the sequence correctly, whereas if you evict everything it might break. Document that your model supports this “endless generation” mode with attention sinks (very few personal projects will have this!).

5. **Train the Model (Days 8-11):** With the architecture ready, kick off training on your chosen dataset:

   * Given CPU constraints, you might need to keep the model tiny and dataset small. For example, you could train on a character-level task or very small text just to see some learning. Alternatively, use Google Colab or another free GPU resource for a few hours if possible (since training even a small GPT can be slow on CPU). If that’s not feasible, consider reducing the problem to something like learning a simple algorithm (e.g. addition of two numbers given as input, which the model can learn character by character).
   * Use an optimizer like AdamW. Monitor loss to ensure it’s decreasing. Train for as long as you reasonably can (maybe a few epochs over a small dataset). Since this is for portfolio, even partial training is okay – the focus is the features, not achieving SOTA perplexity.
   * Save checkpoints of the model. You may also train two versions if needed: e.g., one model for the main LLM, and perhaps a second small model to act as a verifier if you plan to train a verifier model. However, due to time, you might skip training a separate verifier model and instead implement the verifier logic as described next.
   * *Outcome:* A trained (or at least partially trained) model checkpoint that you can use for generation demos. Even if it just learns basic structure of text or a toy task, that’s fine. The training process itself is a learning experience – you’ll have encountered issues like initialization, stability (watch if MoE caused any instability – you might need to play with learning rates or gating softmax temperature to keep things stable).

6. **Verifier Feedback Loop Demo (Days 12-13):** Implement a simple verification pipeline to showcase the “universal verifier” concept:

   * If your model was trained on a specific task with objective correctness (math, logic, etc.), you can now build a routine where the model’s answer is checked. For example, if it’s math, you can write a function to calculate the true answer and compare. If it’s text, perhaps use some heuristic or regex as “ground truth” check (not easy for open-ended text, so stick to something checkable if possible).
   * **Option 1:** *Same-model self-reflection:* Prompt your trained model to act as a checker. For instance, append a prompt like: `"Q: <question>\nA: <model_answer>\nChecker: The answer is"` and see if the model can judge it or correct it. This might not work well without fine-tuning, but it’s interesting to try.
   * **Option 2:** *Hard-coded verifier:* Write a script where after model generates answer, your code verifies it (e.g., by computation or looking up an answer if you have an answer key). If the answer is wrong, maybe have the script prompt the model again saying “that was incorrect, please try again” and see if it can correct itself (this requires the model to have learned to try different reasoning on a second attempt – which it may not unless trained, but you can simulate this by maybe providing the solution path in the second prompt).
   * The simplest demonstration: log or print out something like `Q, model's A, verifier result (pass/fail or corrected answer)`. Even a dummy example like Q="2+2", model says "5", verifier says "incorrect, should be 4" shows the principle of an automated check.
   * *Outcome:* A Jupyter notebook or script in your repo showing a few Q\&A examples where the verifier mechanism catches an error or at least evaluates the answers. This part shows that you’re aware of *reinforcement learning from AI feedback* approaches and can incorporate them.

7. **Evaluation and Showcase (Day 14):** Spend time in the final days to **document and polish**:

   * Evaluate the final model on whatever small test you can. Gather some sample generations (with and without attention sink mode, perhaps) to include in your portfolio README. For example, show a snippet of a long text continuation to illustrate the infinite context, or show how the MoE gating routes different example inputs to different experts (you can instrument the code to print which expert was chosen for certain tokens – maybe show that tokens that are numeric go to a “numeric expert”, etc., if any such pattern emerged).
   * Create visuals if possible: a diagram of your architecture (you can draw a block diagram showing the transformer with MoE, etc.), or training curves if you have them. These complement the code and make your project stand out.
   * Write a clear **README** summarizing what you built, why it’s interesting, and referencing the research inspirations. Cite the techniques (e.g. “This project implements a Mixture-of-Experts layer as in Switch Transformer, and uses Rotary Embeddings for position encoding, etc.”). Showing that you know the original sources will impress readers.
   * *Outcome:* A polished project ready to push to GitHub and share. It should impress recruiters or researchers by demonstrating familiarity with state-of-the-art LLM concepts and the ability to implement them.

## Learning Outcomes & Frontier Awareness

By completing this project, you’ll achieve multiple learning outcomes:

* **LLM Architecture Mastery:** You will deeply understand how a Transformer decoder is built and gain intuition for attention mechanisms, multi-head attention, and training dynamics of language models.
* **Experience with Advanced Research Ideas:** Hands-on implementation of MoE, RoPE, attention sinks, etc., means you’ve essentially recreated (in miniature) components from Google and OpenAI’s latest models. This kind of exposure is exactly what teams at “frontier labs” look for – you can discuss not just theory but practical challenges (like gating instabilities or long-sequence issues) that you encountered and solved.
* **Problem-Solving and Engineering:** Working under compute constraints forces you to optimize and be clever (maybe using smaller data, or finding free GPU time, or vectorizing operations on CPU). This is great experience in research engineering – figuring out how to get a concept working in practice.
* **Portfolio Visibility:** The final deliverable isn’t just an idea, it’s a tangible code artifact. It shows you didn’t simply follow a tutorial, but *innovated* by combining multiple advanced techniques. When others see your repository, with references to recent papers and your own notes on what worked or not, it will signal that you are up-to-date with the fast-moving LLM field.

## Conclusion

This two-week deep learning side project is designed to **propel you to the frontier of LLM development**. You’ll catch up on the innovations that happened after GPT-2 and GPT-3 by implementing them yourself on a small scale. From Mixture-of-Experts scaling to infinite-context handling and automated verification, you’re touching on many of the hot research topics in 2024-2025. By the end, you won’t be “left behind” – you’ll have hands-on experience with a mini GPT-style model that embodies several state-of-the-art ideas. Not only will this be a rich learning experience (and no doubt *fun* to build!), but it will also result in a standout portfolio project. Future interviews or applications (even at places like OpenAI) can be leveraged with this experience – you can discuss how you implemented a transformer, the hurdles with MoE gating or long context, and how you integrated a feedback loop for answer verification. This signals a combination of **research awareness** and **practical ability**.

Embark on this project with enthusiasm – it’s like *building your own mini-GPT-4*! Good luck, and enjoy the process of learning and innovation.

**Sources:** The design above is informed by recent research and insights, including Mixture-of-Expert LLM architectures, Rotary Positional Embedding used in LLaMA, attention sink techniques for infinite text generation, and OpenAI’s “universal verifier” approach for model self-improvement. These cutting-edge developments have been distilled into a cohesive project roadmap to ensure you learn and implement the *latest and greatest* in LLMs. Good coding!



# 🧠 Modern LLM From Scratch

This project is a step-by-step journey to build a modern Transformer-based language model **from scratch**, starting from a **vanilla decoder-only Transformer** (like GPT-2) and incrementally upgrading it with **state-of-the-art innovations** found in models like GPT-4, Claude, and LLaMA.

Every component is implemented by hand to deeply understand the evolution of large language models.

---

## 🧭 Project Flow

This is the implementation path — follow this order and build each phase *from scratch* before moving on.

### 🔰 Phase 1 — Tokenizer: Word-level → Byte-level BPE

📁 [`tokenizer/`](src/modern_llm/tokenizer/)

- `word/build_vocab.py` – learn a vocabulary from a corpus
- `word/tokenizer.py` – encode/decode methods
- ✅ `tests/tokenizer/test_word_tokenizer.py`

📝 Upgrade later to byte-level BPE (like GPT-2)

---

### 🔧 Phase 2 — Vanilla Decoder-Only Transformer

📁 [`models/neural/`](src/modern_llm/models/neural/)

- `attention.py` – causal self-attention w/ masking
- `transformer.py` – token embeddings, stacked blocks, LM head
- `block.py` (optional) – clean transformer layer abstraction
- ✅ `tests/transformer/test_transformer_training_smoke.py`

Also:
📁 [`data/`](src/modern_llm/data/)
- `split.py` – split raw text to train/val
- `loader.py` – token sequence batching

---

### 🚂 Phase 3 — Training & Generation

📁 [`train/`](src/modern_llm/train/)

- `train_transformer.py` – main training CLI
- `train_ngram.py` / `train_neural.py` – alternate training modes

📁 [`eval/`](src/modern_llm/eval/)

- `sampling.py` – greedy/temperature decoding
- `perplexity.py` – evaluation metric

Also:
📁 [`utils/`](src/modern_llm/utils/)
- `checkpoint.py`, `config.py`, `seeding.py` – general training utilities

---

### 🚀 Phase 4 — Modern Upgrades

📁 [`layers/`](src/modern_llm/layers/)

Each file is a drop-in replacement for a vanilla component:

| File                | Purpose                              |
|---------------------|--------------------------------------|
| `rope.py`           | Rotary Positional Embeddings         |
| `rmsnorm.py`        | RMSNorm + Pre-Norm                   |
| `swiglu.py`         | Gated FFN (SwiGLU)                   |
| `moe.py`            | Mixture of Experts (Top-1 Routing)   |
| `attention_sink.py` | Sliding window + attention sinks     |

📁 [`tests/upgrades/`](src/modern_llm/tests/upgrades/)

Each test compares the upgrade vs baseline:
- `test_rope_vs_absolute.py`
- `test_rmsnorm_vs_layernorm.py`
- `test_swiglu_vs_gelu.py`
- `test_moe_vs_dense_ffn.py`
- `test_attention_sinks_vs_baseline.py`

---

### 🧪 Phase 5 — Verifier & Feedback Loop (Optional)

📁 [`eval/` or `infer/`] *(you can add this)*

- Verifier model or rule-based scorer to critique outputs
- Rerun generation if output is flagged as incorrect

---

## 🌱 Recommended Build Order (File by File)


1. tokenizer/word/build\_vocab.py

2. tokenizer/word/tokenizer.py

3. tests/tokenizer/test\_word\_tokenizer.py

4. models/neural/attention.py

5. models/neural/transformer.py

6. tests/transformer/test\_transformer\_training\_smoke.py

7. data/split.py

8. data/loader.py

9. train/train\_transformer.py

10. eval/sampling.py

11. eval/perplexity.py

\[ Now run baseline training + generation ]

12. layers/rope.py
13. layers/rmsnorm.py
14. layers/swiglu.py
15. layers/moe.py
16. layers/attention\_sink.py

\[ Test upgrades one by one in tests/upgrades/ ]



---

## 🧠 Learning Philosophy

- Every file you implement teaches a real technique used in GPT-4-era models.
- Each “phase” should end with a README note: what worked, what broke, what you learned.
- Use `tests/` to validate correctness and compare against baselines.
- Think like a frontier engineer: **what would Anthropic or OpenAI do differently here?**

---

## 🛠️ Run Commands (to be implemented)

- Train model:  
  ```bash
  python src/train/train_transformer.py --config configs/vanilla.yaml

* Generate samples:

  ```bash
  python src/eval/sampling.py --checkpoint checkpoints/step_10000.pt --prompt "The future of AI is"
  ```

---

## 📁 Repo Philosophy

* `tokenizer/` → Text-to-token and vocab logic
* `models/neural/` → Model architectures
* `layers/` → Modern drop-in upgrades
* `train/` → CLI and training pipelines
* `eval/` → Inference and evaluation
* `data/` → Raw data prep and batching
* `tests/` → Unit + scientific comparisons
* `utils/` → Configs, logging, reproducibility

---

## 🧠 Author Notes

This repo is built for **learning, not copying**. Everything is implemented from scratch to understand how Transformers evolved — and what makes frontier LLMs work today.


🚀 Goal: By the end, this model will support rotary embeddings, MoE routing, attention sinks, and verifier feedback — all built from scratch.


---
