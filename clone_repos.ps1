$repos = @(
  "https://github.com/lucidrains/rotary-embedding-torch.git",
  "https://github.com/bzhangGo/rmsnorm.git",
  "https://github.com/huggingface/peft.git",
  "https://github.com/huggingface/trl.git",
  "https://github.com/deepseek-ai/DeepSeek-R1.git",
  "https://github.com/karpathy/nanochat.git",
  "https://github.com/lucidrains/mixture-of-experts.git"
)

foreach ($url in $repos) {
  git clone $url
}
