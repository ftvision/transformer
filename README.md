# Transformer Learning Course

A hands-on, test-driven course for learning transformer architectures from the ground up.

## Who This Is For

Software engineers with Python and linear algebra background who want to deeply understand transformers - not just use them, but know how they work at every level: from attention mechanisms to custom GPU kernels.

## Course Philosophy

- **Code-first**: Implement before you theorize. Understanding comes from building.
- **Test-driven**: Every concept has runnable tests. Green tests = understanding verified.
- **Full-stack**: From pure Python attention to Triton kernels to distributed training.
- **Real ecosystem**: Learn the tools professionals use (PyTorch, HuggingFace, vLLM, etc.)

## Course Structure

| Phase | Chapters | Focus | Hardware |
|-------|----------|-------|----------|
| **Foundation** | 1-4 | Attention → Blocks → Full Transformer → Training | Laptop |
| **Attention Variants** | 5-7 | Linear Attention → Flash Linear/GLA → DeepSeek MLA | Laptop |
| **Production** | 8-9 | KV-cache, Quantization → vLLM, llama.cpp | Laptop/GPU |
| **Hardware** | 10-12 | Flash Attention → Distributed → Triton/JAX/TPU | GPU/TPU |

**Total: 12 chapters, 54 labs**

See [SYLLABUS.md](./SYLLABUS.md) for detailed chapter contents.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Basic linear algebra (matrix multiplication, dot products)
- PyTorch familiarity helpful but not required

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd topeka

# Install dependencies with uv
uv sync

# Verify setup
uv run pytest shared/tests/
```

### How to Use This Course

1. **Read the chapter docs** - Start with `chapters/ch01_attention/docs/` to understand concepts
2. **Work through labs** - Each lab has skeleton code in `src/` that you complete
3. **Run tests to verify** - `uv run pytest chapters/ch01_attention/lab01_dot_product/tests/`
4. **Check solutions if stuck** - Reference implementations in `solutions/`

### Example Workflow

```bash
# Start Chapter 1, Lab 1
cd chapters/ch01_attention/lab01_dot_product

# Read the lab instructions
cat README.md

# Edit the skeleton code
# ... implement YOUR CODE HERE sections ...

# Run tests until green
uv run pytest tests/

# Move to next lab
cd ../lab02_visualization
```

## Repository Structure

```
topeka/
├── README.md                    # This file
├── SYLLABUS.md                  # Detailed course syllabus
├── pyproject.toml               # uv workspace config
│
├── chapters/
│   ├── ch01_attention/
│   │   ├── README.md            # Chapter overview
│   │   ├── docs/                # Lecture materials
│   │   ├── lab01_dot_product/
│   │   │   ├── README.md        # Lab instructions
│   │   │   ├── src/             # YOUR CODE HERE
│   │   │   ├── tests/           # Verification tests
│   │   │   └── solutions/       # Reference implementation
│   │   └── ...
│   └── ...
│
├── shared/                      # Shared utilities
│   └── src/shared/
│       ├── utils.py
│       └── data/
│
└── scripts/                     # Helper scripts
```

## Progress Tracking

Each chapter has a milestone that marks completion:

- [ ] **Ch 1**: Multi-head attention matches PyTorch within 1e-5
- [ ] **Ch 2**: Transformer block matches GPT-2 block output
- [ ] **Ch 3**: Full transformer matches HuggingFace GPT-2 logits
- [ ] **Ch 4**: Train ~1M param model on Shakespeare
- [ ] **Ch 5**: Linear attention 10x faster at seq_len=4096
- [ ] **Ch 6**: GLA matches `fla` library reference
- [ ] **Ch 7**: MLA reduces KV cache by 4x
- [ ] **Ch 8**: KV-cache gives 10x generation speedup
- [ ] **Ch 9**: Serve 7B model with vLLM at >100 tok/sec
- [ ] **Ch 10**: Train with 4x longer sequences via Flash Attention
- [ ] **Ch 11**: Train model too large for single GPU with FSDP
- [ ] **Ch 12**: Custom Triton kernel within 80% of Flash Attention

## Contributing

This is a learning repository. Contributions welcome:
- Bug fixes in labs or solutions
- Improved explanations in docs
- Additional test cases
- Typo fixes

## License

MIT
