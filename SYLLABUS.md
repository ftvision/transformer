# Transformer Learning Course - Syllabus

A structured, test-driven transformer learning course for software engineers.

## Design Principles

1. **Code-first, math-for-intuition**: Lead with implementations, explain math when it builds understanding
2. **Test-driven & verifiable**: Each concept has runnable tests, not notebooks
3. **Chapter + Labs structure**: Chapters cover concepts, labs are hands-on exercises
4. **Progressive complexity**: Laptop-friendly basics → GPU/TPU advanced topics
5. **Real ecosystem**: Use PyTorch, HuggingFace, vLLM etc. - learn the tools professionals use
6. **First-principles when valuable**: Build from scratch only when it deepens understanding

---

## Course Overview

| Phase | Chapter | Title | Labs | Hardware |
|-------|---------|-------|------|----------|
| **Foundation** | 1 | The Attention Mechanism | 4 | Laptop |
| | 2 | Building a Transformer Block | 4 | Laptop |
| | 3 | The Complete Transformer | 4 | Laptop |
| | 4 | Training Fundamentals | 4 | Laptop |
| **Attention Variants** | 5 | Linear Attention | 4 | Laptop |
| | 6 | Flash Linear Attention & State-Space | 5 | Laptop |
| | 7 | Sparse Attention (DeepSeek MLA, MoE) | 5 | Laptop |
| **Production** | 8 | Memory & Inference Optimization | 4 | Laptop |
| | 9 | Production Inference Frameworks | 5 | Laptop/GPU |
| **Hardware** | 10 | Flash Attention Deep Dive | 5 | GPU |
| | 11 | Distributed Training | 5 | Multi-GPU |
| | 12 | Custom Kernels & Hardware | 5 | GPU/TPU |

**Total: 12 chapters, 54 labs**

---

## Foundation Chapters

### Chapter 1: The Attention Mechanism

*Learning Objectives:*
- Understand attention as a soft lookup / weighted retrieval mechanism
- Derive and implement scaled dot-product attention
- Understand why we use multiple heads and how they specialize

*Key Concepts:*
- Query, Key, Value abstraction (database analogy)
- Dot product as similarity measure
- Softmax for normalization → attention weights
- Scaling factor √d_k and why it matters (variance control)
- Multi-head: parallel attention in different subspaces

*Docs:*
1. `attention_intuition.md` - What problem does attention solve? (seq2seq bottleneck, alignment)
2. `scaled_dot_product.md` - The math: Q, K, V, softmax, scaling
3. `multihead_attention.md` - Why multiple heads? How do they combine?

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | Dot-Product Attention | `attention(Q, K, V)` from scratch | Core formula, softmax |
| lab02 | Attention Visualization | Heatmaps of attention weights | See what attention "looks at" |
| lab03 | Multi-Head Attention | `MultiHeadAttention` class | Heads, projections, concatenation |
| lab04 | PyTorch Comparison | Match `nn.MultiheadAttention` output | Validate correctness, learn API |

*Milestone:* Your multi-head attention matches PyTorch's output within 1e-5 tolerance.

---

### Chapter 2: Building a Transformer Block

*Learning Objectives:*
- Understand the components that surround attention in a transformer
- Implement layer normalization and understand pre-norm vs post-norm
- Understand positional encodings and why transformers need them
- Build a complete transformer block

*Key Concepts:*
- Residual connections (why they help training)
- Layer normalization (vs batch norm, why per-token)
- Pre-norm vs post-norm architecture
- Positional encodings: sinusoidal, learned, RoPE
- Feed-forward network: expand → activate → contract
- Activation functions: ReLU → GELU → SwiGLU evolution

*Docs:*
1. `residuals_and_normalization.md` - Why residuals? Pre-norm vs post-norm
2. `positional_encoding.md` - Why position matters, different approaches
3. `feed_forward_network.md` - The "MLP" part, activation functions
4. `transformer_block.md` - Putting it all together

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | Layer Normalization | `LayerNorm` from scratch | Mean/var, learnable params |
| lab02 | Positional Encodings | Sinusoidal + RoPE | Frequency intuition, rotation |
| lab03 | Feed-Forward Network | `FFN` with GELU/SwiGLU | Gating, activation patterns |
| lab04 | Transformer Block | Complete block assembly | Residuals, ordering |

*Milestone:* Your transformer block forward pass matches HuggingFace GPT-2 block output.

---

### Chapter 3: The Complete Transformer

*Learning Objectives:*
- Understand how blocks stack to form complete models
- Learn the differences between encoder, decoder, encoder-decoder
- Understand tokenization and embeddings
- Load and run pretrained models

*Key Concepts:*
- Encoder vs decoder vs encoder-decoder architectures
- Causal masking for autoregressive generation
- Token embeddings and vocabulary
- Tied embeddings (input/output sharing)
- Output heads: LM head, classification head

*Docs:*
1. `encoder_decoder_architectures.md` - When to use which
2. `causal_masking.md` - Preventing future peeking
3. `embeddings_and_vocabulary.md` - Tokens, subwords, embedding tables
4. `pretrained_models.md` - Loading weights, HuggingFace ecosystem

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | Causal Masking | Implement attention mask | Triangle mask, -inf trick |
| lab02 | Token Embeddings | Embedding layer + position | Vocabulary, lookup tables |
| lab03 | Decoder-Only Transformer | Stack blocks into GPT-style model | Full architecture |
| lab04 | Load Pretrained Weights | Load GPT-2 weights into your code | Weight mapping, shapes |

*Milestone:* Your implementation generates same logits as HuggingFace GPT-2 for same input.

---

### Chapter 4: Training Fundamentals

*Learning Objectives:*
- Understand the loss functions used to train language models
- Visualize and understand gradient flow through attention
- Learn modern optimizer and LR scheduling techniques
- Train a small model from scratch

*Key Concepts:*
- Cross-entropy loss for next-token prediction
- Perplexity as evaluation metric
- Gradient flow and vanishing/exploding gradients
- AdamW optimizer (weight decay done right)
- Learning rate schedules: warmup, cosine decay
- Gradient clipping

*Docs:*
1. `loss_and_perplexity.md` - What are we optimizing?
2. `gradient_flow.md` - How gradients move through attention
3. `optimizers.md` - Adam, AdamW, why weight decay matters
4. `lr_schedules.md` - Warmup, decay strategies

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | Loss Functions | Cross-entropy, perplexity | Probability interpretation |
| lab02 | Gradient Visualization | Plot gradients through layers | See vanishing/exploding |
| lab03 | Training Loop | Complete training loop | Data loading, optimization |
| lab04 | Train Tiny Model | Train on tiny_shakespeare | End-to-end training |

*Milestone:* Train a ~1M param model that generates coherent Shakespeare-like text.

---

## Attention Variants Chapters

### Chapter 5: Linear Attention

*Learning Objectives:*
- Understand why O(n²) attention is problematic for long sequences
- Learn the kernel trick that enables O(n) attention
- Implement linear attention and understand its trade-offs

*Key Concepts:*
- Attention complexity: O(n²) memory and compute
- The associativity trick: (QK^T)V → Q(K^T V)
- Feature maps / kernel functions: φ(x)
- Causal linear attention: cumulative sum formulation
- Trade-off: efficiency vs expressiveness

*Docs:*
1. `quadratic_bottleneck.md` - Why O(n²) hurts, real-world examples
2. `kernel_trick.md` - Math behind linearization, associativity
3. `feature_maps.md` - Different φ functions and their properties
4. `causal_linear.md` - Making it work for autoregressive models

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | Complexity Analysis | Benchmark standard attention | See the O(n²) wall |
| lab02 | Kernel Trick | Implement Q(K^T V) formulation | Associativity insight |
| lab03 | Feature Maps | Try different φ: ELU+1, ReLU, exp | Impact on quality |
| lab04 | Causal Linear Attention | Cumsum-based implementation | Recurrent view |

*Milestone:* Linear attention that's 10x faster than standard for seq_len=4096.

---

### Chapter 6: Flash Linear Attention & State-Space Duality

*Learning Objectives:*
- Understand linear attention's connection to RNNs and state-space models
- Learn chunkwise parallel algorithms for efficient training
- Implement Gated Linear Attention (GLA) and understand the Kimi/Moonshot line of work

*Key Concepts:*
- Linear attention as RNN: hidden state = K^T V
- Parallel vs recurrent: training vs inference tradeoff
- Chunkwise computation: best of both worlds
- Flash Linear Attention: memory-efficient training
- Gating mechanisms: data-dependent forgetting
- DeltaNet, GLA, Mamba connections

*Docs:*
1. `linear_attention_as_rnn.md` - The recurrent interpretation
2. `chunkwise_parallel.md` - Chunking for efficient training
3. `flash_linear_attention.md` - The algorithm explained
4. `gated_linear_attention.md` - GLA, DeltaNet, Kimi variants
5. `state_space_connection.md` - How this relates to Mamba/S4

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | RNN View | Linear attention in recurrent form | State accumulation |
| lab02 | Chunkwise Parallel | Hybrid parallel/recurrent | Efficiency trick |
| lab03 | Flash Linear Attention | Memory-efficient version | Tiling for linear attn |
| lab04 | Gated Linear Attention | Implement GLA | Data-dependent decay |
| lab05 | DeltaNet | Implement DeltaNet variant | Delta rule, Kimi approach |

*Milestone:* Working GLA that matches reference implementation from `fla` library.

---

### Chapter 7: Sparse Attention

*Learning Objectives:*
- Understand sparse attention patterns and when to use them
- Implement sliding window and global token patterns
- Deep dive into DeepSeek's MLA (Multi-head Latent Attention)
- Understand how MoE integrates with attention

*Key Concepts:*
- Sparse patterns: local, strided, dilated, fixed
- Sliding window attention (Longformer, Mistral)
- Global tokens for long-range dependencies
- DeepSeek MLA: latent compression of KV
- Low-rank KV projection
- Mixture-of-Experts (MoE) basics
- Router design and load balancing

*Docs:*
1. `sparse_patterns.md` - Taxonomy of sparse attention
2. `sliding_window.md` - Local attention + global tokens
3. `deepseek_mla.md` - Multi-head Latent Attention explained
4. `kv_compression.md` - Why compress KV, how it works
5. `mixture_of_experts.md` - MoE basics, routing, load balancing

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | Sparse Patterns | Implement local/strided masks | Mask construction |
| lab02 | Sliding Window | Longformer-style attention | Local + global |
| lab03 | KV Compression | Low-rank KV projection | Latent space idea |
| lab04 | DeepSeek MLA | Full MLA implementation | Latent attention |
| lab05 | Basic MoE | Simple MoE layer | Routing, top-k |

*Milestone:* MLA implementation that reduces KV cache by 4x while maintaining quality.

---

## Production Chapters

### Chapter 8: Memory & Inference Optimization

*Learning Objectives:*
- Understand why inference is memory-bound, not compute-bound
- Implement KV-cache and understand its memory implications
- Learn batching strategies for throughput optimization
- Understand quantization basics

*Key Concepts:*
- Memory bandwidth vs compute (roofline model)
- KV-cache: caching key/value for autoregressive generation
- Memory growth: O(batch × seq × layers × heads × dim)
- Continuous batching (iteration-level scheduling)
- Quantization: int8, int4, fp8
- Quantization-aware training vs post-training quantization

*Docs:*
1. `memory_bound_inference.md` - Why inference is memory-limited
2. `kv_cache.md` - What it is, how it grows, memory analysis
3. `batching_strategies.md` - Static vs dynamic vs continuous
4. `quantization_basics.md` - Types, trade-offs, when to use

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | KV-Cache | Add KV-cache to your transformer | Incremental decoding |
| lab02 | Generation Loop | Complete text generation | Sampling, temperature |
| lab03 | Batched Generation | Handle multiple sequences | Padding, attention masks |
| lab04 | Basic Quantization | int8 linear layers | Quantize/dequantize |

*Milestone:* Generation that's 10x faster with KV-cache vs recomputing.

---

### Chapter 9: Production Inference Frameworks

*Learning Objectives:*
- Learn the production inference ecosystem
- Understand PagedAttention and its memory benefits
- Use vLLM, llama.cpp, and SGLang for real workloads
- Compare frameworks for different use cases

*Key Concepts:*
- PagedAttention: virtual memory for KV-cache
- vLLM architecture and optimizations
- llama.cpp: GGUF format, CPU inference, quantization
- SGLang: structured generation, constraint decoding
- Speculative decoding
- Choosing the right framework

*Docs:*
1. `paged_attention.md` - Virtual memory for KV-cache
2. `vllm_architecture.md` - How vLLM works
3. `llama_cpp.md` - CPU inference, GGUF, quantization schemes
4. `sglang.md` - Structured generation, RadixAttention
5. `framework_comparison.md` - When to use what

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | HuggingFace Basics | Load, generate, fine-tune | Ecosystem entry point |
| lab02 | vLLM Serving | Deploy model with vLLM | High-throughput serving |
| lab03 | llama.cpp | Quantize and run on CPU | GGUF, efficient CPU |
| lab04 | SGLang | Structured JSON generation | Constrained decoding |
| lab05 | Benchmark | Compare all frameworks | Throughput, latency, memory |

*Milestone:* Serve a 7B model with vLLM, achieve >100 tokens/sec throughput.

---

## Hardware Chapters

### Chapter 10: Flash Attention Deep Dive

*Learning Objectives:*
- Understand GPU memory hierarchy in depth
- Learn how Flash Attention achieves memory efficiency
- Implement the core ideas of tiling and recomputation
- Use Flash Attention in practice

*Key Concepts:*
- GPU memory hierarchy: registers → shared memory (SRAM) → HBM
- Memory bandwidth bottleneck
- Tiling: compute attention in blocks
- Online softmax (numerically stable incremental)
- Recomputation in backward pass
- Flash Attention 2 and 3 improvements
- Gradient checkpointing

*Docs:*
1. `gpu_memory_hierarchy.md` - SRAM vs HBM, bandwidth limits
2. `tiling_and_blocking.md` - Why and how to tile attention
3. `online_softmax.md` - Incremental, numerically stable softmax
4. `flash_attention_algorithm.md` - The full algorithm explained
5. `flash_attention_v2_v3.md` - Improvements and optimizations
6. `gradient_checkpointing.md` - Trade compute for memory

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | Memory Profiling | Profile attention memory usage | See the problem |
| lab02 | Online Softmax | Incremental softmax | Numerical stability |
| lab03 | Tiled Attention | Block-by-block attention | Core Flash idea |
| lab04 | Use Flash Attention | Integrate flash-attn library | Practical usage |
| lab05 | Gradient Checkpointing | Implement checkpointing | Memory/compute trade |

*Milestone:* Train with 4x longer sequences using Flash Attention + checkpointing.

---

### Chapter 11: Distributed Training

*Learning Objectives:*
- Understand parallelism strategies for large model training
- Implement data parallelism with DDP
- Understand model parallelism concepts
- Use FSDP and DeepSpeed for real training

*Key Concepts:*
- Data parallelism: replicate model, partition data
- Distributed Data Parallel (DDP): gradient all-reduce
- Model parallelism: tensor vs pipeline
- ZeRO stages: optimizer, gradient, parameter sharding
- FSDP (Fully Sharded Data Parallel)
- Communication primitives: all-reduce, all-gather, reduce-scatter
- Mixed precision training (fp16, bf16)

*Docs:*
1. `parallelism_strategies.md` - Data, tensor, pipeline, expert
2. `ddp.md` - How DDP works, gradient synchronization
3. `model_parallelism.md` - Tensor and pipeline parallelism
4. `zero_and_fsdp.md` - Memory-efficient data parallelism
5. `mixed_precision.md` - fp16, bf16, loss scaling

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | Multi-GPU Setup | Configure multi-GPU environment | Environment basics |
| lab02 | DDP Training | Train with DistributedDataParallel | Gradient sync |
| lab03 | FSDP | Train with Fully Sharded DP | Memory efficiency |
| lab04 | Mixed Precision | Add AMP to training | fp16/bf16 training |
| lab05 | DeepSpeed | Use DeepSpeed ZeRO | Production setup |

*Milestone:* Train a model too large for single GPU using FSDP.

---

### Chapter 12: Custom Kernels & Hardware

*Learning Objectives:*
- Write custom GPU kernels in Triton
- Understand XLA and JAX compilation model
- Learn TPU programming basics
- Master profiling and optimization workflow

*Key Concepts:*
- Triton: Python-like GPU kernel programming
- Kernel fusion: combine operations to reduce memory traffic
- XLA: graph compilation and optimization
- JAX: functional transformations (jit, vmap, pmap)
- TPU architecture: systolic arrays, HBM
- Profiling tools: nsight, torch profiler, JAX profiler

*Docs:*
1. `triton_basics.md` - Writing kernels in Triton
2. `kernel_fusion.md` - Why fuse, what to fuse
3. `xla_compilation.md` - How XLA optimizes
4. `jax_transformations.md` - jit, vmap, pmap explained
5. `tpu_architecture.md` - How TPUs work
6. `profiling.md` - Finding and fixing bottlenecks

*Labs:*
| Lab | Title | What you build | Key learning |
|-----|-------|----------------|--------------|
| lab01 | Triton Basics | Simple Triton kernels | Block-level programming |
| lab02 | Fused Attention | Attention kernel in Triton | Kernel fusion |
| lab03 | JAX Intro | Attention in JAX | Functional ML |
| lab04 | JAX JIT & vmap | Optimize with transformations | Compilation, batching |
| lab05 | Profiling | Profile and optimize a model | Find bottlenecks |

*Milestone:* Custom Triton attention kernel within 80% of Flash Attention performance.
