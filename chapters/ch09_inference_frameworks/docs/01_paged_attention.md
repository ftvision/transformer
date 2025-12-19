# PagedAttention: Virtual Memory for KV-Cache

## The Memory Fragmentation Problem

From Chapter 8, you know that KV-cache memory grows with:
```
KV memory = batch × seq_len × layers × 2 × d_model × dtype_size
```

But there's a hidden problem: **memory fragmentation**.

### The Naive Approach

Traditional serving allocates contiguous memory for each request:

```
Request 1: [=============================] 2048 tokens pre-allocated
Request 2: [=============================] 2048 tokens pre-allocated
Request 3: [=============================] 2048 tokens pre-allocated
```

**Problems**:
1. We allocate for max_seq_len even if request is short
2. Memory is wasted on unused slots
3. When requests complete, memory becomes fragmented

```
Memory after some requests complete:
[Req1: 500 tokens used][    WASTED    ][Req3: 800 tokens][  WASTED  ]
```

### Real-World Impact

For a 13B model serving batch of 256 requests:
- Naive allocation: 256 × 2048 × (KV per token) ≈ **32 GB**
- Actual usage: Maybe 256 × 500 average ≈ **8 GB**
- **Waste: 75% of allocated memory!**

## The PagedAttention Solution

PagedAttention, introduced by vLLM, applies virtual memory concepts to KV-cache.

### Core Idea: Pages (Blocks)

Instead of contiguous allocation, divide KV-cache into fixed-size **blocks**:

```
Block size = 16 tokens (typical)

Physical Memory (GPU):
[Block0][Block1][Block2][Block3][Block4][Block5][Block6]...

Request 1 (35 tokens): Block0 → Block1 → Block2 (3 blocks)
Request 2 (20 tokens): Block3 → Block4 (2 blocks)
Request 3 (50 tokens): Block5 → Block6 → Block7 → Block8 (4 blocks)
```

### The Block Table

Each request has a **block table** mapping logical to physical blocks:

```
Request 1 Block Table:
  Logical Block 0 → Physical Block 0
  Logical Block 1 → Physical Block 1
  Logical Block 2 → Physical Block 2

Request 2 Block Table:
  Logical Block 0 → Physical Block 3
  Logical Block 1 → Physical Block 4
```

This is exactly like OS page tables for virtual memory!

## How PagedAttention Works

### Step 1: Request Arrives

```python
def new_request(prompt_tokens):
    # Calculate blocks needed for prompt
    num_blocks = ceil(len(prompt_tokens) / block_size)

    # Allocate physical blocks
    block_table = allocate_blocks(num_blocks)

    # Store KV for prompt tokens
    for i, block_id in enumerate(block_table):
        start = i * block_size
        end = min((i + 1) * block_size, len(prompt_tokens))
        physical_blocks[block_id] = compute_kv(prompt_tokens[start:end])

    return block_table
```

### Step 2: Generation (Decode)

```python
def generate_token(request):
    # Check if current block has space
    if current_block_full(request):
        # Allocate new block
        new_block = allocate_block()
        request.block_table.append(new_block)

    # Compute attention using block table
    # The magic: attention kernel reads from scattered physical blocks
    output = paged_attention(
        query=request.current_query,
        block_table=request.block_table,
        kv_cache=physical_blocks
    )

    return output
```

### Step 3: Request Completes

```python
def complete_request(request):
    # Free all blocks
    for block_id in request.block_table:
        free_block(block_id)
```

No fragmentation! Freed blocks can be reused by any new request.

## The Attention Kernel

The paged attention kernel must handle non-contiguous memory:

```python
# Pseudocode for paged attention
def paged_attention_kernel(query, block_table, kv_cache, block_size):
    output = zeros_like(query)
    max_score = -inf

    for logical_block_idx, physical_block_idx in enumerate(block_table):
        # Load this block's K, V
        k_block = kv_cache.k[physical_block_idx]  # Shape: (block_size, head_dim)
        v_block = kv_cache.v[physical_block_idx]

        # Compute attention scores for this block
        scores = query @ k_block.T / sqrt(d_k)

        # Online softmax update
        block_max = scores.max()
        new_max = max(max_score, block_max)

        # Rescale previous output
        output *= exp(max_score - new_max)

        # Add this block's contribution
        weights = exp(scores - new_max)
        output += weights @ v_block

        max_score = new_max

    # Normalize
    output /= output_sum

    return output
```

Key insight: We iterate through blocks, computing attention incrementally using **online softmax** (just like Flash Attention!).

## Memory Efficiency

### Before PagedAttention
```
Allocated: [=================][=================][=================]
Used:      [====]             [========]         [==]
Waste:     ~75%
```

### After PagedAttention
```
Physical Blocks: [B0][B1][B2][B3][B4][B5][B6][B7]...

Req1: B0→B1 (uses 2 blocks for 20 tokens)
Req2: B2→B3→B4→B5 (uses 4 blocks for 60 tokens)
Req3: B6 (uses 1 block for 10 tokens)

Waste: Only last block of each request (~block_size/2 average)
```

With block_size=16, waste per request ≈ 8 tokens. For 256 requests: 2048 tokens wasted vs 400K+ tokens wasted with naive allocation.

## Advanced Features

### Copy-on-Write (CoW)

For parallel sampling (beam search, multiple completions):

```
Original request: Block0 → Block1 → Block2
Fork for beam 1:  Block0 → Block1 → Block2 → Block3
Fork for beam 2:  Block0 → Block1 → Block2 → Block4 (shares prefix!)
```

Prefix blocks are **shared** until modified. Only copy when writing.

### Prefix Caching

Common prompts can share KV-cache:

```
System prompt: "You are a helpful assistant..."
→ Pre-cached in blocks B100-B110

Request 1: B100-B110 → B0 → B1 (user query)
Request 2: B100-B110 → B2 → B3 (different user query)
```

Both requests share the system prompt's KV-cache!

## Implementation in vLLM

vLLM's PagedAttention is implemented in CUDA:

```
vllm/
├── attention/
│   ├── backends/
│   │   ├── paged_attention.py    # High-level interface
│   │   └── flash_attn.py         # Flash Attention integration
│   └── ops/
│       └── paged_attn.py         # CUDA kernel bindings
├── core/
│   ├── block_manager.py          # Block allocation logic
│   └── scheduler.py              # Request scheduling
```

The scheduler and block manager work together:
1. Scheduler decides which requests to process
2. Block manager allocates/frees blocks
3. Attention kernel reads from physical blocks via block tables

## Performance Comparison

| Metric | Naive Allocation | PagedAttention |
|--------|------------------|----------------|
| Memory utilization | ~25-30% | ~95%+ |
| Max batch size | Limited by fragmentation | Limited only by total memory |
| Throughput | Lower | 2-4x higher |
| Latency | Similar | Similar |

The throughput gain comes from being able to run larger batches!

## What's Next

Now that you understand how PagedAttention manages memory, let's see how vLLM puts it all together with continuous batching and other optimizations. See `02_vllm_architecture.md`.
