# Gradient Checkpointing: Trading Compute for Memory

## The Activation Memory Problem

During training, we need to store activations for the backward pass:

```
Forward:  x0 → [Layer 1] → x1 → [Layer 2] → x2 → ... → [Layer N] → loss
                  ↓            ↓            ↓
              Store x0      Store x1      Store x2  ... (for backward)
```

For a transformer with L layers, sequence length N, and hidden dimension d:
- Activations per layer: O(N × d)
- Total activations: O(L × N × d)

**Example (GPT-3 175B scale):**
- L = 96 layers
- N = 2048 tokens
- d = 12288
- Activations: ~120 GB per batch!

This often exceeds GPU memory, limiting batch size and training efficiency.

## The Checkpointing Idea

Instead of storing all activations, store only some ("checkpoints") and recompute the rest:

```
Without checkpointing:
Forward:  Store x0, x1, x2, x3, x4, x5, x6, x7  (8 activations)
Backward: Use stored activations

With checkpointing (every 4 layers):
Forward:  Store x0, x4  (2 checkpoints)
Backward: Recompute x1, x2, x3 from x0
          Recompute x5, x6, x7 from x4
```

**Trade-off**: ~33% more compute, but 75% less memory!

## How Checkpointing Works

### Standard Backward Pass

```python
# Forward pass - store all activations
activations = []
x = input
for layer in layers:
    x = layer(x)
    activations.append(x)  # Store for backward

# Backward pass - use stored activations
grad = loss_grad
for layer, act in zip(reversed(layers), reversed(activations)):
    grad = layer.backward(grad, act)  # Need activation!
```

### Checkpointed Backward Pass

```python
# Forward pass - store only checkpoints
checkpoints = [input]
x = input
for i, layer in enumerate(layers):
    x = layer(x)
    if (i + 1) % checkpoint_interval == 0:
        checkpoints.append(x)  # Store checkpoint

# Backward pass - recompute from checkpoints
grad = loss_grad
for segment_idx in reversed(range(num_segments)):
    # Recompute activations for this segment
    x = checkpoints[segment_idx]
    segment_activations = [x]
    for layer in layers[segment_idx * interval : (segment_idx + 1) * interval]:
        x = layer(x)
        segment_activations.append(x)

    # Backward through segment
    for layer, act in zip(reversed(segment_layers), reversed(segment_activations)):
        grad = layer.backward(grad, act)
```

## Checkpointing Strategies

### 1. Uniform Checkpointing

Checkpoint every k layers:

```
k = 4:
Layer:      1   2   3   4   5   6   7   8   9  10  11  12
Checkpoint: ✓           ✓           ✓           ✓
```

Memory: O(L/k × N × d)
Extra compute: O((k-1)/k × Forward) ≈ O(Forward) for large k

### 2. Sqrt Checkpointing

Optimal for uniform compute/memory trade-off:
- Checkpoint every √L layers
- Memory: O(√L × N × d)
- Extra compute: O(√L × Forward per layer)

```
For L = 16:
Checkpoint at: 0, 4, 8, 12
```

### 3. Selective Checkpointing

Only checkpoint expensive operations:

```python
# Don't checkpoint cheap ops (layernorm, dropout)
# Do checkpoint expensive ops (attention, FFN)

class TransformerBlock(nn.Module):
    def forward(self, x):
        # Checkpoint attention (expensive)
        attn_out = checkpoint(self.attention, x)

        # Don't checkpoint layernorm (cheap)
        x = self.norm1(x + attn_out)

        # Checkpoint FFN (expensive)
        ffn_out = checkpoint(self.ffn, x)

        x = self.norm2(x + ffn_out)
        return x
```

### 4. Activation Compression

Store compressed checkpoints:
- Quantize activations to FP16/INT8
- Use random projection to reduce dimension
- Decompress during backward

```python
def checkpoint_with_compression(x):
    # Compress: FP32 → FP16
    return x.half()

def restore_checkpoint(x_compressed):
    # Decompress: FP16 → FP32
    return x_compressed.float()
```

## PyTorch Implementation

### Using torch.utils.checkpoint

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class CheckpointedTransformer(nn.Module):
    def __init__(self, layers, use_checkpoint=True):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint:
                # Checkpoint this layer
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

# Or checkpoint sequential groups
model = nn.Sequential(*layers)
x = checkpoint_sequential(model, segments=4, input=x)
```

### HuggingFace Integration

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("gpt2")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Now training uses checkpointing
output = model(input_ids)
loss = output.loss
loss.backward()  # Recomputes activations
```

### Custom Checkpointing

```python
class CustomCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        # Don't save activations during forward
        ctx.run_function = run_function
        ctx.save_for_backward(*args)

        with torch.no_grad():
            output = run_function(*args)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Recompute activations during backward
        args = ctx.saved_tensors

        with torch.enable_grad():
            # Rerun forward to get activations
            detached_args = [a.detach().requires_grad_() for a in args]
            output = ctx.run_function(*detached_args)

        # Compute gradients
        grads = torch.autograd.grad(output, detached_args, grad_output)
        return (None,) + grads
```

## Memory-Compute Trade-offs

| Strategy | Memory | Extra Compute |
|----------|--------|---------------|
| No checkpointing | O(L × N × d) | 0 |
| Every layer | O(N × d) | ~100% |
| Every k layers | O(L/k × N × d) | ~(k-1)/k × 100% |
| √L checkpoints | O(√L × N × d) | ~100% |
| Selective (attn only) | O(L/2 × N × d) | ~30-50% |

## Combining Flash Attention + Checkpointing

Flash Attention already recomputes attention in backward:
- Forward: Don't store N×N attention matrix
- Backward: Recompute attention from Q, K, V

With gradient checkpointing on top:
- Don't store intermediate layer outputs
- Recompute everything from checkpoints

```python
class EfficientTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = FlashAttention(d_model, n_heads)  # Already recomputes
        self.ffn = FFN(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Flash Attention handles its own recomputation
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Checkpoint entire blocks
class EfficientTransformer(nn.Module):
    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return x
```

## Practical Recommendations

### When to Use Checkpointing

1. **Training large models**: >1B parameters
2. **Long sequences**: >2048 tokens
3. **Memory-constrained**: Can't fit desired batch size
4. **Fine-tuning**: When memory is tight

### When NOT to Use Checkpointing

1. **Inference**: No backward pass needed
2. **Small models**: Memory isn't the bottleneck
3. **Latency-critical**: Extra compute adds latency
4. **Already using FSDP/ZeRO**: May have sufficient memory

### Optimal Configuration

```python
# Recommended setup for large model training
model = TransformerModel(
    num_layers=48,
    hidden_size=4096,
    num_heads=32,
)

# Enable both Flash Attention and checkpointing
model.enable_flash_attention()
model.gradient_checkpointing_enable()

# Use mixed precision for additional memory savings
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Advanced: Offloading

For extreme memory constraints, offload checkpoints to CPU:

```python
class OffloadCheckpoint(nn.Module):
    def forward(self, x):
        # Move checkpoint to CPU during forward
        checkpoint = x.cpu()

        # Continue forward on GPU
        for layer in self.layers:
            x = layer(x)

        # Store CPU checkpoint for backward
        self.checkpoint = checkpoint
        return x

    def backward_hook(self, grad):
        # Move checkpoint back to GPU for recomputation
        x = self.checkpoint.cuda()
        # Recompute from checkpoint...
```

This adds PCIe transfer latency but enables training even larger models.

## What's Next

You now understand both Flash Attention and gradient checkpointing - the two main techniques for memory-efficient transformer training. See `07_references.md` for papers, implementations, and further reading.
