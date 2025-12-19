# SGLang: Structured Generation and RadixAttention

## What is SGLang?

SGLang (Structured Generation Language) is a framework for efficient LLM programming, featuring:
- **Structured output generation** (JSON, regex, grammar)
- **RadixAttention** for KV-cache reuse across requests
- **High-level programming primitives** for complex LLM workflows

## The Structured Output Problem

LLMs generate free-form text, but applications often need structured data:

```python
# What we want
response = {"name": "John", "age": 30, "city": "NYC"}

# What LLMs might generate
"The person's name is John, who is 30 years old and lives in NYC."
"Here's the JSON: {name: John, age: 30}"  # Invalid JSON!
'{"name": "John", "age": "thirty"}'  # Wrong type!
```

**Structured generation** constrains the LLM to produce valid output.

## How Constrained Decoding Works

### Basic Idea

At each generation step, mask invalid tokens:

```python
def constrained_sample(logits, valid_tokens):
    # Mask invalid tokens
    mask = torch.full_like(logits, float('-inf'))
    mask[valid_tokens] = 0
    masked_logits = logits + mask

    # Sample from valid tokens only
    return sample(masked_logits)
```

### JSON Schema Constraint

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

# After generating '{"name": "', valid next tokens are:
# - Any string character
# - The closing quote "

# After generating '{"name": "John"', valid next tokens are:
# - , (comma to continue)
# - } (close if all required fields present - NO, age missing)
# So only , is valid!
```

### State Machine Approach

SGLang compiles JSON schemas into finite state machines:

```
State: START
  '{' → State: OBJECT_START

State: OBJECT_START
  '"' → State: KEY_START

State: KEY_START
  [a-z]* → State: KEY_CONTENT
  '"' → State: KEY_END

State: KEY_END
  ':' → State: COLON

State: COLON
  ' '? → State: VALUE_START

...
```

At each state, only valid tokens can be sampled.

## RadixAttention: KV-Cache Reuse

### The Problem

Many requests share common prefixes:

```
Request 1: "System: You are helpful.\n\nUser: What is 2+2?"
Request 2: "System: You are helpful.\n\nUser: What is 3+3?"
Request 3: "System: You are helpful.\n\nUser: Explain AI."
```

All share the system prompt! With naive caching, we compute KV for this prefix 3 times.

### RadixAttention Solution

Store KV-cache in a **radix tree** (prefix tree):

```
Root
└── "System: You are helpful.\n\nUser: "
    ├── "What is 2+2?" → KV for full request 1
    ├── "What is 3+3?" → KV for full request 2
    └── "Explain AI." → KV for full request 3
```

When a new request arrives:
1. Find longest matching prefix in tree
2. Reuse that KV-cache
3. Only compute KV for the new suffix

### Speedup

```
Without RadixAttention:
Request 1: Compute KV for 50 tokens (full)
Request 2: Compute KV for 50 tokens (full)
Request 3: Compute KV for 50 tokens (full)
Total: 150 token KV computations

With RadixAttention:
Request 1: Compute KV for 50 tokens
Request 2: Reuse 40 tokens, compute 10 new
Request 3: Reuse 40 tokens, compute 10 new
Total: 70 token KV computations (~2x speedup)
```

For applications with common prompts (chatbots, agents), this is huge!

## Using SGLang

### Installation

```bash
pip install sglang[all]
```

### Basic Generation

```python
import sglang as sgl

@sgl.function
def simple_qa(s, question):
    s += "Question: " + question + "\n"
    s += "Answer:" + sgl.gen("answer", max_tokens=100)

# Run
state = simple_qa.run(question="What is the capital of France?")
print(state["answer"])
```

### Structured JSON Output

```python
import sglang as sgl
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

@sgl.function
def extract_person(s, text):
    s += f"Extract person info from: {text}\n"
    s += "JSON: " + sgl.gen("person", regex=Person.schema_json())

state = extract_person.run(
    text="John is a 30-year-old living in New York."
)
person = Person.parse_raw(state["person"])
print(f"{person.name}, {person.age}, {person.city}")
```

### Constrained Generation with Regex

```python
@sgl.function
def generate_email(s):
    s += "Generate a valid email address: "
    s += sgl.gen(
        "email",
        regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    )

state = generate_email.run()
print(state["email"])  # Always a valid email format!
```

### Choice Selection

```python
@sgl.function
def sentiment(s, text):
    s += f"Text: {text}\n"
    s += "Sentiment: " + sgl.gen("sentiment", choices=["positive", "negative", "neutral"])

state = sentiment.run(text="I love this product!")
print(state["sentiment"])  # One of: positive, negative, neutral
```

### Forking for Parallel Generation

```python
@sgl.function
def parallel_qa(s, question):
    s += f"Question: {question}\n"

    # Fork into multiple branches
    forks = s.fork(3)
    for i, f in enumerate(forks):
        f += f"Answer {i+1}: " + sgl.gen(f"answer_{i}", max_tokens=50)

    # All answers generated in parallel
    return [state[f"answer_{i}"] for i, state in enumerate(forks)]
```

## SGLang Server

### Starting the Server

```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 30000
```

### Using the Server

```python
import sglang as sgl

# Set backend
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def chat(s, message):
    s += "User: " + message + "\n"
    s += "Assistant: " + sgl.gen("response", max_tokens=100)

state = chat.run(message="Hello!")
print(state["response"])
```

## Comparison: SGLang vs Outlines vs Guidance

| Feature | SGLang | Outlines | Guidance |
|---------|--------|----------|----------|
| JSON generation | ✅ | ✅ | ✅ |
| Regex constraints | ✅ | ✅ | ⚠️ |
| Grammar constraints | ✅ | ✅ | ⚠️ |
| RadixAttention | ✅ | ❌ | ❌ |
| High throughput | ✅ | ⚠️ | ⚠️ |
| Forking/branching | ✅ | ❌ | ✅ |
| Batched serving | ✅ | ⚠️ | ❌ |

## Performance Benefits

### Benchmark: JSON Generation

```
Task: Generate 1000 JSON objects with schema
Model: Llama-2-7B

Without constraints (post-hoc parsing):
- 30% invalid JSON
- Had to retry ~50% of requests
- Effective throughput: 50 obj/sec

With SGLang constraints:
- 100% valid JSON
- No retries needed
- Throughput: 80 obj/sec
```

### Benchmark: RadixAttention

```
Task: Chatbot with common system prompt
Concurrent users: 100

Without RadixAttention:
- Avg latency: 2.5s
- Throughput: 40 req/sec

With RadixAttention:
- Avg latency: 1.2s  (2x faster)
- Throughput: 85 req/sec (2x higher)
```

## When to Use SGLang

**Best for**:
- Structured output (JSON, code, forms)
- Applications with common prefixes (chatbots, agents)
- Complex LLM workflows (branching, selection)
- Production APIs needing reliable output formats

**Not ideal for**:
- Simple text generation (overkill)
- Models not supported by SGLang
- When you need the absolute lowest latency

## What's Next

Now you've seen the major inference frameworks: HuggingFace for flexibility, vLLM for throughput, llama.cpp for efficiency, and SGLang for structured generation. See `05_framework_comparison.md` for guidance on choosing the right tool.
