# Lab 04: SGLang - Structured Generation

## Objective

Generate structured outputs (JSON, constrained text) using SGLang.

## What You'll Build

Functions and utilities for:
- Constrained text generation
- JSON schema-based generation
- Regex-constrained generation
- Choice/selection generation

## Prerequisites

Read these docs first:
- `../docs/04_sglang.md`

## Instructions

1. Open `src/structured_gen.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

**Note**: Full SGLang requires installation. For testing, we simulate constrained generation.

## Functions to Implement

### `generate_json(model, tokenizer, prompt, schema)`
Generate valid JSON matching a schema.
- Use constrained decoding
- Ensure output parses as valid JSON
- Validate against schema

### `generate_with_regex(model, tokenizer, prompt, pattern)`
Generate text matching a regex pattern.
- Constrain generation to valid continuations
- Return text matching the pattern

### `generate_choice(model, tokenizer, prompt, choices)`
Generate from a fixed set of choices.
- Only allow tokens leading to valid choices
- Return the selected choice

### `format_chat_prompt(messages, template="chatml")`
Format messages into a prompt string.
- Support different templates (chatml, llama, etc.)
- Handle system, user, assistant roles

### `parse_json_response(text)`
Extract and parse JSON from generated text.
- Handle JSON embedded in other text
- Return parsed object or None

### `validate_json_schema(data, schema)`
Validate JSON data against a schema.
- Use jsonschema or manual validation
- Return (is_valid, errors)

### `create_json_constraint(schema)`
Create a constraint object from JSON schema.
- Parse schema to state machine
- Return constraint for generation

### `simulate_constrained_generation(prompt, constraint, model, tokenizer)`
Simulate constrained generation (for testing without full SGLang).
- Generate token by token
- Apply constraint at each step

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_structured_gen.py::TestGenerateJson

# Run with verbose output
uv run pytest tests/ -v
```

## Example Usage

```python
from structured_gen import (
    generate_json,
    generate_with_regex,
    generate_choice,
    parse_json_response,
    validate_json_schema
)

# JSON generation
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    },
    "required": ["name", "age"]
}
result = generate_json(
    model, tokenizer,
    "Extract: John is 30 years old",
    schema
)
print(result)  # {"name": "John", "age": 30}

# Regex-constrained generation
email = generate_with_regex(
    model, tokenizer,
    "Generate an email: ",
    r"[a-z]+@[a-z]+\.[a-z]+"
)
print(email)  # "john@example.com"

# Choice selection
sentiment = generate_choice(
    model, tokenizer,
    "The movie was great! Sentiment: ",
    ["positive", "negative", "neutral"]
)
print(sentiment)  # "positive"
```

## JSON Schema Support

```python
# Basic types
{"type": "string"}
{"type": "integer"}
{"type": "number"}
{"type": "boolean"}
{"type": "null"}

# Objects
{
    "type": "object",
    "properties": {
        "field1": {"type": "string"},
        "field2": {"type": "integer"}
    },
    "required": ["field1"]
}

# Arrays
{
    "type": "array",
    "items": {"type": "string"}
}

# Enums
{
    "type": "string",
    "enum": ["option1", "option2", "option3"]
}
```

## Hints

- JSON schema validation uses standard JSON Schema spec
- Regex patterns should be Python re-compatible
- Choice generation can use token masking
- State machines make constraint checking efficient

## Verification

All tests pass = you understand structured generation!

Milestone: Generate valid JSON that passes schema validation 100% of the time.
