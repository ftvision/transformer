"""Tests for Lab 04: SGLang - Structured Generation."""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structured_gen import (
    format_chat_prompt,
    parse_json_response,
    validate_json_schema,
    generate_json,
    generate_with_regex,
    generate_choice,
    create_json_constraint,
    ValidationResult,
    sglang_available,
)


class TestFormatChatPrompt:
    """Tests for format_chat_prompt."""

    def test_chatml_format(self):
        """Should format with ChatML template."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]
        prompt = format_chat_prompt(messages, template="chatml")
        assert "<|im_start|>system" in prompt
        assert "You are helpful." in prompt
        assert "<|im_start|>user" in prompt
        assert "Hello!" in prompt

    def test_simple_format(self):
        """Should format with simple template."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        prompt = format_chat_prompt(messages, template="simple")
        assert "user" in prompt.lower()
        assert "Hello" in prompt

    def test_handles_assistant(self):
        """Should handle assistant messages."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"}
        ]
        prompt = format_chat_prompt(messages, template="simple")
        assert "Hello!" in prompt
        assert "How are you?" in prompt

    def test_empty_messages(self):
        """Should handle empty messages list."""
        prompt = format_chat_prompt([], template="simple")
        assert isinstance(prompt, str)


class TestParseJsonResponse:
    """Tests for parse_json_response."""

    def test_pure_json(self):
        """Should parse pure JSON string."""
        result = parse_json_response('{"name": "John", "age": 30}')
        assert result == {"name": "John", "age": 30}

    def test_json_with_prefix(self):
        """Should extract JSON with prefix text."""
        result = parse_json_response('Here is the result: {"x": 1}')
        assert result == {"x": 1}

    def test_json_with_suffix(self):
        """Should extract JSON with suffix text."""
        result = parse_json_response('{"x": 1} is the output')
        assert result == {"x": 1}

    def test_nested_json(self):
        """Should parse nested JSON."""
        result = parse_json_response('{"a": {"b": [1, 2, 3]}}')
        assert result == {"a": {"b": [1, 2, 3]}}

    def test_no_json(self):
        """Should return None when no JSON found."""
        result = parse_json_response("No JSON here at all")
        assert result is None

    def test_invalid_json(self):
        """Should return None for invalid JSON."""
        result = parse_json_response('{"name": John}')  # Missing quotes
        assert result is None

    def test_json_array(self):
        """Should parse JSON arrays."""
        result = parse_json_response('[1, 2, 3]')
        assert result == [1, 2, 3]


class TestValidateJsonSchema:
    """Tests for validate_json_schema."""

    def test_valid_string(self):
        """Should validate string type."""
        schema = {"type": "string"}
        result = validate_json_schema("hello", schema)
        assert result.is_valid is True

    def test_invalid_string(self):
        """Should reject non-string for string type."""
        schema = {"type": "string"}
        result = validate_json_schema(123, schema)
        assert result.is_valid is False

    def test_valid_integer(self):
        """Should validate integer type."""
        schema = {"type": "integer"}
        result = validate_json_schema(42, schema)
        assert result.is_valid is True

    def test_valid_object(self):
        """Should validate object with properties."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        result = validate_json_schema({"name": "John", "age": 30}, schema)
        assert result.is_valid is True

    def test_missing_required(self):
        """Should fail when required property missing."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        result = validate_json_schema({}, schema)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_enum_valid(self):
        """Should validate enum values."""
        schema = {
            "type": "string",
            "enum": ["red", "green", "blue"]
        }
        result = validate_json_schema("red", schema)
        assert result.is_valid is True

    def test_enum_invalid(self):
        """Should reject invalid enum value."""
        schema = {
            "type": "string",
            "enum": ["red", "green", "blue"]
        }
        result = validate_json_schema("yellow", schema)
        assert result.is_valid is False

    def test_array_type(self):
        """Should validate array type."""
        schema = {
            "type": "array",
            "items": {"type": "integer"}
        }
        result = validate_json_schema([1, 2, 3], schema)
        assert result.is_valid is True

    def test_array_invalid_items(self):
        """Should reject array with invalid items."""
        schema = {
            "type": "array",
            "items": {"type": "integer"}
        }
        result = validate_json_schema([1, "two", 3], schema)
        assert result.is_valid is False


class TestCreateJsonConstraint:
    """Tests for create_json_constraint."""

    def test_creates_constraint(self):
        """Should create a constraint object."""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        constraint = create_json_constraint(schema)
        assert constraint is not None

    def test_valid_prefix(self):
        """Should recognize valid JSON prefix."""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        constraint = create_json_constraint(schema)
        assert constraint.is_valid_prefix('{"x":')

    def test_complete_json(self):
        """Should recognize complete JSON."""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        constraint = create_json_constraint(schema)
        assert constraint.is_complete('{"x": 1}')

    def test_incomplete_json(self):
        """Should recognize incomplete JSON."""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        constraint = create_json_constraint(schema)
        assert not constraint.is_complete('{"x":')


# Tests that require a model - will be skipped without proper setup
MODEL_AVAILABLE = False  # Set to True when testing with actual model


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not available")
class TestGenerateJson:
    """Tests for generate_json (requires model)."""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Load model for testing."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return model, tokenizer

    def test_generates_valid_json(self, model_and_tokenizer):
        """Should generate valid JSON."""
        model, tokenizer = model_and_tokenizer
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        result = generate_json(model, tokenizer, "Name: John", schema)
        assert result is not None
        assert isinstance(result, dict)


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not available")
class TestGenerateChoice:
    """Tests for generate_choice (requires model)."""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Load model for testing."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return model, tokenizer

    def test_returns_valid_choice(self, model_and_tokenizer):
        """Should return one of the choices."""
        model, tokenizer = model_and_tokenizer
        choices = ["yes", "no", "maybe"]
        result = generate_choice(model, tokenizer, "Answer: ", choices)
        assert result in choices


class TestMilestone:
    """Integration tests for structured generation."""

    def test_json_parsing_workflow(self):
        """Complete JSON parsing workflow."""
        # Test data
        test_cases = [
            ('{"name": "Alice", "age": 25}', {"name": "Alice", "age": 25}),
            ('Result: {"x": 1, "y": 2}', {"x": 1, "y": 2}),
            ('[1, 2, 3]', [1, 2, 3]),
        ]

        for text, expected in test_cases:
            result = parse_json_response(text)
            assert result == expected, f"Failed for: {text}"

        print("\n✅ JSON parsing: All cases passed")

    def test_schema_validation_workflow(self):
        """Complete schema validation workflow."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"}
            },
            "required": ["name", "age"]
        }

        # Valid data
        valid = {"name": "John", "age": 30, "email": "john@example.com"}
        result = validate_json_schema(valid, schema)
        assert result.is_valid, "Valid data should pass"

        # Missing required
        missing = {"name": "John"}
        result = validate_json_schema(missing, schema)
        assert not result.is_valid, "Missing required should fail"

        # Wrong type
        wrong_type = {"name": "John", "age": "thirty"}
        result = validate_json_schema(wrong_type, schema)
        assert not result.is_valid, "Wrong type should fail"

        print("\n✅ Schema validation: All cases passed")

    def test_chat_formatting(self):
        """Test chat prompt formatting."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]

        # Test different templates
        for template in ["chatml", "simple"]:
            prompt = format_chat_prompt(messages, template=template)
            assert "Hello!" in prompt
            assert "How are you?" in prompt

        print("\n✅ Chat formatting: All templates work")

    def test_full_milestone(self):
        """Complete structured generation milestone test."""
        print("\n✅ Milestone Test - Structured Generation")

        # JSON parsing
        json_text = 'The result is: {"status": "success", "count": 42}'
        parsed = parse_json_response(json_text)
        assert parsed == {"status": "success", "count": 42}
        print("   JSON parsing: ✓")

        # Schema validation
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "count": {"type": "integer"}
            },
            "required": ["status"]
        }
        result = validate_json_schema(parsed, schema)
        assert result.is_valid
        print("   Schema validation: ✓")

        # Constraint creation
        constraint = create_json_constraint(schema)
        assert constraint.is_valid_prefix('{"status":')
        assert constraint.is_complete('{"status": "success"}')
        print("   Constraint creation: ✓")

        print("   All structured generation tests passed!")
