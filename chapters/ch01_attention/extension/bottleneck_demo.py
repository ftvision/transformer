"""
Bottleneck Problem Demo: Understanding Hidden States and Attention

This module demonstrates:
1. What hidden states (h1, h2, ...) actually are
2. Why compressing to a single vector loses information (bottleneck)
3. How attention solves this by preserving all hidden states

Run this file directly to see the demonstrations.
"""

import numpy as np

np.random.seed(42)


# =============================================================================
# Part 1: What are Hidden States?
# =============================================================================

def simulate_rnn_encoder(sentence: list[str], hidden_size: int = 8) -> dict:
    """
    Simulates an RNN encoder processing a sentence word by word.

    Each hidden state h_t captures information about:
    - The current word
    - All previous context (compressed)

    Returns all hidden states, not just the final one.
    """
    # Pretend each word has an embedding (in reality, these are learned)
    word_embeddings = {
        "The": np.array([0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        "cat": np.array([0.0, 0.0, 0.9, 0.8, 0.1, 0.0, 0.0, 0.0]),  # "cat" has strong animal signal
        "sat": np.array([0.0, 0.0, 0.1, 0.0, 0.7, 0.6, 0.0, 0.0]),  # "sat" has action signal
        "on":  np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2]),
        "the": np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "mat": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7]),  # "mat" has object signal
    }

    hidden_states = []
    h = np.zeros(hidden_size)  # Initial hidden state

    print("=== RNN Encoder Processing ===\n")

    for i, word in enumerate(sentence):
        embedding = word_embeddings.get(word, np.zeros(hidden_size))

        # Simplified RNN update: h_new = tanh(h_old * 0.5 + embedding)
        # Real RNNs have learned weight matrices, but this shows the concept
        h = np.tanh(h * 0.5 + embedding)
        hidden_states.append(h.copy())

        print(f"h{i+1} (after '{word}'): {np.round(h, 2)}")

    print(f"\n→ Notice: h2 (after 'cat') has strong values at positions 2,3")
    print(f"→ These encode 'cat' information that might be needed later\n")

    return {
        "all_hidden_states": hidden_states,  # [h1, h2, h3, h4, h5, h6]
        "final_hidden_state": hidden_states[-1],  # h6 only (the bottleneck!)
        "words": sentence,
    }


# =============================================================================
# Part 2: The Bottleneck Problem
# =============================================================================

def demonstrate_bottleneck(encoder_output: dict):
    """
    Shows why using only the final hidden state loses information.
    """
    print("=== The Bottleneck Problem ===\n")

    final_state = encoder_output["final_hidden_state"]
    all_states = encoder_output["all_hidden_states"]

    print("WITHOUT attention (bottleneck):")
    print(f"  Decoder only sees: {np.round(final_state, 2)}")
    print(f"  → The strong 'cat' signal from h2 has been diluted!\n")

    # Show how much "cat" information was lost
    h2_cat_signal = all_states[1][2:4]  # positions 2,3 in h2
    final_cat_signal = final_state[2:4]

    print(f"  'cat' signal in h2:    {np.round(h2_cat_signal, 2)}")
    print(f"  'cat' signal in h6:    {np.round(final_cat_signal, 2)}")
    print(f"  → Information lost: {np.round(h2_cat_signal - final_cat_signal, 2)}\n")

    print("WITH attention:")
    print(f"  Decoder can access ALL hidden states: h1, h2, h3, h4, h5, h6")
    print(f"  → When translating 'cat', it can directly look at h2")
    print(f"  → No information is lost!\n")


# =============================================================================
# Part 3: How Attention Solves It
# =============================================================================

def attention_mechanism(
    query: np.ndarray,
    keys: list[np.ndarray],
    values: list[np.ndarray],
    words: list[str]
) -> np.ndarray:
    """
    Simple attention: query attends to all key-value pairs.

    In encoder-decoder attention:
    - Query comes from the decoder (what am I looking for?)
    - Keys come from encoder hidden states (what does each position contain?)
    - Values come from encoder hidden states (what to retrieve?)

    For simplicity, we use keys = values = hidden states here.
    """
    # Compute attention scores (dot product similarity)
    scores = np.array([np.dot(query, key) for key in keys])

    # Softmax to get weights that sum to 1
    exp_scores = np.exp(scores - np.max(scores))  # subtract max for numerical stability
    weights = exp_scores / exp_scores.sum()

    # Weighted sum of values
    context = sum(w * v for w, v in zip(weights, values))

    print("Attention weights:")
    for word, weight in zip(words, weights):
        bar = "█" * int(weight * 30)
        print(f"  {word:6s}: {weight:.2f} {bar}")

    return context, weights


def demonstrate_attention(encoder_output: dict):
    """
    Shows how attention allows the decoder to focus on relevant parts.
    """
    print("=== Attention in Action ===\n")

    all_states = encoder_output["all_hidden_states"]
    words = encoder_output["words"]

    # Scenario: Decoder is generating "chat" (French for "cat")
    # It needs to find "cat" in the English input

    # The decoder creates a query: "I'm looking for an animal word"
    # (In reality, this is learned; here we craft it to match "cat")
    query_for_cat = np.array([0.0, 0.0, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0])

    print("Decoder is generating 'chat' (French for 'cat')")
    print(f"Decoder query (looking for animal): {np.round(query_for_cat, 2)}\n")

    context, weights = attention_mechanism(
        query=query_for_cat,
        keys=all_states,      # In simple attention, keys = hidden states
        values=all_states,    # values = hidden states too
        words=words
    )

    print(f"\n→ Attention focuses on 'cat' (weight={weights[1]:.2f})")
    print(f"→ The decoder gets 'cat' information directly from h2!")
    print(f"→ No bottleneck - the information was preserved.\n")

    # Now show a different query
    print("-" * 50)
    print("\nDecoder is generating 'tapis' (French for 'mat')")
    query_for_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.6])
    print(f"Decoder query (looking for object): {np.round(query_for_mat, 2)}\n")

    context, weights = attention_mechanism(
        query=query_for_mat,
        keys=all_states,
        values=all_states,
        words=words
    )

    print(f"\n→ Attention now focuses on 'mat' (weight={weights[5]:.2f})")
    print(f"→ Same hidden states, different query, different focus!\n")


# =============================================================================
# Part 4: Encoder-Decoder Flow
# =============================================================================

def demonstrate_translation_flow():
    """
    Shows the full encoder-decoder flow with attention.
    """
    print("=== Full Translation Flow ===\n")

    print("STEP 1: Encoder processes entire English sentence")
    print("-" * 50)
    print('"The cat sat on the mat"')
    print("     ↓")
    print("  [h1] [h2] [h3] [h4] [h5] [h6]")
    print("     ↓")
    print("  All hidden states stored (no bottleneck!)")
    print()

    print("STEP 2: Decoder generates French one word at a time")
    print("-" * 50)

    steps = [
        ("<START>", "Le", "article", [0.3, 0.1, 0.1, 0.1, 0.3, 0.1]),
        ("Le", "chat", "cat (animal)", [0.05, 0.75, 0.05, 0.05, 0.05, 0.05]),
        ("Le chat", "était", "sat (action)", [0.05, 0.1, 0.65, 0.05, 0.05, 0.1]),
        ("Le chat était", "assis", "sat (more)", [0.05, 0.1, 0.6, 0.1, 0.05, 0.1]),
        ("Le chat était assis", "sur", "on (preposition)", [0.05, 0.05, 0.1, 0.6, 0.1, 0.1]),
        ("Le chat était assis sur", "le", "the (article)", [0.1, 0.05, 0.05, 0.1, 0.6, 0.1]),
        ("Le chat était assis sur le", "tapis", "mat (object)", [0.05, 0.05, 0.05, 0.05, 0.1, 0.7]),
    ]

    english_words = ["The", "cat", "sat", "on", "the", "mat"]

    for i, (context, output, focus, weights) in enumerate(steps):
        print(f"\nStep {i+1}:")
        print(f"  Decoder input: '{context}'")
        print(f"  Looking for: {focus}")
        print(f"  Attention weights:")
        for word, w in zip(english_words, weights):
            bar = "█" * int(w * 20)
            print(f"    {word:6s}: {w:.2f} {bar}")
        print(f"  → Outputs: '{output}'")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    sentence = ["The", "cat", "sat", "on", "the", "mat"]

    # Part 1: Show what hidden states are
    encoder_output = simulate_rnn_encoder(sentence)

    print("=" * 60)

    # Part 2: Show the bottleneck problem
    demonstrate_bottleneck(encoder_output)

    print("=" * 60)

    # Part 3: Show how attention solves it
    demonstrate_attention(encoder_output)

    print("=" * 60)

    # Part 4: Show the full translation flow
    demonstrate_translation_flow()
