"""
Text Augmentations for PII Anonymization Robustness
=====================================================
These transforms are applied on-the-fly during training to make the model
robust to real-world input variations (case, typos, whitespace, punctuation).

KEY DESIGN PRINCIPLE:
    The SAME transform is applied to BOTH input (original_text) AND target
    (anonymized_text) so the model learns the correct mapping regardless of
    surface-level text variations.

    Example:
        Original pair:
            input:  "My name is John Smith"
            target: "My name is Elara Vance"
        After lowercase augmentation:
            input:  "my name is john smith"
            target: "my name is elara vance"

Augmentations:
    1. Lowercase All         — "my name is john smith"
    2. Uppercase All         — "MY NAME IS JOHN SMITH"
    3. Title Case            — "My Name Is John Smith"
    4. Random Case (char)    — "mY nAmE iS jOhN sMiTh"
    5. Swap Case             — "mY NAME IS jOHN sMITH"
    6. Punctuation Removal   — strips commas, periods, colons, etc.
    7. Typo Injection        — random adjacent-key char swaps
    8. Whitespace Noise      — extra/missing spaces

References:
    - EDA: Easy Data Augmentation (Wei & Zou, 2019)
    - Character-level noise for NER robustness (Namysl et al., 2020)
    - Case augmentation for low-resource NLP (Xie et al., 2020)
"""

import random
import string
import re


# ============================================================
# INDIVIDUAL AUGMENTATION FUNCTIONS
# ============================================================
# Each function takes a text string and returns the augmented version.
# They are deterministic given the same random seed state.


def lowercase_all(text: str) -> str:
    """Convert entire text to lowercase."""
    return text.lower()


def uppercase_all(text: str) -> str:
    """Convert entire text to UPPERCASE."""
    return text.upper()


def title_case(text: str) -> str:
    """Convert to Title Case (capitalize first letter of each word)."""
    return text.title()


def random_case(text: str) -> str:
    """Randomly toggle case of each character."""
    return "".join(
        c.upper() if random.random() < 0.5 else c.lower()
        for c in text
    )


def swap_case(text: str) -> str:
    """Swap case of every character (upper ↔ lower)."""
    return text.swapcase()


def remove_punctuation(text: str) -> str:
    """
    Remove common punctuation marks (.,;:!?) but preserve:
    - @ (emails), + (phone numbers), - (hyphenated names/numbers),
    - ' (contractions like "don't"), / (dates like 15/07/2023)
    """
    # Only remove these specific punctuation chars
    chars_to_remove = set(".,;:!?\"()[]{}\\")
    return "".join(c for c in text if c not in chars_to_remove)


# Keyboard adjacency map for realistic typo simulation
_KEYBOARD_NEIGHBORS = {
    'a': 'sqwz', 'b': 'vngh', 'c': 'xdfv', 'd': 'sfce', 'e': 'rdws',
    'f': 'dgcr', 'g': 'fhtb', 'h': 'gjyn', 'i': 'ujko', 'j': 'hkum',
    'k': 'jlio', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'etdf', 's': 'adwxz', 't': 'rfyg',
    'u': 'yihj', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tugh',
    'z': 'asx',
    '0': '9', '1': '2', '2': '13', '3': '24', '4': '35',
    '5': '46', '6': '57', '7': '68', '8': '79', '9': '80',
}


def inject_typos(text: str, typo_rate: float = 0.05) -> str:
    """
    Inject realistic typos by replacing characters with keyboard-adjacent keys.
    
    Args:
        text: input text
        typo_rate: probability of each character being replaced (default 5%)
    
    Only affects alphanumeric characters. Preserves spaces and special chars.
    """
    result = []
    for c in text:
        if random.random() < typo_rate and c.lower() in _KEYBOARD_NEIGHBORS:
            neighbor = random.choice(_KEYBOARD_NEIGHBORS[c.lower()])
            # Preserve original case
            result.append(neighbor.upper() if c.isupper() else neighbor)
        else:
            result.append(c)
    return "".join(result)


def whitespace_noise(text: str, noise_rate: float = 0.1) -> str:
    """
    Add whitespace noise: randomly double spaces or remove spaces between words.
    
    Args:
        text: input text
        noise_rate: probability of each space being modified (default 10%)
    """
    result = []
    for c in text:
        if c == ' ' and random.random() < noise_rate:
            # Either double the space or skip it
            if random.random() < 0.5:
                result.append('  ')  # double space
            # else: skip the space entirely (remove it)
        else:
            result.append(c)
    return "".join(result)


# ============================================================
# AUGMENTATION REGISTRY
# ============================================================
# Maps augmentation name → (function, apply_to_both)
#
# apply_to_both: if True, the SAME transform is applied to both
# input and target. If False, only applied to input (for transforms
# that should only affect the input side, like typos).

AUGMENTATION_REGISTRY = {
    "lowercase":           (lowercase_all,      True),
    "uppercase":           (uppercase_all,      True),
    "title_case":          (title_case,         True),
    "random_case":         (random_case,        True),
    "swap_case":           (swap_case,          True),
    "remove_punctuation":  (remove_punctuation, True),
    "typo":                (inject_typos,       False),  # only on input
    "whitespace_noise":    (whitespace_noise,   False),  # only on input
}


# ============================================================
# AUGMENTATION PIPELINE
# ============================================================

class TextAugmentor:
    """
    On-the-fly text augmentation pipeline for training.
    
    At each __call__, randomly decides whether to augment, and if so,
    picks one augmentation from the enabled list.
    
    Usage:
        augmentor = TextAugmentor(
            augmentation_prob=0.3,
            enabled_augmentations=["lowercase", "uppercase", "random_case", "typo"],
        )
        
        # In dataset __getitem__:
        input_text, target_text = augmentor(input_text, target_text)
    
    Args:
        augmentation_prob: probability of applying ANY augmentation to a sample.
            0.0 = no augmentation, 1.0 = always augment.
            Recommended: 0.3-0.5 (so model still sees ~50-70% clean data).
        
        enabled_augmentations: list of augmentation names to use.
            See AUGMENTATION_REGISTRY for valid names.
        
        augmentation_weights: optional dict mapping augmentation name → weight.
            Higher weight = more likely to be chosen. Default: uniform.
    """
    
    def __init__(
        self,
        augmentation_prob: float = 0.3,
        enabled_augmentations: list[str] = None,
        augmentation_weights: dict[str, float] = None,
    ):
        self.augmentation_prob = augmentation_prob
        
        # Default: all case augmentations + punctuation removal
        if enabled_augmentations is None:
            enabled_augmentations = [
                "lowercase", "uppercase", "title_case",
                "random_case", "swap_case",
                "remove_punctuation", "typo", "whitespace_noise",
            ]
        
        # Validate
        for name in enabled_augmentations:
            if name not in AUGMENTATION_REGISTRY:
                raise ValueError(
                    f"Unknown augmentation '{name}'. "
                    f"Available: {list(AUGMENTATION_REGISTRY.keys())}"
                )
        
        self.enabled = enabled_augmentations
        
        # Build weights list (for weighted random choice)
        if augmentation_weights is not None:
            self.weights = [augmentation_weights.get(name, 1.0) for name in self.enabled]
        else:
            # Default weights: case augmentations get higher weight since that's
            # the primary issue; typo/whitespace are less common
            default_weights = {
                "lowercase": 3.0,       # most important — user input is often lowercase
                "uppercase": 1.5,       # less common but worth handling
                "title_case": 1.5,      # common in forms/headers
                "random_case": 1.0,     # edge case robustness
                "swap_case": 0.5,       # rare but adds diversity
                "remove_punctuation": 1.0,
                "typo": 1.5,            # very common in real user input
                "whitespace_noise": 1.0,
            }
            self.weights = [default_weights.get(name, 1.0) for name in self.enabled]
    
    def __call__(
        self, input_text: str, target_text: str
    ) -> tuple[str, str]:
        """
        Maybe apply an augmentation to the input/target pair.
        
        Args:
            input_text: original text (with real PII)
            target_text: anonymized text (with fake PII)
        
        Returns:
            (augmented_input, augmented_target)
        """
        # Random chance: skip augmentation entirely
        if random.random() > self.augmentation_prob:
            return input_text, target_text
        
        # Pick one augmentation (weighted random)
        chosen_name = random.choices(self.enabled, weights=self.weights, k=1)[0]
        aug_fn, apply_to_both = AUGMENTATION_REGISTRY[chosen_name]
        
        # For transforms that use randomness (random_case, typo, whitespace),
        # we need to apply them independently to input and target since the
        # random positions won't align between different-length texts.
        # The case transforms that are deterministic (lowercase, uppercase, etc.)
        # naturally produce consistent results.
        
        augmented_input = aug_fn(input_text)
        
        if apply_to_both:
            augmented_target = aug_fn(target_text)
        else:
            augmented_target = target_text  # keep target clean
        
        return augmented_input, augmented_target
    
    def __repr__(self) -> str:
        return (
            f"TextAugmentor(prob={self.augmentation_prob}, "
            f"augmentations={self.enabled})"
        )


# ============================================================
# QUICK DEMO / TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TEXT AUGMENTATION DEMO")
    print("=" * 70)
    
    input_text = "My name is John Smith and my email is john.smith@gmail.com."
    target_text = "My name is Elara Vance and my email is contact@example.com."
    
    print(f"\n  Original Input:  {input_text}")
    print(f"  Original Target: {target_text}")
    
    print(f"\n  {'Augmentation':<22} {'Input':<55} {'Target'}")
    print("  " + "-" * 120)
    
    for name, (fn, apply_both) in AUGMENTATION_REGISTRY.items():
        random.seed(42)  # reproducible demo
        aug_input = fn(input_text)
        aug_target = fn(target_text) if apply_both else target_text
        print(f"  {name:<22} {aug_input:<55} {aug_target}")
    
    print(f"\n  {'─' * 70}")
    print(f"  Pipeline demo (prob=1.0, 10 random samples):")
    print(f"  {'─' * 70}")
    
    augmentor = TextAugmentor(augmentation_prob=1.0)
    for i in range(10):
        aug_in, aug_tgt = augmentor(input_text, target_text)
        print(f"  [{i+1:2d}] IN:  {aug_in}")
        print(f"       OUT: {aug_tgt}")
        print()
