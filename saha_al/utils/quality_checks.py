import re
import math
from saha_al.config import MAX_LENGTH_RATIO, MIN_LENGTH_RATIO

def check_leakage(original_text, anonymized_text, entities):
    """
    Checks if any PII values from the original_text leaked into anonymized_text.
    Returns: list of leaked entity values. Empty list means no leakage.
    """
    leaks = []
    anon_text_lower = anonymized_text.lower()
    
    for ent in entities:
        if not ent or "value" not in ent or not ent["value"]:
            continue
            
        val_lower = ent["value"].lower()
        
        # Don't check tiny values (e.g., 1-2 chars) that might naturally occur
        if len(val_lower) <= 2:
            continue
            
        # Optional: could use regex word boundaries \b but depends if the text
        # has punct attached. A simple 'in' check is safer for strict leakage detection
        if val_lower in anon_text_lower:
            leaks.append(ent["value"])
            
    return list(set(leaks))


def check_length_ratio(original_text, anonymized_text):
    """
    Ensures the anonymized text isn't excessively longer or shorter
    than the original, as a sanity check for annotator rewrites.
    """
    orig_len = len(original_text.strip().split())
    if orig_len == 0:
        return True # Can't check
        
    anon_len = len(anonymized_text.strip().split())
    ratio = anon_len / orig_len
    
    if ratio > MAX_LENGTH_RATIO or ratio < MIN_LENGTH_RATIO:
        return False
        
    return True
