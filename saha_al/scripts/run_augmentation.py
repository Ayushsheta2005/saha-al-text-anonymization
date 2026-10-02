#!/usr/bin/env python3
"""
Run data augmentation on gold-standard entries.

Usage:
    python scripts/run_augmentation.py --strategy all
    python scripts/run_augmentation.py --strategy swap --multiplier 6
    python scripts/run_augmentation.py --strategy template --count 10000
    python scripts/run_augmentation.py --strategy eda --multiplier 5 --alpha 0.15
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from saha_al.augmentation import main

if __name__ == "__main__":
    main()
