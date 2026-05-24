"""
Shared pytest configuration and fixtures.

Adds the project root to sys.path so tests can import rag.* and utils.*
without installing the package.
"""

import os
import sys

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
