"""Shared test fixtures and configuration for fastai tests."""
import sys
import os
import pytest

# Ensure the fastai package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
