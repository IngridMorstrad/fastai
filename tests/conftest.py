"""Shared test fixtures and configuration for fastai tests."""
import sys
import os
import pytest

# Ensure the fastai package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Ensure test helper modules (e.g. _tracker_test_helpers) are importable
sys.path.insert(0, os.path.dirname(__file__))
