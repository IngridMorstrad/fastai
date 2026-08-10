"""Shared test fixtures and configuration for fastai tests."""
import sys
import os

# Ensure the tests directory is on sys.path so shared helpers
# (e.g., _tracker_mock) can be imported by test modules.
sys.path.insert(0, os.path.dirname(__file__))

# Ensure the fastai package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
