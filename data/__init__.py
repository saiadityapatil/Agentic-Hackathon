"""Data loading and extraction utilities for the multi-agent system.

This package handles:
- Azure cost and metrics data loading
- GitHub code extraction and analysis
- Terraform configuration parsing
- Git diff processing
"""

from data.github_extractor import extract_repo_code

__all__ = [
    'extract_repo_code',
]
