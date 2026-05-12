"""This test module is for tests that check the test fixtures themselves work.

Tests here are mainly used to manually verify fixtures with external dependencies
(e.g., file downloads) before creating a release.
"""
import pytest


@pytest.mark.slow
def test_large_files_download(large_files_dir):
    assert True
