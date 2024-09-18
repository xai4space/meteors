import sys
import os
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import commit_processor


def test_parse_commit():
    mock_commit = MagicMock(spec=commit_processor.Commit)
    mock_commit.message = "feat: Add new feature"
    assert commit_processor.parse_commit(mock_commit) == ("feat", "Add new feature")

    mock_commit.message = "Invalid commit message"
    assert commit_processor.parse_commit(mock_commit) == (None, "Invalid commit message")


@pytest.fixture
def mock_repo():
    with patch("commit_processor.Repo") as MockRepo:
        mock_repo = MagicMock()
        MockRepo.return_value = mock_repo
        yield mock_repo


def test_generate_changelog(mock_repo):
    mock_commits = [
        MagicMock(spec=commit_processor.Commit, message="feat: Add feature A"),
        MagicMock(spec=commit_processor.Commit, message="fix: Fix bug B"),
        MagicMock(spec=commit_processor.Commit, message="docs: Update README"),
    ]
    mock_repo.iter_commits.return_value = mock_commits

    changelog = commit_processor.generate_changelog("/fake/path", "v1.0.0", "v0.9.0")

    expected_changelog = """### 🔨 Features
- Add feature A

### 🩺 Bug Fixes
- Fix bug B

### 📚 Documentation
- Update README
"""
    assert changelog == expected_changelog


def test_generate_changelog_no_previous_tag(mock_repo):
    mock_commits = [
        MagicMock(spec=commit_processor.Commit, message="feat: Initial feature"),
        MagicMock(spec=commit_processor.Commit, message="docs: Initial documentation"),
    ]
    mock_repo.iter_commits.return_value = mock_commits

    changelog = commit_processor.generate_changelog("/fake/path", "v1.0.0")

    expected_changelog = """### 🔨 Features
- Initial feature

### 📚 Documentation
- Initial documentation
"""
    assert changelog == expected_changelog
    mock_repo.iter_commits.assert_called_once_with()


@patch("commit_processor.open", new_callable=mock_open, read_data="# Changelog\n\n## Old content\n")
@patch("commit_processor.Repo")
def test_update_changelog(MockRepo, mock_file):
    mock_repo = MagicMock()
    MockRepo.return_value = mock_repo

    mock_commits = [MagicMock(spec=commit_processor.Commit, message="feat: Add feature X")]
    mock_repo.iter_commits.return_value = mock_commits

    mock_head_commit = MagicMock()
    mock_head_commit.committed_datetime = datetime(2023, 1, 1)
    mock_repo.head.commit = mock_head_commit

    commit_processor.update_changelog("/fake/path", "v1.0.0")

    expected_content = """# Changelog

## v1.0.0 (2023-01-01)

### 🔨 Features
- Add feature X

## Old content
"""
    mock_file().write.assert_called_once_with(expected_content)
