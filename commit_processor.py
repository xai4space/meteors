from __future__ import annotations
import re
from git import Repo
from git.objects.commit import Commit
from collections import defaultdict


def parse_commit(commit: Commit) -> tuple[str | None, str]:
    pattern = r"^(\w+): ([^\n]+)"
    match = re.match(pattern, commit.message.strip())
    if match:
        return match.groups()  # type: ignore
    return None, commit.message.strip()


def generate_changelog(repo_path: str, current_tag: str, previous_tag: str | None = None) -> str:
    repo = Repo(repo_path)

    if previous_tag:
        commits = list(repo.iter_commits(f"{previous_tag}..HEAD"))
    else:
        commits = list(repo.iter_commits())

    grouped_commits = defaultdict(list)
    for commit in commits:
        type, message = parse_commit(commit)
        if type:
            grouped_commits[type].append(message)

    changelog = []
    type_emojis = {
        "feat": "🔨",
        "fix": "🩺",
        "docs": "📚",
        "style": "🛎️",
        "refactor": "🛎️",
        "perf": "🛎️",
        "test": "🛎️",
        "chore": "🛎️",
        "build": "🔨",
    }

    type_titles = {
        "feat": "Features",
        "fix": "Bug Fixes",
        "docs": "Documentation",
        "style": "Styles",
        "refactor": "Code Refactoring",
        "perf": "Performance Improvements",
        "test": "Tests",
        "chore": "Chores",
        "build": "Build System",
    }

    for type, commits in grouped_commits.items():
        emoji = type_emojis.get(type, "")
        title = type_titles.get(type, type.capitalize())
        changelog.append(f"### {emoji} {title}")
        for message in commits:
            changelog.append(f"- {message}")
        changelog.append("")

    return "\n".join(changelog)


def update_changelog(repo_path: str, current_tag: str, previous_tag: str | None = None) -> str:
    changelog = generate_changelog(repo_path, current_tag, previous_tag)
    repo = Repo(repo_path)
    date = repo.head.commit.committed_datetime.strftime("%Y-%m-%d")

    with open("changelog.md", "r+") as f:
        content = f.read()
        content_without_header = re.sub(r"^# Changelog\n\n", "", content)
        f.seek(0, 0)
        f.write(f"# Changelog\n\n## {current_tag} ({date})\n\n{changelog}\n{content_without_header}")

    return changelog


if __name__ == "__main__":
    import sys

    repo_path = sys.argv[1]
    current_tag = sys.argv[2]
    previous_tag = sys.argv[3] if len(sys.argv) > 3 else None
    if previous_tag is not None and len(previous_tag) == 0:
        previous_tag = None

    changelog = update_changelog(repo_path, current_tag, previous_tag)
    print(changelog)
