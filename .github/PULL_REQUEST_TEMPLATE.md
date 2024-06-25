# Contributing to Meteors

Thank you for contributing to meteors! Please follow the guidelines below to ensure a smooth and efficient contribution process.

## PR Checklist

- [ ] **PR Title**: `"semantic tag: description"`

  - **Package**: Use the appropriate semantic tag for your PR:
    - `feat`: A new feature.
    - `fix`: A bug fix.
    - `docs`: Documentation only changes.
    - `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).
    - `refactor`: A code change that neither fixes a bug nor adds a feature.
    - `perf`: A code change that improves performance.
    - `test`: Adding missing tests or correcting existing tests.
    - `chore`: Changes to the build process or auxiliary tools and libraries such as documentation generation.
    - `build`: Changes that affect the build system or external dependencies.
  - **Example**: `"fix: typo in README.md"`

- [ ] **Related Issue(s)**: If your PR fixes an issue, please link it in the PR description.

- [ ] **PR Message**: **_Delete this entire checklist_** and replace it with:

  - **Description**: A detailed description of the change.
  - **Dependencies**: Any dependencies required for this change.

- [ ] **Add Tests and Docs**: If adding a new features, please include:

  1. A test for the feature, preferably unit tests that do not rely on network access.
  2. An example notebook showing its use, located in the `examples` directory.

- [ ] **Lint**: Ensure your code passes linting and all tests:

  - Code is formatted with `pre-commit` hooks. Run `pre-commit install` and `pre-commit run --all-files` to format your code.

- [ ] **Tests**: Ensure all tests pass:
  - Run `pytest` in the root directory to run all tests.

## Additional Guidelines

- Import optional dependencies within functions.
- Avoid adding dependencies to `pyproject.toml` files unless required for unit tests.
- Most PRs should not modify more than one package.
- Ensure changes are backwards compatible.

## Review Process

If no one reviews your PR within a few days, please `@-mention` one of the maintainers.
