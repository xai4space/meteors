# Contributing to Meteors

Thank you for contributing to Meteors! Please follow the guidelines below to ensure a smooth and efficient contribution process.

## PR Checklist

- [ ] **PR Title**: `"semantic tag: description"`

  - **Package**: Use the appropriate semantic tag for your PR:
    - `fix`: A bug fix.
    - `feat`: A new feature.
    - `docs`: Documentation only changes.
    - `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).
    - `refactor`: A code change that neither fixes a bug nor adds a feature.
    - `perf`: A code change that improves performance.
    - `test`: Adding missing tests or correcting existing tests.
    - `chore`: Changes to the build process or auxiliary tools and libraries such as documentation generation.
    - `build`: Changes that affect the build system or external dependencies.
  - **Example**: `"fix: typo in README.md"`

- [ ] **PR Message**: **_Delete this entire checklist_** and replace it with:

  - **Description**: A detailed description of the change.
  - **Dependencies**: Any dependencies required for this change.

- [ ] **Add tests and docs**: If you're adding a new integration, please include

  1. a test for the integration, preferably unit tests that do not rely on network access,
  2. an example notebook showing its use, located in the `examples` directory.

- [ ] **Lint and test**: Run `rye run pre-commit run --all-files` and `rye test` to ensure your changes pass all checks. See [contribution guidelines](https://xai4space.github.io/meteors/latest/how-to-guides/) for more.

- [ ] **Related Issue(s)**: If your PR fixes an issue, please link it in the PR description.

## Additional Guidelines

- Import optional dependencies within functions.
- Avoid adding dependencies to `pyproject.toml` files unless required for unit tests.
- Most PRs should not modify more than one package.
- Ensure changes are backwards compatible.

## Review Process

If no one reviews your PR within a few days, please `@-mention` one of the maintainers.
