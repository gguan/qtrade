# Releasing

This project publishes to PyPI via GitHub Actions using PyPI's
[trusted publishing](https://docs.pypi.org/trusted-publishers/) (no API
tokens). The workflow is at [.github/workflows/release.yml](.github/workflows/release.yml).

## One-time setup (PyPI side)

These steps need to be performed once by a project maintainer with PyPI
access. Skip if it's already done.

1. Sign in to https://pypi.org and open the project at
   https://pypi.org/manage/project/qtrade-lib/settings/publishing/.
2. Click **Add a new pending publisher** (or **Add a new publisher** if
   the workflow has already run once).
3. Fill in:
   - **Owner**: `gguan`
   - **Repository name**: `qtrade`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
4. Repeat for [TestPyPI](https://test.pypi.org/manage/project/qtrade-lib/settings/publishing/)
   if you want to test releases without polluting PyPI history. Use the
   same fields, just with environment name `testpypi`.

## One-time setup (GitHub side)

Create the deployment environments referenced by the workflow so they
can have protection rules later (manual approval, branch restriction):

1. Repo → **Settings** → **Environments** → **New environment** →
   `pypi`. Optional: require manual approval for first-time use.
2. Repeat with environment name `testpypi`.

## Cutting a release

Releases are triggered by pushing a tag matching `v*`. The version in
the tag must match the version in `pyproject.toml` and in
`docs/conf.py`; the workflow asserts this and aborts otherwise.

```bash
# 1. Update versions in pyproject.toml and docs/conf.py.
# 2. Move the unreleased section in CHANGELOG.md under a new
#    "## [X.Y.Z]" heading. Update the link references at the bottom.
# 3. Commit, open a PR, get it merged.
# 4. After merge, tag the merge commit on main:
git checkout main
git pull
git tag v0.4.1
git push origin v0.4.1
```

That's it. The workflow will:

1. Run the test suite on Python 3.10 / 3.11 / 3.12.
2. Build the wheel and sdist with `python -m build`.
3. Verify the built version matches the tag.
4. Publish to PyPI via trusted publishing.
5. Create a GitHub Release with the matching CHANGELOG section.

## Test releases

Manually dispatch the **Release to PyPI** workflow from the Actions
tab and check the **Publish to TestPyPI** box. The build skips the
PyPI publish step and pushes to https://test.pypi.org instead.

Install from TestPyPI to verify:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            qtrade-lib
```

(The `--extra-index-url` is needed so dependencies still resolve
from regular PyPI.)
