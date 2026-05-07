# Translating the documentation

QTrade docs are written in English in `docs/`. Other languages live as
gettext `.po` translation files under `docs/locale/<lang>/LC_MESSAGES/`.
The English source is the single source of truth — edits to it
automatically generate fresh translation slots in the `.po` files.

Currently translated:
- `zh_CN` (Simplified Chinese) — partial. Untranslated strings fall
  back to the English source.

## Building locally

```bash
pip install -r docs/requirements.txt
pip install -e .

# English
sphinx-build -b html docs docs/_build/html_en

# Chinese
sphinx-build -b html -D language=zh_CN docs docs/_build/html_zh
```

The `sphinx-build` is identical apart from `-D language=zh_CN`. CI
builds both and deploys them to `/<base>/en/` and `/<base>/zh/` on
GitHub Pages.

## Adding or updating a translation

After editing any markdown file in `docs/`, regenerate the gettext
templates and merge them into the existing `.po` files:

```bash
# 1. Extract strings from the latest English source
sphinx-build -b gettext docs docs/_gettext

# 2. Merge into existing translations (creates new entries, preserves old)
sphinx-intl update -p docs/_gettext -d docs/locale -l zh_CN

# 3. Open the updated .po files and translate the new entries
#    (search for `msgstr ""` to find untranslated strings)
$EDITOR docs/locale/zh_CN/LC_MESSAGES/guide/concepts.po

# 4. Optional: drop the gettext templates (regenerate any time)
rm -rf docs/_gettext
```

A `.po` entry looks like:

```
#: ../guide/concepts.md:42
msgid "Some English source paragraph."
msgstr "对应的中文翻译。"
```

Translate by filling in `msgstr`. Empty `msgstr` means "fall back to
the English source on this paragraph" — leave them empty rather than
deleting; that way `sphinx-intl update` knows what's still pending.

## Adding a new language

```bash
sphinx-build -b gettext docs docs/_gettext
sphinx-intl update -p docs/_gettext -d docs/locale -l fr_FR    # for French
```

Then update `.github/workflows/docs-build-dev.yml` to add a new build
+ deploy step for that locale, and the language switcher
(`docs/_static/lang_switcher.js`) to include it in the cycle.
