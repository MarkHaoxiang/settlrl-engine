# Getting started

## Requirements

- Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18 (only for `settlrl-app`)

## Install

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/).
Install everything from the repo root:

```bash
uv sync
```

Each package can also be targeted directly with `uv run --package <name> …`.

## Building these docs

The docs site is an opt-in dependency group:

```bash
uv run --group docs mkdocs serve   # live-reload preview at http://127.0.0.1:8000
uv run --group docs mkdocs build   # render the static site into ./site
```

The **Reference** section is generated from the workspace sources by
[mkdocstrings](https://mkdocstrings.github.io/). To document something new, add a
`::: dotted.module.path` directive to the relevant `docs/reference/*.md` page;
write narrative pages as plain Markdown under `docs/`.

## Hosting

The site is served as static files (the `mkdocs build` output) by the VPS's
Caddy. Build, sync the output, and let Caddy serve it under a path:

```bash
uv run --group docs mkdocs build        # writes ./site
rsync -a --delete site/ <vps>:/srv/settlrl-engine/site/
```

```caddy
# inside the existing www.markhaoxiang.com site block
handle_path /settlrl-engine/* {
    root * /srv/settlrl-engine/site
    file_server
}
```

`handle_path` strips the prefix so files resolve against the build root; pages
use relative links, so the docs work under any path. `site_url` in `mkdocs.yml`
is set to the served URL (canonical links / sitemap only).
