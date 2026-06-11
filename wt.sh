#!/usr/bin/env bash
# One worktree per branch, so parallel sessions/agents never share a checkout.
#   ./wt.sh <branch> [uv sync args]
#                          create (or re-open) ../catan-engine.wt/<branch>, synced;
#                          extra args go to uv sync (e.g. --extra cuda)
#   ./wt.sh ls             list worktrees
#   ./wt.sh rm <branch>    remove the worktree when merged (refuses if dirty; the branch is kept)
# New branches start from local main. Each worktree gets its own venv; the JAX
# compilation cache (~/.cache/jax-catan) is shared, so warm compiles carry across.
# Git hooks are shared too (.git/hooks serves every worktree) — never run
# `pre-commit install` from a worktree, or the hook shebang points at a venv
# that dies with `wt.sh rm` and commits break everywhere.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
wt_base="$(dirname "$root")/$(basename "$root").wt"

case "${1:-}" in
ls)
	git worktree list
	;;
rm)
	git worktree remove "$wt_base/${2:?usage: ./wt.sh rm <branch>}"
	;;
"")
	echo "usage: ./wt.sh <branch> | ls | rm <branch>" >&2
	exit 2
	;;
*)
	branch="$1"
	shift
	dir="$wt_base/$branch"
	if git show-ref --verify --quiet "refs/heads/$branch"; then
		git worktree add "$dir" "$branch"
	else
		git worktree add "$dir" -b "$branch" main
	fi
	(cd "$dir" && uv sync "$@")
	echo "$dir"
	;;
esac
