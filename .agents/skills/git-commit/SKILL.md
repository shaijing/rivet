---
name: git-commit
description: Use when generating Git commit messages for this project, when the user asks to "commit", "write a commit message", or asks the assistant to summarize staged/unstaged changes into a Conventional Commits subject and body.
---

# Git Commit Message

Generate Conventional Commits messages for this project.

## Format

```text
type(scope?): subject

body?

footer?
```

## Type

Pick the closest match. Bug fix → `fix`. New user-visible capability → `feat`. Internal cleanup without behavior change → `refactor`.

| type | use for |
|------|---------|
| feat | new user-visible capability |
| fix | bug fix |
| refactor | internal cleanup, no behavior change |
| perf | speed/memory improvement |
| docs | docs only |
| test | tests only |
| build | build or dependencies |
| ci | CI configuration |
| chore | anything else |

`scope` is optional (e.g. `api`, `parser`, `cache`, `core`, `exp`, `isom`). Omit parens when none fits.

## Subject

- Imperative, ≤50 chars, no trailing period
- Describe **what changed**, not how

## Body

Optional. Explain *why*, not what. Skip when the subject already says it.

## Breaking change

Use `!` after type/scope, or a `BREAKING CHANGE:` footer. Required when removing/renaming public APIs, changing external behavior, or breaking config format.

## Multi-file changes

Pick one primary type. Do not stack multiple types in one commit.

## Style

Concise, direct, implementation-focused. Avoid vague wording (`fix bug`, `update code`, `modify some things`).

## Examples

```text
fix(parser): fix JSON empty object parsing error

feat(cache): add disk caching support

Add local disk caching to reduce redundant network requests.
Cache entries are retained for 24 hours by default.

feat(api)!: refactor REST API

Standardize the response format across all endpoints.

BREAKING CHANGE: remove the v1 API; please migrate to v2.
```