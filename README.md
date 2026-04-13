> [!WARNING]
> This package has been migrated to the [TeamTomo monorepo](https://github.com/teamtomo/teamtomo).
> Future development, bug fixes, and releases will happen there.
> This repository is archived and no longer maintained.
> This package is still published to and installable from the same PyPI project, but development installations should be made from the monorepo.

# torch-ctf

[![License](https://img.shields.io/pypi/l/torch-ctf.svg?color=green)](https://github.com/jdickerson95/torch-ctf/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-ctf.svg?color=green)](https://pypi.org/project/torch-ctf)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-ctf.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/torch-ctf/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/torch-ctf/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/torch-ctf/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/torch-ctf)

CTF calculation for cryoEM in torch

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork jdickerson95/torch-ctf --clone
# or just
# gh repo clone jdickerson95/torch-ctf
cd torch-ctf
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
