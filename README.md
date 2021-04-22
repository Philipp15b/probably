# Probably

---

Visit the **[Probably website](https://philipp15b.github.io/probably/)** for API documentation and how to use the command-line interface!

---

**Probably** is a Python package for parsing, type-checking, and analyzing probabilistic programs written in the pGCL language.

Features:

* pGCL language support: Constants, variables, static types (boolean, (bounded) natural numbers, floats), literals, unary and binary operators, while loops, conditional if statements, assignments, probabilistic choices.
* Weakest pre-expectation calculation for loop-free and linear pGCL programs.
    * Translation from general expectations to linear expectations and expectations in summation normal form.
* Program AST traversal and modification using iterators over mutable references.
* General algorithm for variable substitution in program expressions with substitution expressions.
* Detailed documentation, lots of examples, extensive automated tests.

## Usage

Visit the **[Probably website](https://philipp15b.github.io/probably/)** for API documentation and how to use the command-line interface.

Use command line:
```
$ poetry install; poetry shell
$ poetry run probably example.pgcl
Program source:
bool f;
nat c;
while (c < 10 & f) {
     {c := c+1} [0.8] {f:=true}
}

[...]

Program is linear.
Summation Normal Expectation: [(c < 10) & f] * 0.8 * (Y)[c/c + 1] + [(c < 10) & f] * (1.0 - 0.8) * (Y)[f/true] + [not ((c < 10) & f)]
```

## Installation

We use [poetry](https://github.com/python-poetry/poetry) for dependency management.
See [here](https://python-poetry.org/docs/) for installation instructions for poetry.
<small>You may need to adjust the python dependency in `pyproject.toml` to `python = "^3.6"` and call `poetry env use python3` for some reason.</small>

**Add as a dependency:**
You can add this project as a dependency with ([more information](https://python-poetry.org/docs/dependency-specification/#git-dependencies)):
```
poetry add git+https://github.com/Philipp15b/probably.git
```

**Work on probably itself:**
Just execute `poetry install` to hack on this project locally.
You can use [`path` dependencies](https://python-poetry.org/docs/dependency-specification/#path-dependencies) so you can modify `probably` locally as a dependency of another project.

## Development

**Docs:** Build the documentation with `make docs`: It'll be in `docs/build/html`.

**Typechecking:** Run `mypy` with `make mypy`.

**Tests:** Run tests with `make test` ([`pytest`](https://docs.pytest.org/en/latest/)).
Tests also produce a coverage report. It can be found in the generated `htmlcov` directory.

**Lint:** Run `pylint` with `make lint`.

**Formatting:** We use the [`yapf`](https://github.com/google/yapf) formatter: `yapf --recursive -i probably/` and `isort probably/`.
