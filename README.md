# 3LA PLDI Evaluation

[![Nightly (pipsqueak)](https://github.com/uwsampl/3la-pldi-2022-evaluation/actions/workflows/nightly-pipsqueak.yml/badge.svg)](https://github.com/uwsampl/3la-pldi-2022-evaluation/actions/workflows/nightly-pipsqueak.yml)
[![Nightly (GitHub runners)](https://github.com/uwsampl/3la-pldi-2022-evaluation/actions/workflows/nightly-github-runners.yml/badge.svg)](https://github.com/uwsampl/3la-pldi-2022-evaluation/actions/workflows/nightly-github-runners.yml)

In accordance with
  [*lex Lyubomiricus,*](https://homes.cs.washington.edu/~sslyu/lex.html)
  this repo
  is the single source of truth
  for the UW PLSE portion
  of the 3LA project's
  PLDI push.

[`run.sh`](run.sh)
  should be a readable script
  which runs
  all components of the evaluation.
All components
  of the evaluation
  should be version-controlled
  (e.g. by pointing to specific commits
    using git submodules.)
The entire repository
  is built and tested
  nightly
  on GitHub's machines
  and our own.

## Build and Run

To build, you need to use Docker's BuildKit option, so use a command like the following:
```
DOCKER_BUILDKIT=1 docker build . -t 3la-pldi-2022-evaluation --build-arg tvm_build_threads=32
```
