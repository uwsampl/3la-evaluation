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
It borrows heavily from [3la-integrate](https://github.com/PrincetonUniversity/3la-integrate).

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

To build, 
  you need to use Docker's BuildKit option, 
  so you need to set the `DOCKER_BUILDKIT` flag to 1. 
You also need 
  to have an ssh agent running 
  and give the `--ssh` flag to access our private repos.
The below commands should work assuming you have the appropriate ssh credentials:
```
eval `ssh-agent -s`
ssh-add
DOCKER_BUILDKIT=1 docker build . -t 3la-pldi-2022-evaluation --ssh default
```
