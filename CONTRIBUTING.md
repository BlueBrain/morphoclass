# Contributing
Thank you for your interest in contributing to MorphoClass!

Before you open an issue or start working on a pull request, please make sure
to read the following guidelines.

1. [Creating Issues](#creating-issues)
1. [Creating Pull Requests](#creating-pull-requests)
    1. [Refer to an issue](#refer-to-an-issue)
    1. [Add unit test](#add-unit-tests)
    1. [Add type annotations](#add-type-annotations)
    1. [Update dependencies](#update-dependencies)
    1. [Ensure all CI tests pass](#ensure-all-ci-tests-pass)
    1. [Get reviews and approval](#get-reviews-and-approval)


## Creating Issues
Before you submit an issue, please search our [issue
tracker](https://github.com/BlueBrain/morphoclass/issues) to verify if any
similar issue was raised in the past. Even if you cannot find a direct answer
to your question, this can allow you to collect a list of related issues that
you can link in your new issue.

Once you have verified that no similar issues exist yet, feel free to open a
new issue. 


## Creating Pull Requests
If you wish to contribute to the code base, opening a pull request on GitHub is
the right thing to do!
 
Please read the following paragraphs to make sure that your work can be
considered for a potential integration in our source code. 

### Refer to an issue
In general, every pull request should refer to a specific issue. If you would
like to provide your contribution on a untracked issue, please create first an
issue as explained [here](#CreatingIssues) so that we can discuss the value of
the proposed contribution and its implementation.

### Add unit tests
Concerning CI tests, we are running various checks on linting, unit tests,
docs, and packaging. If you are adding or modifying a functionality in the
code, you are also expected to provide a unit test demonstrating and checking
the new behavior. 

### Add type annotations
We are gradually introducing type annotations to our code, and our CI performs
type checking with [mypy](https://mypy.readthedocs.io/en/stable/index.html). If
your PR introduces a new function or heavily modifies an existing function, you
should also add type annotations for such function.   

### Update dependencies
If your PR introduces a dependency on a new python package, this should be
added to both the `setup.py` and the `requirements.txt` files.

### Ensure all CI tests pass
We use GitHub Actions to run [our CI
tests](https://github.com/BlueBrain/morphoclass/actions?query=workflow%3A%22ci+testing%22).
Once you open a PR, the workflow that runs all CI tests is automatically
triggered. This workflow is also triggered every time a new commit is pushed to
that PR branch. If you want to push a commit without triggering the CI tests
(e.g. if a feature is still work in progress and you want to save time), your
commit message should include an appropriate label like `[skip ci]`, `[ci
skip]`, `[no ci]` etc. (see
[here](https://github.blog/changelog/2021-02-08-github-actions-skip-pull-request-and-push-workflows-with-skip-ci/)
for the complete list).

All CI tests must pass before the pull request can be considered for review.

### Get reviews and approval
Once you have satisfied all the previous points, feel free to open your pull
request!

We will get back to you as soon as possible with comments and feedback in the
format of pull request reviews. A positive review from one of the maintainers
is required for the pull request to be merged into the master.
