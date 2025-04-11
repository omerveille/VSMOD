# Contributing to this project
<!-- TOC -->
- [Contributing to this project](#contributing-to-this-project)
  - [What can be considered as a contribution ❓](#what-can-be-considered-as-a-contribution-)
  - [For developers 🧑‍💻](#for-developers-)
    - [Contributing process 📋](#contributing-process-)
    - [Code formatting 🗃️](#code-formatting-️)
    - [Setting up pre-commit 🏗️](#setting-up-pre-commit-️)
    - [Enforce pre-commit to run 🏃](#enforce-pre-commit-to-run-)
<!-- TOC -->

## What can be considered as a contribution ❓

Absolutely anything!
For example, you may want to:

- Give us a feedback on the plugin
- Report a bug
- Submit a feature
- Simply ask a question for clarification

Write us a ticket using the [Issues](https://github.com/omerveille/VSMOD/issues).

If you want to directly contribute with code, please refer to the next sections.

## For developers 🧑‍💻

We welcome all kinds of contributions — from small documentation fixes to major changes that might start discussion.

As long as your code is clearly commented, uses [type hints](https://docs.python.org/3/library/typing.html) and pass the [tests](https://slicer.readthedocs.io/en/latest/developer_guide/modules/selftests.html) it should be good to go.

### Contributing process 📋

In order to contribute to this project follow these steps :

- [Fork and clone](https://help.github.com/articles/fork-a-repo/) the repository.
- Create a branch.
- [Push](https://help.github.com/articles/pushing-to-a-remote/) the branch to your GitHub fork.
- Create a [Pull Request](https://github.com/omerveille/VSMOD/pulls).

As for commit message, we follow these conventions for commit messages and kindly ask you to do the same:

- fix: when you patch a functionality that works another way as intended
- update: when you update something existing by refactoring code, updating the documentation, adding tests (the functionality should remain the same)
- feature: when your code allows user to do something new they could not do before

Feel free to write detailed commit messages — even full paragraphs if needed. We prefer clear, informative messages over short, vague ones. That said, avoid writing a novel when a sentence or two would do!

### Code formatting 🗃️

The code formatting is managed by [pre-commit](https://pre-commit.com/) a handy git hook managing tool. This tool allows you to run pre-commit scripts to format code and check for basic mistakes in code. You **must** install it before committing, it ensures a stable code formatting across the whole project. Feel free to check the `.pre-commit-config.yaml` if you wish to know what is exactly happening when it runs.

### Setting up pre-commit 🏗️

Make sure you have Python and pip installed on your machine.
Then you can install pre-commit on this repo using these commands:

```shell
# Install pre-commit
pip install pre-commit
# Prepare the hooks for this repository
pre-commit install
```

After that, all <ins>added and tracked</ins> Python and Markdown files will be automatically formatted when committed.

### Enforce pre-commit to run 🏃

You may want to manually run the code formatting pipeline, here is the command :

```shell
pre-commit run --all-files
```
