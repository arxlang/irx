# Installation

## Stable release

To install IRx, run this command in your terminal:

```bash
$ pip install pyirx
```

This is the preferred method to install IRx, as it will always install the most
recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this
[Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/)
can guide you through the process.

## From sources

The sources for IRx can be downloaded from the
[Github repo](https://github.com/arxlang/irx).

You can either clone the public repository:

```bash
$ git clone https://github.com/arxlang/irx
```

Or download the [tarball](https://github.com/arxlang/irx/tarball/main):

```bash
$ curl -OJL https://github.com/arxlang/irx/tarball/main
```

Once you have a copy of the source, you can install it with:

```bash
$ poetry install
```

> Note for contributors
>
> If you are setting up a development environment to run tests or contribute code, please follow the steps in the Contributing guide instead. The development workflow requires creating the Conda environment first, then installing Poetry dependencies:
>
> ```bash
> mamba env create --file conda/dev.yaml
> conda activate irx
> poetry install
> ```
>
> See the full instructions at: https://irx.arxlang.org/contributing/
