# Continuous integration

GitHub Actions configuration.

The workflow in `workflows/tests.yml` installs the dependencies, byte-compiles every
module, runs the test suite and then runs the end-to-end smoke test on synthetic
data, on Python 3.10 and 3.12.

This is the file that makes the rest of the repository believable: a green badge
means someone other than the author has run this code, on a machine that is not
the author's laptop, with no 120 MB download.

_Part of **Olist Delivery Performance Analytics**. Project overview: [`../README.md`](../README.md)._
