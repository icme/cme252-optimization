# ICME Optimization Short Course

Authors: [AJ Friend](mailto:ajfriend@stanford.edu) and [Nick Henderson](mailto:nwh@stanford.edu)

## Course Description
This course introduces mathematical optimization and modeling, with a focus on convex optimization. We will use convexity as a starting point from which to consider some nonconvex problem types. The course will have a practical focus, with participants formulating and solving optimization problems early and often using Python and the open source modeling framework CVXPY. By introducing common models from machine learning and other fields, the course aims to make students comfortable with optimization modeling so that they may use it for rapid prototyping and experimentation in their own work.

We'll cover simple but useful optimization algorithms such as gradient descent for when problems become too large for black-box solvers and cover majorization-maximization as a useful framework to view optimization algorithms for both convex and non-convex problems.

Time permitting, we will also discuss methods to solve nonconvex models for nonnegative matrix factorization, matrix completion, and neural networks. Students should be comfortable with linear algebra, differential multivariable calculus, and basic probability and statistics. Experience with Python will be helpful, but not required.

### Topics

- varieties of mathematical optimization
- convexity of functions and sets
- convex optimization modeling with CVXPY
- gradient descent and basic distributed optimization
- in-depth examples from machine learning, statistics and other fields
- applications of bi-convexity and non-convex gradient descent

## Lecture Outline
1. Introduction
2. Least-Squares
3. Convex Sets and Functions
4. Convex Sets and Functions
5. SVM/Logistic In-depth
6. Gradient Descent
7. Gradient Descent
8. Non-convex optimization: NNMF, non-linear least-squares?

## Reqirements

- [TeXLive](https://www.tug.org/texlive/) or [MacTeX](https://www.tug.org/mactex/)
- [Pandoc](http://pandoc.org/)
- [Poppler](http://poppler.freedesktop.org/)

Fedora 21 instructions:

```sh
$ sudo yum install texlive pandoc poppler-utils
```

Mac OS X instructions:

- Install [Homebrew](http://brew.sh/)
- Install [MacTeX](https://www.tug.org/mactex/)

```sh
$ brew install pandoc poppler
```

## GNU `make` targets

- `$ make`: generate `pdf` slides for each lecture
- `$ make clean`: clean up `pdf` and `tex` files for slides
- `$ make cleanall`: clean up all generated files
