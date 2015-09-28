% CME 252: Introduction to Optimization
% AJ Friend \
  ICME, Stanford University

# Introduction
## Course Goals
- get students using CVXPY to solve real (convex) optimization
problems as quickly as possible
- teach just enough theory to do so
- focus on **modeling**: **what** you want to solve instead of **how** to solve it
- convex optimization as a starting point to consider
more general optimization problems

## Audience
- anyone interested in using optimization in their work
- no background in optimization is necessary
- do need to be comfortable with
    - linear algebra
    - basic programming (any language)

## Logistics
- course website at [ajfriend.github.io/cme252](http://ajfriend.github.io/cme252/)
    - announcements
    - homework
    - lecture materials
- schedule
    - MW 3:30-4:50 in McCullough 115
    - 8 sessions
    - office hours TBD (Thursday?)
- Piazza [http://piazza.com/stanford/fall2015/cme252](http://piazza.com/stanford/fall2015/cme252)

## Pyhton/CVXPY
- we'll use Python and CVXPY to solve optimization problems
- need a working Python distribution with
    - numpy
    - scipy
    - matplotlib
    - CVXPY
- example code given in Jupyter (IPython) notebooks
- HW0 to get you set up (out before Wednesday)
- additional help Wednesday, office hours, Piazza

## Homework
- 1 assignment per week, due on Friday
- submit python script solving a few optimization problems
- HW1 released by this Friday, due next Friday
- HW0 (not graded)
    - Python/CVXPY setup
    - Jupyter (IPython) notebooks
    - homework submission

## Topic Outline (tentative)
- types of optimization
- convex sets and functions
- convex optimization and modeling
- regression, least-squares, curve-fitting and variants
- in-depth examples from various fields (SVM, logistic regression,...)
- basics of gradient descent
- non-convex problems (and convex approaches)

vary topics based on student interest

## Won't Cover
- other optimization classes (global optimization, integer programming,...)
- within convex optimization:
    - optimality conditions (maybe a *little*)
    - duality and Lagrange multipliers
    - in-depth algorithms
    - fancy convex sets and functions (SDP, perspective functions,...)

## Questions?

# Optimization Overview
## Optimization

Optimization

:   given an objective function, finding a best (or good enough) choice among a set of (possibly constrained) options

## Mathematical optimization

Mathematical optimization problem has form
$$
\begin{array}{ll} \mbox{minimize} & f(x)\\
\mbox{subject to} & x \in C
\end{array}
$$

* $x\in \reals^n$ is \textbf{decision variable} (to be found)
* $C$ is a set describing **acceptable** points
* $f$ is objective function (choose best acceptable point)
* problem data are hidden inside $f$ and $C$
* variations: different ways to represent problem, maximize a utility function,
  satisfaction (feasibility), optimal trade off, and more

## The good news

__Everything__ is an optimization problem

* *choose parameters* in model to fit data
  (minimize misfit or error on observed data)
* *optimize actions* (minimize cost or maximize profit)
* *allocate resources* over time
  (minimize cost, power; maximize utility)
* *engineering design*
  (trade off weight, power, speed, performance, lifetime)


## The bad news

In full generality, optimization problems can be quite difficult

* generally NP-hard
* local vs. global minimizers
* heuristics required, hand-tuning, luck, babysitting

. . .

But...

* we can do a lot by restricting to convex models
* local minimizers are global
* we have good computational tools

    * modeling languages (CVX, CVXPY, JuMP, AMPL, GAMS) to write problems down

    * good solver software to obtain solutions

## Optimization in one variable

$$
\begin{array}{ll}
\mbox{minimize} & f(x) \in C:\reals \to \reals
\end{array}
$$

. . .

* $x$ is a real variable

. . .

* $f(x)$ is the objective function, which returns a single real number

. . .

* Local optimization: look for a point $x^*$ such that $f(x^*) \le f(x)$ for all
  points $x$ near $x^*$

. . .

* Global optimization: look for a point $x^*$ such that $f(x^*) \le f(x)$ for all
points $x$ in domain of interest

## Optimization in one variable: example objective function
\includegraphics[width=\textwidth]{fig/graph-sequence-3.pdf}

## Optimization in one variable: critical points, $f'(x) = 0$
\includegraphics[width=\textwidth]{fig/graph-sequence-4.pdf}

## Optimization in one variable: local optima, $f''(x) = ?$
\includegraphics[width=\textwidth]{fig/graph-sequence-6.pdf}

## Optimization in one variable: unbounded below
\includegraphics[width=\textwidth]{fig/graph-sequence-7.pdf}

## Optimization in one variable: convex objective
\includegraphics[width=\textwidth]{fig/graph-sequence-9.pdf}


## Optimization in one variable: algorithm basics
\includegraphics[width=\textwidth]{fig/graph-sequence-10.pdf}

## Optimization in one variable: algorithm basics
\includegraphics[width=\textwidth]{fig/graph-sequence-11.pdf}

## Optimization in one variable: algorithm basics
\includegraphics[width=\textwidth]{fig/graph-sequence-12.pdf}

## Optimization in one variable: algorithm basics
\includegraphics[width=\textwidth]{fig/graph-sequence-13.pdf}

## Optimization in one variable: algorithm basics
\includegraphics[width=\textwidth]{fig/graph-sequence-14.pdf}

## Optimization in one variable: algorithm basics

. . .

* Start with an initial guess $x_0$

. . .

* Goal: generate sequence that converges to solution
$$
x_0, x_1, x_2, x_3, \dots \to x^*
$$

. . .

* Notation for sequence and convergence: $\{x_k\} \to x^*$

. . .

* Key algorithm property: **_descent condition_**
$$
f(x_{k+1}) < f(x_k)
$$

* the simplest algorithm works! (for local minimizers)

. . .

## Why Convexity?
Convex optimization:

- local minimizers are global
- useful theory of convexity
- effective algorithms and available software that provide: global solutions,
  polynomial complexity, and algorithms that scale
- convenient **language** to discuss problems
- expressive: **lots of applications**

Non-convex optimization:

- in general, no guarantee that minimizers are global
- solvers often use convex optimization as a sub-routine
- modeling tools are more difficult to use
- solution process may require expert guidance or tweaking

## Classes of mathematical optimization problems

* There are many (!) classes of mathematical optimization problems (and
  associated solvers)

* The primary problem features are:

    * Variable type: {continuous, discrete}

    * Domain: {unconstrained, constrained}

    * Model: {convex, non-convex}

## Variables

Continuous variables take real numbers as values (within limits):
$$
x \in \reals
$$

Discrete variables typically take integers as values:
$$
x \in \{0, 1, 2, 3, \dots \}
$$

Boolean or binary variables are a special case of this:
$$
x \in \{0, 1\}
$$

Problems with discrete variables are generally harder than those with continuous
variables.

## Variables

Example of continuous variables:

* maximum likelihood estimate of the mean
* parameters in a linear model
* asset allocation in mean-variance portfolio optimization
* position in a standard coordinate system
* speed (in, say, a model to minimize fuel consumption)

Example of discrete variables:

* A $\{0, 1\}$ selector for facility location.  Say variable $x_{ij} = 1$ if and
  only if resource $i$ is placed in location $j$ and zero otherwise.

* An integer representing the number of people allocated to a task. It would be
  unwise and perhaps illegal to allocate half a person.

## Domain

Unconstrained mathematical optimization problems only require an objective function:

$$
\begin{array}{ll}
\mbox{minimize} & f_0(x)
\end{array}
$$

Constrained optimization problems limit the domain with equations or inequalities:

$$
\begin{array}{ll} \mbox{minimize} & f_0(x)\\
\mbox{subject to} & f_i(x) \leq 0, \quad i=1,\ldots,m
\end{array}
$$

Solvers for constrained optimization typically rely on a solver for
unconstrained optimization.

## Domain
\includegraphics[width=\textwidth]{fig/domain.pdf}

## This class

This class will primarily cover the **bold** topics:

* Variable type: {**continuous**, discrete}

* Domain: {**unconstrained**, **constrained**}

* Model: {**convex**, non-convex}

## Summary

* Mathematical optimization is an important and useful tool in science,
  engineering, and industry

* The optimization community has produced a large set of good tools to solve
  problems

    * there are a mix of open-source and commercial packages

* Art: mapping your problem into a mathematical model that can be attacked using an existing tool

* Next class: jump into theory and examples of convex sets and functions

# CVXPY/Jupyter Example
## CVXPY/Jupyter Example