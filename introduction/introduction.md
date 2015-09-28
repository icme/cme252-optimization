---
title: Introduction to (Mathematical) Optimization
author: |
  | Nick Henderson, AJ Friend
  | Stanford University
date: August 18, 2015
---

## Optimization

Optimization

:   finding a best (or good enough) choice among the set of options for a certain
    objective subject to a set of constraints

## Mathematical optimization

Mathematical optimization problem has form
$$
\begin{array}{ll} \mbox{minimize} & f_0(x)\\
\mbox{subject to} & f_i(x) \leq 0, \quad i=1,\ldots,m
\end{array}
$$

* $x\in \reals^n$ is \textbf{decision variable} (to be found)

* $f_0$ is objective function; $f_i$ are constraint functions

* problem data are hidden inside $f_0,\ldots, f_m$

* variations: add equality constraints, maximize a utility function,
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

* heuristics required, hand-tuning, luck, babysitting

. . .

But...

* we can do a lot by restricting to convex models

* we have good computational tools

    * modeling languages (CVX, CVXPY, JuMP, AMPL, GAMS) to write problems down

    * solvers (IPOPT, SNOPT, Gurobi, CPLEX, Sedumi, SDPT3, ...) to obtain solutions

## Example: The Raptor Problem

See other slides

## Optimization in one variable

$$
\begin{array}{ll}
\mbox{minimize} & f(x) \in C^2:\reals \to \reals
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

. . .

* When $f(x)$ is twice continuously differentiable, then local optimization
  involves finding a point $x^*$ such that $f'(x^*)=0$ and $f''(x^*)>0$

## Optimization in one variable: axis
\includegraphics[width=\textwidth]{fig/graph-sequence-1.pdf}

## Optimization in one variable: definitions
\includegraphics[width=\textwidth]{fig/graph-sequence-2.pdf}

## Optimization in one variable: example objective function
\includegraphics[width=\textwidth]{fig/graph-sequence-3.pdf}

## Optimization in one variable: critical points, $f'(x) = 0$
\includegraphics[width=\textwidth]{fig/graph-sequence-4.pdf}

## Optimization in one variable: local optima
\includegraphics[width=\textwidth]{fig/graph-sequence-5.pdf}

## Optimization in one variable: local optima, $f''(x) = ?$
\includegraphics[width=\textwidth]{fig/graph-sequence-6.pdf}

## Optimization in one variable: unbounded below
\includegraphics[width=\textwidth]{fig/graph-sequence-7.pdf}

## Optimization in one variable: saddle point, $f'(x)=0$ and $f''(x) = 0$
\includegraphics[width=\textwidth]{fig/graph-sequence-8.pdf}

## Optimization in one variable: convex objective
\includegraphics[width=\textwidth]{fig/graph-sequence-9.pdf}

## Optimization in one variable: key definitions

* *variable*: a number representing the unknown or parameter we desire to find,
  usually $x$
* *objective*: a scalar mathematical function we want to maximize or minimize,
  designated $f(x)$
* *domain*: space for input variable $x$, we consider real numbers in this
  discussion
* *range*: space for output of objective function $f(x)$, a single real number
* *critical point*: $f'(x) = 0$
* *local minimizer*: $f'(x) = 0$ and $f''(x) > 0$
* *local maximizer*: $f'(x) = 0$ and $f''(x) < 0$
* *saddle point*: $f'(x) = 0$ and $f''(x) = 0$
* *global minimizer*: $x^*$ such that $f(x^*) \le f(x)$ for all $x$ in domain

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

. . .

* Technical algorithm property: **_convergence to solution_**
$$
|x_{k+1} - x_k| \to 0\ \text{if and only if}\ f'(x_k) \to 0
\ \text{and}\ \lim_{k\to\infty} f''(x_k) \ge 0
$$

## Optimization in many variables

$$
\begin{array}{ll}
\mbox{minimize} & f(x) \in C^2:\reals^n \to \reals
\end{array}
$$

* $x$ is an $n$-dimensional vector of real variables

* $f(x)$ is the objective function (twice continuously differentiable)

    * First derivative or gradient of $f$ is written $\nabla f(x)$

    * Second derivative or Hessian of $f$ is written $\nabla^2 f(x)$

* We are looking for a point $x^*$ such that $\nabla f(x)=0$ and
  $\nabla^2 f(x) \succeq 0$.  Note that this is a *local* optimizer

    * $\nabla^2 f(x) \succeq 0$ means that all the eigenvalues of $\nabla^2
      f(x)$ are non-negative


## The gradient $\nabla f(x)$ in 2 variables

Vector of variables:

$$
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

Gradient of $f$:

$$
\nabla f(x) = \begin{bmatrix}
  \frac{\partial f}{\partial x_1} \\[.5em]
  \frac{\partial f}{\partial x_2}
\end{bmatrix}
$$

## The Hessian $\nabla^2 f(x)$ in 2 variables

$$
\nabla^2 f(x) = \begin{bmatrix}
  \frac{\partial^2 f}{\partial x_1^2}            && \frac{\partial^2 f}{\partial x_1 \partial x_2}\\[1em]
  \frac{\partial^2 f}{\partial x_2 \partial x_1} && \frac{\partial^2 f}{\partial x_2^2}
\end{bmatrix}
$$

## Let's look at an example

The Rosenbrock function:

$$
f(x,y) = \left(1-x\right)^2 + 100\left(y-x^2\right)^2
$$

## Rosenbrock contours

\centering
\includegraphics[width=.5\textwidth]{code/rosen-contour.pdf}

## Basic optimization algorithm

\centering
\includegraphics[width=.8\textwidth]{fig/basic-algo.pdf}

## Line search algorithms

1. compute a search direction $p_k$

    * for minimization, $p_k$ must be a descent direction, that is $p_k^T g_k < 0$

2. select a step length $\alpha_k$ along $p_k$ such that
   $f(x_k + \alpha_k p_k) < f(x_k)$

    * (we need more technical requirements here)

3. update the guess $x_{k+1}\gets x_k + \alpha_k p_k$

## Example line search algorithms

Algorithm: $$ x_{k+1}\gets x_k + \alpha_k p_k $$

Gradient descent: $$ p_k = -g_k = -\nabla f(x_k)$$

Modified Newton's method:
$$p_k = -(H_k+\lambda_k I)^{-1} g_k = -(\nabla^2 f(x_k) +\lambda_k I)^{-1} \nabla f(x_k) $$

## Step length selection: backtracking

Goal: given $p_k$ find $\alpha$ such that $f(x_k + \alpha p_k) < f(x_k)$.

Procedure: start with initial guess $\alpha > 0$ (use $\alpha=1$ for Newton's method)

1. if $f(x_k + \alpha p_k) < f(x_k)$, then return $\alpha$, otherwise continue

2. decrease $\alpha$ by some factor $0 < \delta < 1$: $\alpha \gets
   \delta\alpha$

3. repeat

## Optimization on Rosenbrock function

\centering
\includegraphics[width=\textwidth]{code/gd-nm-iter.pdf}

## Optimization on Rosenbrock function

\centering
\includegraphics[width=\textwidth]{code/gd-nm-iter-2.pdf}

## Optimization on Rosenbrock function

\centering
\includegraphics[width=\textwidth]{code/rosen-conv.pdf}


## Considerations in selecting optimization algorithms

- Computational cost/scale of objective function
- Computational cost of linear algebra associated with optimization algorithm
- Accuracy requirement in your application

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

## Domain
\includegraphics[width=\textwidth]{fig/domain.pdf}

## Domain

Solvers for constrained optimization typically rely on a solver for
unconstrained optimization. Consider a mathematical optimization problem with
equality constraints:

$$
\begin{array}{ll} \mbox{minimize} & f_0(x)\\
\mbox{subject to} & f_i(x) = 0, \quad i=1,\ldots,m
\end{array}
$$

This can be turned into an unconstrained problem and approximately solved using
the *penalty method*:

$$
\begin{array}{ll} \mbox{minimize} & f_0(x) + \mu \sum_{i=1}^m (f_i(x))^2
\end{array}
$$

Typically need to solve a sequence of problems increasing $\mu$.  This is a very
common "engineering" technique.

## Model

Convex optimization:

- local minimizers are global
- useful theory of convexity
- effective algorithms and available software that provide: global solutions,
  polynomial complexity, and algorithms that scale
- convenient **language** to discuss problems
- expressive: **lots of applications**
- linear programming is an important sub class of convex optimization

Non-convex optimization:

- in general, no guarantee that minimizers are global
- solvers often use convex optimization as a sub-routine
- modeling tools are more difficult to use
- solution process may require expert guidance or tweaking

## This workshop

This workshop will primarily cover the **bold** topics:

* Variable type: {**continuous**, discrete}

* Domain: {**unconstrained**, **constrained**}

* Model: {**convex**, non-convex}

## Summary

* Mathematical optimization is an important and useful tool in science,
  engineering, and industry

* The optimization community has produced a large set of good tools to solve
  problems

    * there are a mix of open-source and commercial packages

* Art: mapping your problem into a mathematical model that can be attacked using
  an existing tool
