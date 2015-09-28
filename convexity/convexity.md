---
title: Convex Sets, Functions, and Problems
author: |
  | Nick Henderson, AJ Friend
  | Stanford University
---

# Set Notation
## Set Notation
- $\reals^n$: set of $n$-dimensional real vectors
- $x \in C$: the point $x$ is an element of set $C$
- $C \subseteq \reals^n$: $C$ is a **subset** of $\reals^n$, \ie,
elements of $C$ are $n$-vectors
- can describe set elements explicitly: $1 \in \lbrace 3, \mbox{"cat"}, 1 \rbrace$
- **set builder notation**
$$C = \lbrace x\ \vert\ P(x) \rbrace$$
gives the points for which property $P(x)$ is true
- $\reals^n_+ = \lbrace x\ \vert\ x_i \geq 0\ \mbox{for all}\ i\rbrace$: $n$-vectors with all nonnegative elements
- **set intersection**
$$C = \bigcap_{i=1}^N C_i$$
is the set of points which are simultaneously present in each $C_i$

# Convexity
## Convex Sets
- $C \subseteq \reals^n$ is **convex** if
$$t x + (1-t)y \in C$$
for any $x, y \in C$ and $0 \leq t \leq 1$
- that is, a set is convex if the line connecting **any** two
points in the set is entirely inside the set

## Convex Set
\centering
\includegraphics[width=0.5\textwidth]{fig/cvx_set.pdf}

## Nonconvex Set
\centering
\includegraphics[width=0.5\textwidth]{fig/non_cvx_set.pdf}

## Convex Functions
- $f: \reals^n \to \reals$ is **convex** if $\dom(f)$ (the domain of $f$)
is a convex set, and
$$f\left(tx + (1-t)y\right) \leq t f(x) + (1-t)f(y)$$
for any $x,y \in \dom(f)$ and $0 \leq t \leq 1$
- that is, convex functions are "bowl-shaped"; the line connecting any two points on the graph of the function stays above the graph
- $f$ is **concave** if $-f$ is **convex**

## Convex Function
\centering
\includegraphics[width=0.65\textwidth]{fig/cvx_func.pdf}

## Nonconvex Function
\centering
\includegraphics[width=0.65\textwidth]{fig/non_cvx_func.pdf}

## Convex Optimization Problem
- the optimization problem
$$
\begin{array}{ll}
\mbox{minimize} & f(x) \\
\mbox{subject to} & x \in C
\end{array}
$$
is **convex** if $f: \reals^n \to \reals$ is convex and $C \subseteq \reals^n$ is convex
- any **concave** optimization problem
$$
\begin{array}{ll}
\mbox{maximize} & g(x) \\
\mbox{subject to} & x \in C
\end{array}
$$
for **concave** $g$ and convex $C$ can be rewritten as a **convex** problem by
minimizing $-g$ instead


# Why Convexity?

## Minimizers
- no worries about local minimizers; local minimizers are global

\centering
\includegraphics[width=0.5\textwidth]{fig/local_global_min.pdf}


## Algorithms
- intuitive algorithms work: "just go down" leads you to the global minimum
- can't get stuck close to local minimizers
\note{Imagine yourself on a hilly landscape. If your goal is to get to the lowest point, you direct yourself "down", take a step, re-evaluate which way is down, and repeat. That gets you to a local minimizer, but you can't see the rest of the landscape, so you don't know if you're global. You would have to explore the whole space (exhaustive search). If someone tells you that your valley is convex, you have a guarantee that you're at the global minimizer.}
- lots of good, existing software to solve convex optimization problems
- morally, writing down a convex optimization problem is as good as having the (computational) solution (for problems that aren't too big!)

## Expressiveness
- convexity is a modeling constraint; most problems are **not** convex
- however, convex optimization is **very** expressive, with many applications:
    - machine learning
    - engineering design
    - finance
    - signal processing
- convex modeling tools like CVX (MATLAB) or CVXPY (Python) make it easier to describe convex problems

## Focus on Modeling
- modeling tools (CVX, CVXPY) and good solution algorithms let users (usually) focus on **what** their model should be instead of **how** to solve it
- coming up:
    - learn to manipulate **simple** convex sets and functions
    to **construct** more complicated convex models
    - theory of convex sets and functions
    - practical modeling with CVXPY

## Nonconvex Extensions
- even though most problems are not convex, convex optimization can still be useful
- approximate nonconvex problem with a convex model
- convex optimization can be used as a subroutine in a (heuristic) nonconvex solver:
    - locally approximate the problem as convex
    - solve local model
    - step to new point
    - re-approximate and repeat


# Convex Sets
## Examples
- empty set: $\emptyset$
- set containing a single point: $\lbrace x_0 \rbrace$ for $x_0 \in \reals^n$
- $\reals^n$
- positive orthant: $\reals^n_+ = \lbrace x \vert x_i \geq 0,\ \forall i \rbrace$

\centering
\includegraphics[width=0.3\textwidth]{fig/pos_orth.pdf}


## Hyperplanes and Halfspaces
- **hyperplane** $C = \lbrace x \vert a^Tx = b \rbrace$

\centering
\includegraphics[width=0.25\textwidth]{fig/hyperplane.pdf}

- **halfspace** $C = \lbrace x \vert a^Tx \geq b \rbrace$

\centering
\includegraphics[width=0.25\textwidth]{fig/halfspace.pdf}

## Norm Balls
- a norm $\|\cdot\|:\reals^n \to \reals$ is any function such that
    - $\|x\| \geq 0$, and $\|x\| = 0$ if and only if $x=0$
    - $\| tx\| = |t| \| x\|$ for $t \in \reals$
    - $\|x+y\| \leq \|x\| + \|y\|$
- $\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$
- $\|x\|_1 = \sum_{i=1}^n |x_i|$
- $\|x\|_\infty = \max_i |x_i|$
- **unit norm ball**, $\lbrace x \vert \| x \| \leq 1\rbrace$, is **convex** for any norm

## Norm Ball Proof
- let $C = \lbrace x \vert \| x \| \leq 1\rbrace$
- to check convexity, assume $x, y \in C$, and $0 \leq t \leq 1$
- then, \begin{align*}
\|t x + (1-t)y \| &\leq \|t x\| + \|(1-t)y \| \\
&= t \|x\| + (1-t) \|y\| \\
&\leq t + (1-t) \\
&= 1
\end{align*}
- so $t x + (1-t)y \in C$, showing convexity
- this proof is typical for showing convexity

## Intersection of Convex Sets
- the intersection of any number of convex sets is convex 
- **example**: polyhedron is the intersection of halfspaces
$$
\includegraphics[width=0.25\textwidth]{fig/poly.pdf}
$$
- rewrite $\bigcap_{i=1}^m \lbrace x \vert a_i^T x \leq b_i\rbrace$ as
$\lbrace x \vert Ax \leq b \rbrace$, where
$$
A = \begin{bmatrix}
a_1^T\\
\vdots\\
a_m^T
\end{bmatrix},\ 
b = \begin{bmatrix}
b_1^T\\
\vdots\\
b_m^T
\end{bmatrix}
$$
- $Ax \leq b$ is **componentwise** or **vector inequality**


## More Examples
- solutions to a linear equation $Ax = b$ forms a convex set (intersection of hyperplanes)
- probability simplex, $C = \lbrace x \vert x \geq 0, 1^Tx = 1\rbrace$ is convex (intersection of positive orthant and hyperplane)

## CVXPY for Convex Intersection
- use CVXPY to solve the **convex set intersection problem**
$$
\begin{array}{ll}
\mbox{minimize} & 0 \\
\mbox{subject to} & x \in C_1 \cup \cdots \cup C_m
\end{array}
$$
- set intersection given by list of constraints
- **example**: find a point in the intersection of two lines
\begin{align*}
2x + y = 4\\
-x + 5y = 0
\end{align*}

## CVXPY code
```
from cvxpy import *

x = Variable()
y = Variable()

obj = Minimize(0)
constr = [2*x + y == 4,
          -x + 5*y == 0]

Problem(obj, constr).solve()

print x.value, y.value
```
- results in $x \approx 1.8$, $y \approx .36$

## Diet Problem
- a classic problem in optimization is to meet the nutritional requirements
of an army via various foods (with different nutritional benefits and prices) under cost constraints
- one soldier requires 1, 2.1, and 1.7 units of meat, vegetables, and grain, respectively, per day ($r = (1, 2.1, 1.7)$)
- one unit of hamburgers has nutritional value $h = (.8, .4, .5)$ and costs $1
- one unit of cheerios has nutritional value $c = (0, .3, 2.0)$ and costs $0.25
- prices $p = (1, 0.25)$
- you have a budget of $130 to buy hamburgers and cheerios for one day
- can you meet the dietary needs of 50 soldiers?

## Diet Problem
- write as optimization problem
$$
\begin{array}{ll}
\mbox{minimize} & 0 \\
\mbox{subject to} & p^T x \leq 130\\
&x_1 h + x_2 c \geq 50 r\\
&x \geq 0
\end{array}
$$
with $x$ giving units of hamburgers and cheerios
- or, with $A = [h, c]$,
$$
\begin{array}{ll}
\mbox{minimize} & 0 \\
\mbox{subject to} & p^T x \leq 130\\
&Ax \geq 50 r\\
&x \geq 0
\end{array}
$$

## Diet Problem: CVXPY Code
```
x = Variable(2)
obj = Minimize(0)
constr = [x.T*p <= 130,
          h*x[0] + c*x[1] >= 50*r,
          x >= 0]

prob = Problem(obj, constr)
prob.solve(solver='SCS')
print x.value
```
- non-unique solution $x \approx (62.83, 266.57)$


# Convex Functions
## First-order condition
- for **differentiable** $f:\reals^n \to \reals$, the **gradient** $\nabla f$ exists at each point in $\dom(f)$
- $f$ is convex if and only if $\dom(f)$ is convex and
$$
f(y) \geq f(x) + \nabla f(x)^T(y-x)
$$
for all $x,y \in \dom(f)$
- that is, the first-order Taylor approximation is a **global underestimator** of $f$
$$
\includegraphics[width=0.5\textwidth]{fig/under_est.pdf}
$$

## Second-order condition
- for **twice differentiable** $f: \reals^n \to \reals$, the **Hessian** $\nabla^2 f$, or second derivative matrix, exists at each point in $\dom(f)$
- $f$ is convex if and only if for all $x \in \dom(f)$,
$$
\nabla^2 f(x) \succeq 0
$$
- that is, the Hessian matrix must be **positive semidefinite**
- if $n=1$, simplifies to $f^{''}(x) \geq 0$
- useful to determine convexity
- of course, there are many non-differentiable convex functions and the first- and second-order conditions generalize

## Positive semidefinite matrices

- a matrix $A \in \reals^{n \times n}$ is **positive semidefinite** ($A \succeq 0$) if
    - $A$ is **symmetric**: $A = A^T$
    - $x^TAx \geq 0$ for all $x \in \reals^n$
- $A \succeq 0$ if and only if all **eigenvalues** of $A$ are nonnegative
- intuition: graph of $f(x) = x^T A x$ looks like a bowl

## Examples in $\reals$
+---------------------+--------------+
| $f(x)$              | $f''(x)$     |
+=====================+==============+
| $x$                 | $0$          |
+---------------------+--------------+
| $x^2$               | $1$          |
+---------------------+--------------+
| $e^{ax}$            | $a^2 e^{ax}$ |
+---------------------+--------------+
| $1/x\ (x > 0)$      | $2/x^3$      |
+---------------------+--------------+
| $-\log(x)\ (x > 0)$ | $1/x^2$      |
+---------------------+--------------+

## Quadratic functions
- for $A \in R^{n \times n}$, $A \succeq 0$, $b \in \reals^n$, $c \in \reals$, the quadratic function
$$
f(x) = x^T A x + b^T x + c
$$
is convex, since $\nabla^2 f(x) = A \succeq 0$
- in particular, the least squares objective
$$\|Ax - b \|^2_2 = x^T A^T A x - 2(Ab)^T x + b^T b$$
is convex since $A^TA \succeq 0$

## Epigraph
- the **epigraph** of a function is given by the set
$$
\epi(f) = \lbrace (x,t)\ \vert\ f(x) \leq t \rbrace
$$
- if $f$ is convex, then $\epi(f)$ is convex
$$
\includegraphics[width=0.25\textwidth]{fig/epigraph.pdf}
$$
- the **sublevel sets** of a convex function
$$
\lbrace x\ \vert\ f(x) \leq c \rbrace
$$
are convex for any fixed $c \in \reals$

## Ellipsoid
- any **ellipsoid**
$$
C = \lbrace x\ \vert\  (x-x_c)^TP(x-x_c) \leq 1\rbrace
$$
with $P \succeq 0$ is convex because it is the sublevel set of a convex quadratic function
$$
\includegraphics[width=0.5\textwidth]{fig/ellipsoid.pdf}
$$

## More convex and concave functions
- any norm is convex: $\|\cdot\|_1$, $\|\cdot\|_2$, $\|\cdot\|_\infty$
- $\max(x_1, \ldots, x_n)$ is convex
- $\min(x_1, \ldots, x_n)$ is concave
- absolute value $|x|$ is convex
- $x^a$ is **convex** for $x > 0$ if $a \geq 1$ or $a \geq 0$
- $x^a$ is **concave** for $x > 0$ if $0 \leq a \leq 1$

- **lots** more; for reference:
    - CVX Users' Guide, `http://web.cvxr.com/cvx/doc/funcref.html` 
    - CVXPY Tutorial, `http://www.cvxpy.org/en/latest/tutorial/functions/index.html`
    - *Convex Optimization* by Boyd and Vandenberghe



## Operations that preserve convexity
### Positive weighted sums
- if $f_1, \ldots, f_n$ are convex and $w_1, \ldots, w_n$ are all positive (or nonnegative) real numbers, then
$$
w_1 f_1(x) + \cdots + w_n f_n(x)
$$
is also convex

- $7x + 2/x$ is convex
- $x^2 - \log(x)$ is convex
- $-e^{-x} + x^{0.3}$ is concave

## Operations that preserve convexity
### Composition with affine function
- if $f: \reals^n \to \reals$ is convex, $A \in \reals^{n \times m}$, and $b \in \reals^n$, then
$$
g(x) = f(Ax + b)
$$
is convex with $g : \reals^m \to \reals$
- mind the domain: $\dom(g) = \lbrace x\ \vert\ Ax + b \in \dom(f) \rbrace$


## Operations that preserve convexity 
### Function composition
- let $f, g:\reals \to \reals$, and $h(x) = f(g(x))$
- if $f$ is **increasing** (or nondecreasing) on its domain:
    - $h$ is convex if $f$ and $g$ are convex
    - $h$ is concave if $f$ and $g$ are concave
- if $f$ is **decreasing** (or nonincreasing) on its domain:
    - $h$ is convex if $f$ is convex and $g$ is concave
    - $h$ is concave if $f$ is concave and $g$ is convex
- mnemonic:
    - "-" (decreasing) swaps "sign" (convex, concave)
    - "+" (increasing) keeps "sign" the same (convex, convex)

## Operations that preserve convexity 
### Function composition examples
- mind the domain and range of the functions
- $\frac{1}{\log(x)}$ is convex (for $x > 1$)
    - $1/x$ is convex, decreasing (for $x > 0$)
    - $\log(x)$ is concave (for $x > 1$)
- $\sqrt{1 - x^2}$ is concave (for $|x| \leq 1$)
    - $\sqrt{x}$ is concave, increasing (for $x > 0$)
    - $1-x^2$ is concave

## Operations that preserve convexity
- `dcp.stanford.edu` website for constructing complex convex expressions to learn composition rules

$$
\includegraphics[width=0.55\textwidth]{fig/dcp.png}
$$

## CVXPY example
- recall that the **least squares** problem
$$
\begin{array}{ll}
\mbox{minimize} & \| Ax - b\|_2^2
\end{array}
$$
is convex
- adding an $\|x\|_1$ term to the objective has an interesting effect: it "encourages" the solution $x$ to be **sparse**
- the problem
$$
\begin{array}{ll}
\mbox{minimize} & \| Ax - b\|_2^2 + \rho \|x\|_1
\end{array}
$$
is called the LASSO and is central to the field of *compressed sensing*

## CVXPY example
- $A \in \reals^{30 \times 100}$, with $A_{ij} \sim \mathcal{N}(0,1)$
- observe $b = Ax + \varepsilon$, where $\varepsilon$ is noise
- more unknowns than observations!
- however, $x$ is known to be sparse
- true $x$:

\centering
\includegraphics[width=0.5\textwidth]{fig/x_true.pdf}


## CVXPY example
least squares recovery given by
```
x = Variable(n)
obj = sum_squares(A*x - b)
Problem(Minimize(obj)).solve()
```
\centering
\includegraphics[width=0.5\textwidth]{fig/ls_recovery.pdf}


## CVXPY example
LASSO recovery given by
```
x = Variable(n)
obj = sum_squares(A*x - b) + rho*norm(x,1)
Problem(Minimize(obj)).solve()
```
\centering
\includegraphics[width=0.5\textwidth]{fig/lasso_recovery.pdf}


# Convex Optimization Problems
## Convex optimization problems
- combines convex objective functions with convex constraint sets
- constraints describe acceptable, or **feasible**, points
- objective gives desirability of feasible points

$$
\begin{array}{ll}
\mbox{minimize} & f(x)\\
\mbox{subject to} & x \in C_1\\
& \vdots\\
& x \in C_n \\
\end{array} 
$$

## Constraints
- in CVXPY and other modeling languages, convex constraints are often given in
epigraph or sublevel set form
    - $f(x) \leq t$ or $f(x) \leq 1$ for convex $f$
    - $f(x) \geq t$ for concave $f$

## Equivalent problems
- loosely, we'll say that two optimization problems are **equivalent**
if the solution from one is easily obtained from the solution to the other
- **epigraph** transformations:
$$
\begin{array}{ll}
\mbox{minimize} & f(x) + g(x)
\end{array} 
$$
equivalent to
$$
\begin{array}{ll}
\mbox{minimize} & t + g(x)\\
\mbox{subject to} & f(x) \leq t
\end{array} 
$$

## Equivalent problems
- **slack variables**:
$$
\begin{array}{ll}
\mbox{minimize} & f(x)\\
\mbox{subject to} & Ax \leq b
\end{array} 
$$
equivalent to
$$
\begin{array}{ll}
\mbox{minimize} & f(x)\\
\mbox{subject to} & Ax + t = b\\
& t \geq 0
\end{array} 
$$

## Equivalent problems
- **dummy variables**:
$$
\begin{array}{ll}
\mbox{minimize} & f(Ax + b)
\end{array} 
$$
equivalent to
$$
\begin{array}{ll}
\mbox{minimize} & f(t)\\
\mbox{subject to} & Ax + b = t
\end{array} 
$$

## Equivalent problems
- **function transformations**:
$$
\begin{array}{ll}
\mbox{minimize} & \|Ax - b\|_2^2
\end{array} 
$$
equivalent to
$$
\begin{array}{ll}
\mbox{minimize} & \|Ax - b\|_2
\end{array} 
$$
since the square-root function is monotone

# Examples
## Diet problem
- wanted to know if we could feed an army of 50 with a budget of $130:
$$
\begin{array}{ll}
\mbox{minimize} & 0 \\
\mbox{subject to} & p^T x \leq 130\\
&x_1 h + x_2 c \geq 50 r\\
&x \geq 0
\end{array}
$$
with $x$ giving units of hamburgers and cheerios
- no objective; a set feasibility problem

## Diet problem
- reformulate the problem to find the cheapest diet:
$$
\begin{array}{ll}
\mbox{minimize} & p^T x \\
\mbox{subject to} &
x_1 h + x_2 c \geq 50 r\\
&x \geq 0
\end{array}
$$
- with CVXPY, we feed the troops for $129.17:
```
x = Variable(2)
obj = Minimize(x.T*p)
constr = [h*x[0] + c*x[1] >= 50*r,
          x >= 0]
Problem(obj, constr).solve()
```


## Image in-painting
\includegraphics[width=\textwidth]{fig/inpaint_text1}

## Image in-painting
guess pixel values in obscured/corrupted parts of image

- **decision variable** $x \in \reals^{m \times n \times 3}$
- $x_{i,j} \in [0,1]^3$ gives RGB values of pixel $(i, j)$
- many pixels missing
- known pixel IDs given by set $K$, values given by **data** $y \in \reals^{m \times n \times 3}$

**total variation in-painting**:
choose pixel values $x_{i,j} \in \reals^3$ to
minimize
$$
\mbox{TV}(x) = \sum_{i,j} \left\| \left[ \begin{array}{c}
x_{i+1,j}-x_{i,j}\\
x_{i,j+1}-x_{i,j}
\end{array}\right]\right\|_2
$$
that is, for each pixel, minimize distance to neighbors below and to the right, subject to known pixel values

## In-painting: Convex model
$$
\begin{array}{ll} \mbox{minimize} & \mbox{TV}(x)\\
\mbox{subject to} & x_{i, j} = y_{i,j} \text{ if } (i,j) \in K
\end{array}
$$

## In-painting: Code example

```python
# K[i, j] == 1 if pixel value known, 0 if unknown
from cvxpy import *
variables = []
constr = []
for i in range(3):
    x = Variable(rows, cols)
    variables += [x]
    constr += [mul_elemwise(K, x - y[:,:,i]) == 0]

prob = Problem(Minimize(tv(*variables)), constr)
prob.solve(solver=SCS)
```

## In-painting: $512 \times 512$ color image; about 800k variables
\includegraphics[width=\textwidth]{fig/inpaint_text1}

## In-painting
\includegraphics[width=\textwidth]{fig/inpaint_text2}

## In-painting ($80\%$ of pixels removed)
\includegraphics[width=\textwidth]{fig/inpaint80_1}

## In-painting ($80\%$ of pixels removed)
\includegraphics[width=\textwidth]{fig/inpaint80_2}


## Vehicle tracking
\includegraphics[width=1\textwidth]{fig/rkf1.pdf}

## Kalman filtering
- estimate vehicle path from noisy position measurements (with outliers)
- dynamic model of vehicle state $x_t$:
$$
x_{t+1} = Ax_t + Bw_t, \quad y_t=Cx_t + v_t
$$
    + $x_t$ is vehicle state (position, velocity)
    + $w_t$ is unknown drive force on vehicle
    + $y_t$ is position measurement; $v_t$ is noise

- Kalman filter: estimate $x_t$ by minimizing
$\sum_t \left(\|w_t\|_2^2+ \gamma \|v_t\|_2^2\right)$
- a least-squares problem; assumes $w_t,v_t$ Gaussian

## Robust Kalman filter
- to handle outliers in $v_t$, replace square cost with Huber cost
- **robust** Kalman filter:
$$
\begin{array}{ll}
\mbox{minimize} & \sum_t \left( \|w_t\|^2_2 + \gamma \phi(v_t) \right)\\
\mbox{subject to} & x_{t+1} = Ax_t + Bw_t, \quad y_t = Cx_t+v_t
\end{array}
$$
where $\phi$ is Huber function

## Robust KF CVXPY code

```python
from cvxpy import *
x = Variable(4,n+1)
w = Variable(2,n)
v = Variable(2,n)
    
obj = sum_squares(w)
obj += sum(huber(norm(v[:,t])) for t in range(n))
obj = Minimize(obj)
constr = []
for t in range(n):
    constr += [ x[:,t+1] == A*x[:,t] + B*w[:,t] ,
                y[:,t]   == C*x[:,t] + v[:,t]   ]

Problem(obj, constr).solve()
```

## Example
- 1000 time steps
- $w_t$ standard Gaussian 
- $v_t$ standard Gaussian, except $30\%$ are outliers with $\sigma = 10$

## Example
\includegraphics[width=1\textwidth]{fig/rkf1.pdf}

## Example
\includegraphics[width=1\textwidth]{fig/rkf2.pdf}

## Example
\includegraphics[width=1\textwidth]{fig/rkf3.pdf}