---
title: Support Vector Machines
author: |
  | Nick Henderson, AJ Friend
  | Stanford University
---

## Support Vector Machines
- SVM finds a hyperplane to separate points into two classes
- maybe an example with data
- abstract points afterwards

## Problem
- data $x_i \in \reals^n$ with labels $y_i \in \lbrace +1, -1 \rbrace$
- want to find a hyperplane $\lbrace x\ \vert\ a^T x = b\rbrace$ that separates the points according to label
\begin{align*}
a^Tx_i - b > 0 \mbox{ for } y_i = +1\\
a^Tx_i - b < 0 \mbox{ for } y_i = -1
\end{align*}
- **strict** separability if and only if 
\begin{align*}
a^Tx_i - b \geq +1 \mbox{ for } y_i = +1\\
a^Tx_i - b \leq -1 \mbox{ for } y_i = -1
\end{align*}

## more problem
- rewrite as
$$
y_i\left(a^Tx_i - b\right) \geq 1
$$
for all $i$
- let $w = [a,b]$, $z_i = [x_i, -1]$, then
$$
y_i w^T z_i \geq 1
$$
- eehhhh, how do i work in hinge loss?

## transformation
- use transformed data $z_i = y_i x_i$ so that we want $a$, $b$
such that
$$
a^Tz_i > b \mbox{ for all } i
$$
(that's just wrong, figure out what I want to say)


## Separable linear classification/discrimination
- many hyperplanes
- maximum margin classifier and robustness

## Nonseparable linear classification
- relaxed feasibility problem
- l1 penality to minimize misclassificaiton: pure LP
- tradeoff between classification and width of slab: SOCP

## Hinge loss
- reformulate as hinge loss objective
- general loss function form... $l(Ax+b)$

## logistic
- change loss function to get logistic loss
- other loss functions

## regularization
- regularize to get sparse classifier...

## nonlinear discrimination
- adding features
- polynomial discrimination any different?
- rbf kernel? radial basis function
- kernel methods and relationship with convex opt...

## algorithms
- note that so far, we have said **nothing** about **how** to compute a supporting vector
- we have focused on modeling
- that's OK, we're focusing on modeling
- algorithms involve duality and optimality conditions

## scikitlearn comparison
- make sure it matches up with python SVM formulation
- maybe even do a timing comparison...

## data science perspective
- cleaning and centering data
- sparse predictors