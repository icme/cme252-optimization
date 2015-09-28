---
title: Cable Laying
author: |
  | Nick Henderson, AJ Friend
  | Stanford University
date: August 6, 2015
---

## Laying cable: another example

* Housing developer wants to minimize the length of cable needed to connect
  houses to the internet

* Each house must be connected to a distribution center (DC) through the trench
  network

* Question: how to choose location of distribution centers and the assignment of
  houses to minimize cable length?

## Laying cable: example development

\centering
\includegraphics[width=.6\textwidth]{fig/cable-map.pdf}

## Laying cable: data

* 1676 houses
* 1904 "junctions"
* 3580 edges
* 28 km of trench

## Laying cable: problem

Problem statement

* Given a network of houses in a fully connected trench network, find the
  location of distribution centers (DCs) and assignment of houses to minimize
  cable length.

Properties:

* The combinatorial nature of this problem likely puts it in the class of NP-hard
problems

* there is no known algorithm to compute an optimal solution in a period of time
that does not grow exponentially with the size of the problem

Solution: alternating minimization

* The first part is the assignment of DCs to junction nodes

* The second is the assignment of houses to DCs

* Not guaranteed to find an optimal solution.  However, experiments show it is a
  practical solution

## Laying cable: notation and variables

\centering
\begin{tabular}{cl}
$h$ & number of houses \\
$i$ & index over houses \\
$c$ & number of DCs \\
$j$ & index over DCs \\
$n$ & number of junctions \\
$k$ & index over junctions \\
$A\in\reals^{h+n \times h+n}$ & adjacency matrix \\
$D\in\reals^{h\times n}$ & distance matrix between houses and junctions\\
$X\in\{0,1\}^{h\times c}$ & assignment matrix of houses to DCs \\
$Y\in\{0,1\}^{c\times n}$ & assignment matrix of DCs to junctions \\
$\delta$ & DC capacity
\end{tabular}

## Laying cable: distribution center placement

Given housing assignment matrix $X$ and distance matrix $D$ the optimal DC
assignment matrix $Y$ is easily computed.  First, compute the matrix of cable
lengths for any DC placement:

$$
Z = X^T D \in \reals^{c\times n}.
$$

DC assignment matrix $Y$ is constructed to assign DC $j$ to junction
$\text{argmin}_k\ Z_{jk}$ for all $j$.

This is a simple computation and does not require optimization software.

## Laying cable: assignment of houses to DCs

Given DC assignment matrix $Y$ and distance matrix $D$ the optimal housing
assignment matrix $X$ is computed using a LP/MIP solver.  First, compute
the matrix of distances from each house to the fixed DC locations:

$$
C = DY^T \in \reals^{h \times c}.
$$

Second, use LP/MIP software to solve the optimization problem:

$$
\begin{array}{ll}
\text{minimize}   & \text{vec}(C)^T \text{vec}(X) \\ 
\text{subject to} & X \in \{0,1\}^{h\times c} \\
                  & Xe_c = e_h \\
                  & X^T e_h \le \delta e_c
\end{array}
$$

## Laying cable: solution

\centering
\includegraphics[width=.8\textwidth]{fig/cable-solution.png}

## Laying cable: results

| configuration | length | cost (of cable) |
|---------------|--------|-----------------|
| initial       | 194 km | $110,580        |
| optimized     | 127 km | $72,390         |
|---------------|--------|-----------------|
| difference    | -67 km | -$38,190        |

## Laying cable: wrapping up

* it is often useful (necessary) to decompose a hard problem into a sequence of
  easier ones

* good (and useful) solutions are not necessarily globally optimal

* also good to map your problem to existing computational tools
