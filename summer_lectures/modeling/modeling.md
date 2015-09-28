---
title: Applied Convex Models
author: |
  | Nick Henderson, AJ Friend
  | Stanford University
---

# Portfolio optimization

## Portfolio allocation vector

- invest fraction $w_i$ in asset $i$ for $i = 1,\dots, n$
- $w \in \reals^n$ is *portfolio allocation vector*
- $\mathbf{1}^T w = 1$
- $w_i < 0$ means *short position* in asset $i$ (borrow shares and sell now;
replace later)
- $w \ge 0$ is a *long only* portfolio
- $\|w\|_1 = \mathbf{1}^T w_+ + \mathbf{1}^T w_-$ is *leverage* (there are other
  definitions)

## Asset Returns

- investments held for one period
- initial prices $p_i >0$; end of period process $p_i^+ >0$
- asset (fractional) returns $r_i = (p_i^+ - p_i)/p_i$
- portfolio (fractional) return $R = r^T w$
- common model: $r$ is a random variable, with mean $\mathbf{E} r = \mu$,
  covariance $\mathbf{E}(r-\mu)(r-\mu)^T = \Sigma$
- so $R$ is a random variable with $\mathbf{E}R = \mu^T w$, $\mathbf{var}(R)=w^T
\Sigma w$
- $\mathbf{E}R$ is (mean) return of portfolio
- $\mathbf{var}(R)=w^T \Sigma w$ is risk of portfolio
- Finance: high return, low risk

## Classical (Markowitz) portfolio optimization

$$
\begin{array}{ll} \mbox{minimize} & \mu^T w - \gamma w^T \Sigma w\\
\mbox{subject to} & \mathbf{1}^T w = 1,\ w \in \mathcal{W}
\end{array}
$$

- variable $w \in \reals^n$
- $\mathcal{W}$ is set of allowed portfolios
- common case $\mathcal{W} = \reals^n_+$ (long only)
- $\gamma > 0$ is risk aversion parameter
- $\mu^T w - \gamma w^T \Sigma w$ is risk-adjusted return
- varying $\gamma$ gives optimal risk-return trade-off
- can also fix return and minimize risk, etc.
- To limit leverage use $\|w\|_1 \le L^{\text{max}}$

# Nonnegative matrix factorization
## Nonnegative matrix factorization
- **goal**: factor $A \in \mathbf{R}^{m \times n}_+$ such that
$$
A \approx WH,
$$
where $W \in \mathbf{R}^{m \times k}_+$, $H \in \mathbf{R}^{k \times n}_+$
and $k \ll n,m$
- $W$, $H$ give nonnegative low-rank approximation to $A$
- low-rank means data more interpretable as combination of just $k$ features
- nonegativity may be natural to the data, e.g., no negative words in a document
- applications in recommendation systems, signal processing, clustering, computer vision, natural language processing


## NMF formulation
- many ways to formalize $A \approx WH$
- for given $A$ and $k$, we'll try to find $W$ and $H$ that solve
$$
\begin{array}{ll}
\mbox{minimize}_{W, H} & \| A - WH\|_F^2\\
\mbox{subject to} & W_{ij} \geq 0 \\
& H_{ij} \geq 0
\end{array}
$$
- $\|X\|_F = \sqrt{\sum_{ij} X_{ij}^2}$ is the matrix **Frobenius norm**

## Principal component analysis
- NMF can be thought of as a dimensionality reduction technique
- PCA is a related dimensionality reduction method, solving the problem
$$
\begin{array}{ll}
\mbox{minimize}_{W, H} & \| A - WH\|_F^2\\
\end{array}
$$
for $W \in \mathbf{R}^{m \times k}_+$, $H \in \mathbf{R}^{k \times n}_+$, without nonnegativity constraint
- PCA has "analytical" solution via the **singular value decomposition**
- won't go further into the interpretation of the models; focus on methods for computing NMF instead

## Biconvexity
- the NMF problem
$$
\begin{array}{ll}
\mbox{minimize}_{W, H} & \| A - WH\|_F^2\\
\mbox{subject to} & W_{ij} \geq 0 \\
& H_{ij} \geq 0
\end{array}
$$
is **nonconvex** due to the product $WH$
- however, the objective function is **biconvex**: convex in either $W$ or $H$ if we hold the other fixed

## Alternating minimization
biconvexity suggests the following algorithm:

- initialize $W^0$
- for $k = 0, 1, 2, \ldots$
- $$
\begin{array}{lll}
H^{k+1} = &\mbox{argmin}_{H} & \| A - W^k H\|_F^2\\
&\mbox{subject to} & H_{ij} \geq 0
\end{array}
$$
- $$
\begin{array}{lll}
W^{k+1} = &\mbox{argmin}_{W} & \| A - W H^{k+1}\|_F^2\\
&\mbox{subject to} & W_{ij} \geq 0
\end{array}
$$

## Discussion
- expression $A - W^k H$ is **linear** in variable $H$
- $\| A - W^k H\|_F^2$ is exactly the least squares objective, but with matrix instead of vector variable
- each subproblem is a convex nonnegative least squares problem
- no guarantee of global minimum, but we do get a local minimum
- due to biconvexity, the objective function **decreases** at each iteration, meaning that the iteration converges

## Extensions
sparse factors with $\ell_1$ penalty
$$
\begin{array}{ll}
\mbox{minimize}_{W, H} & \| A - WH\|_F^2 + \sum_{ij} \left( |W_{ij}| + |H_{ij}|\right)\\
\mbox{subject to} & W_{ij} \geq 0 \\
& H_{ij} \geq 0
\end{array}
$$

## Extensions
**matrix completion**: only observe subet of entries $A_{ij}$ for $(i,j) \in \Omega$

- use low-rank assumption to estimate missing entries

$$
\begin{array}{ll}
\mbox{minimize}_{W, H, Z} & \sum_{i,j \in \Omega} (A_{ij} - Z_{ij})^2\\
\mbox{subject to} & Z = WH\\
& W_{ij} \geq 0 \\
& H_{ij} \geq 0
\end{array}
$$



# Optimal advertising
## Ad display
- $m$ advertisers/ads, $i=1, \ldots, m$
- $n$ time slots, $t=1, \ldots, n$
- $T_t$ is total traffic in time slot $t$
- $D_{it} \geq 0$ is number of ad $i$ displayed in period $t$
- $\sum_i D_{it} \leq T_t$
- contracted minimum total displays: $\sum_t D_{it} \geq c_i$
- goal: choose $D_{it}$

## Clicks and revenue
- $C_{it}$ is number of clicks on ad $i$ in period $t$
- click model: $C_{it} = P_{it}D_{it}$, $P_{it} \in [0,1]$
- payment: $R_i>0$ per click for ad $i$, up to budget $B_i$
- ad revenue
$$
S_i = \min \lbrace R_i \sum_t C_{it}, B_i\rbrace
$$
is a concave function of $D$

## Ad optimization
- choose displays to maximize revenue:
$$
\begin{array}{ll} \mbox{maximize} & \sum_i S_i \\
\mbox{subject to} & D \geq 0, \quad D^T \ones \leq T, \quad
D \ones \geq c
\end{array}
$$
- variable is $D\in \reals^{m \times n}$
- data are $T$, $c$, $R$, $B$, $P$

## Example
- 24 hourly periods, 5 ads (A--E)
- total traffic:
$$
\includegraphics[width=0.65\textwidth]{fig/traffic.png}
$$

## Example
- ad data:

+-------+-------+-------+-------+-------+-------+
| Ad    | A     | B     | C     | D     | E     |
+=======+=======+=======+=======+=======+=======+
| $c_i$ | 61000 | 80000 | 61000 | 23000 | 64000 |
+-------+-------+-------+-------+-------+-------+
| $R_i$ | 0.15  | 1.18  | 0.57  | 2.08  | 2.43  |
+-------+-------+-------+-------+-------+-------+
| $B_i$ | 25000 | 12000 | 12000 | 11000 | 17000 |
+-------+-------+-------+-------+-------+-------+

## Example
- ad revenue

+-----------------+-------+-------+--------+-------+--------+
| Ad              | A     | B     | C      | D     | E      |
+=================+=======+=======+========+=======+========+
| $c_i$           | 61000 | 80000 | 61000  | 23000 | 64000  |
+-----------------+-------+-------+--------+-------+--------+
| $R_i$           | 0.15  | 1.18  | 0.57   | 2.08  | 2.43   |
+-----------------+-------+-------+--------+-------+--------+
| $B_i$           | 25000 | 12000 | 12000  | 11000 | 17000  |
+-----------------+-------+-------+--------+-------+--------+
| $\sum_t D_{it}$ | 61000 | 80000 | 148116 | 23000 | 167323 |
+-----------------+-------+-------+--------+-------+--------+
| $S_i$           | 182   | 12000 | 12000  | 11000 | 7760   |
+-----------------+-------+-------+--------+-------+--------+




