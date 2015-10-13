% CME 252: Support Vector Machines
% AJ Friend \
  ICME, Stanford University

# Introduction
## Support Vector Machines
- many related/overlapping names:
    - maximum margin classifier
    - support vector classifier
    - (robust) linear discrimination/classification 
    - support vector machine
- I won't always use the right name
- we'll start with:
    - find a hyperplane to separate data points into two classes
    - use hyperplane to classify new (unseen) points

## Support Vector Machines
\centering
\includegraphics[width=0.7\textwidth]{fig/first.pdf}

## Scenarios
- classify data in increasingly sophisticated scenarios:
    + strictly linearly separable
    + approximately (not strictly) linearly separable
    + approximately non-linearly separable (hyperplanes won't work)

## Strictly Linearly Separable Data
\centering
\includegraphics[width=0.65\textwidth]{fig/lin_sep.pdf}

## Approximately Linearly Separable Data
\centering
\includegraphics[width=0.65\textwidth]{fig/approx_lin_sep.pdf}

## Approximately Non-linearly Separable
\centering
\includegraphics[width=0.65\textwidth]{fig/non_lin_sep.pdf}

# Linearly Separable Problem
## Linearly Separable Problem
- data: $x_i \in \reals^n$ with labels $y_i \in \lbrace +1, -1 \rbrace$ for $i = 1, \ldots, N$
- assume **strictly** linearly separable
- find hyperplane $\lbrace x\ \vert\ a^T x = b\rbrace$ that separates points by label
\begin{align*}
a^Tx_i - b > 0 \mbox{ if } y_i = +1\\
a^Tx_i - b < 0 \mbox{ if } y_i = -1
\end{align*}
- **rescale** $a$, $b$ so that
\begin{align*}
a^Tx_i - b \geq +1 \mbox{ if } y_i = +1\\
a^Tx_i - b \leq -1 \mbox{ if } y_i = -1
\end{align*}

## Linearly Separable Problem
\centering
\includegraphics[width=0.65\textwidth]{fig/regions.pdf}

## Linearly Separable Problem
- for all $i$, rewrite constraints as
$$
y_i\left(a^Tx_i - b\right) \geq 1
$$
- get **feasibility** problem
$$
\begin{array}{ll}
\mbox{minimize} & 0 \\
\mbox{subject to} & y_i\left(a^Tx_i - b\right) \geq 1 \mbox{ for } i = 1, \ldots, N
\end{array}
$$
with variables $a \in \reals^n$, $b \in \reals$

## CVXPY for Separable Problem

```python
a = Variable(n)
b = Variable()

obj = Minimize(0)
constr = [mul_elemwise(y, X*a - b) >= 1]
Problem(obj, constr).solve()
```

# Which Separator?
## Which Separator?
\centering
\includegraphics[width=0.65\textwidth]{fig/which1.pdf}

## Which Separator?
\centering
\includegraphics[width=0.65\textwidth]{fig/which2.pdf}

## Which Separator?
\centering
\includegraphics[width=0.65\textwidth]{fig/which3.pdf}

## Which Separator?
\centering
\includegraphics[width=0.65\textwidth]{fig/which4.pdf}

# Maximum Margin Classifier
## Maximum Margin Classifier
- infinitely many choices for separating hyperplane
- choose one which maximizes **width** of separating **slab**
$$
\lbrace x \mid -1 \leq a^T x - b \leq +1 \rbrace
$$
- "maximum margin" or "robust linear" classifier

\centering
\includegraphics[width=0.45\textwidth]{fig/slab.pdf}

## Maximum Margin Classifier
- **margin**, or width of separating slab
$$
\lbrace x \mid -1 \leq a^T x - b \leq +1 \rbrace
$$
is $2/\|a\|_2$ (via linear algebra)
- suggests optimization problem
$$
\begin{array}{ll}
\mbox{maximize} & 2/\|a\|_2 \\
\mbox{subject to} & y_i\left(a^Tx_i - b\right) \geq 1 \mbox{ for } i = 1, \ldots, N
\end{array}
$$
- but not convex!

## Maximum Margin Classifier
- reformulate:
$$
\mbox{maximize}\ 2/\|a\|_2 \iff \mbox{minimize}\ \|a\|_2
$$
gives
$$
\begin{array}{ll}
\mbox{minimize} & \|a\|_2 \\
\mbox{subject to} & y_i\left(a^Tx_i - b\right) \geq 1 \mbox{ for } i = 1, \ldots, N,
\end{array}
$$
the **maximum margin classifier** (MMC) problem

## CVXPY
```python
a = Variable(n)
b = Variable()

obj = Minimize(norm(a))
constr = [mul_elemwise(y, X*a - b) >= 1]
Problem(obj, constr).solve()
```

## Example
\centering
\includegraphics[width=0.5\textwidth]{fig/max_margin.pdf}

- note that max margin depends on only 3 tangent data points, called **support vectors**
- could throw away remaining data and get same solution

# Non-separable Linear Classification
## Non-separable Linear Classification
\centering
\includegraphics[width=0.65\textwidth]{fig/non_separable.pdf}

## Non-separable Linear Classification
- no separating hyperplane exists
- try finding linear separator
```
obj = Minimize(0)
constr = [mul_elemwise(y, X*a - b) >= 1]
prob = Problem(obj, constr)
prob.solve()
```
- results in `prob.status == 'infeasible'`

# Sparse Violation Classifier
## Sparse Violation Classifier
- idea: "relax" constraints to make problem feasible
- add **slack** variables $u \in \reals^N_+$ to allow data points to be
on "wrong side" of hyperplane
$$
y_i\left(a^Tx_i - b\right) \geq 1 - u_i,\quad u_i \geq 0
$$
    + $u_i = 0$: $x_i$ on **right** side of hyperplane
    + $0 < u_i < 1$: $x_i$ on **right** side, but **inside slab**
    $\lbrace x \mid -1 \leq a^T x - b \leq +1 \rbrace$
    + $u_i > 1$: $x_i$ on **wrong** side of hyperplane

## Sparse Violation Classifier
- $u$ gives measure of how much constraints are violated
- for large $u$ can make **any** data feasible
- want $u$ "small"; minimize its sum
$$
\begin{array}{ll}
\mbox{minimize} & \mathbf{1}^T u \\
\mbox{subject to} & y_i\left(a^Tx_i - b\right) \geq 1 - u_i \mbox{ for } i = 1, \ldots, N\\
&u \geq 0
\end{array}
$$
- I'll call it **sparse violation classifier** (SpVC)
- $\mathbf{1}^T u = \|u\|_1$, since $u \geq 0$; good **heuristic** for separator with few (sparse) violations

## CVXPY
```python
a = Variable(n)
b = Variable()
u = Variable(N)

obj = Minimize(sum_entries(u))
constr = [mul_elemwise(y, X*a - b) >= 1 - u, u >= 0]
Problem(obj, constr).solve()
```

## Example
\centering
\includegraphics[width=0.6\textwidth]{fig/sparse.pdf}

- solution depends only on points inside of, tangent to, or on wrong side of slab

## Example
\centering
\includegraphics[width=0.5\textwidth]{fig/violations.pdf}

- "$+$" class has 3 misclassified points
- "$-$" class has 2 correctly classified, but inside slab

# Support Vector Classifier
## Support Vector Classifier
- idea: combine aspects of last two classifiers
    - sparse violations of SpVC
    - robustness of large separating slab in MMC
- optimize both:
$$
\begin{array}{ll}
\mbox{minimize} & \|a\|_2 + \rho\mathbf{1}^T u \\
\mbox{subject to} & y_i\left(a^Tx_i - b\right) \geq 1 - u_i \mbox{ for } i = 1, \ldots, N\\
&u \geq 0
\end{array}
$$
- $\rho > 0$ trades-off between margin $2/\|a\|_2$ and classification violations $\mathbf{1}^T u$ (multi-objective optimization)
- **support vector classifier** (SVC)

## CVXPY
```python
a = Variable(n)
b = Variable()
u = Variable(N)
rho = .1

obj = Minimize(norm(a) + rho*sum_entries(u))
constr = [mul_elemwise(y, X*a - b) >= 1 - u, u >= 0]
Problem(obj, constr).solve()
```

## Example with $\rho = .1$
\centering
\includegraphics[width=0.65\textwidth]{fig/svc1.pdf}

## Example with $\rho = 10$
\centering
\includegraphics[width=0.65\textwidth]{fig/svc2.pdf}


# Loss Functions
## Hinge Loss
- in SpVC, it follows from $y_i\left(a^Tx_i - b\right) \geq 1 - u_i$, $u_i \geq 0$, that
$$
u_i = \begin{cases}
0 & y_i\left(a^Tx_i - b\right) \geq 1\\
1 - y_i\left(a^Tx_i - b\right) & y_i\left(a^Tx_i - b\right) < 1
\end{cases}
$$
- rewrite as $u_i = \ell_h\left[y_i\left(a^Tx_i - b\right)\right]$, where
$$
\ell_h(z) = \begin{cases}
0 & z \geq 1\\
1 - z & z < 1
\end{cases}
$$
is the **hinge loss** function, equivalently: $\max(0, 1-z)$ or $(1-z)_+$

## Hinge Loss
- $u_i = \ell_h\left[y_i\left(a^Tx_i - b\right)\right]$
- no penality if $y_i\left(a^Tx_i - b\right) \geq 1$
- linear penalty otherwise

\centering
\includegraphics[width=0.5\textwidth]{fig/hinge_loss.pdf}

## Hinge Loss SpVC
- note that $\ell_h$ is convex, so we can rewrite SpVC
as the **equivalent problem**
$$
\begin{array}{ll}
\mbox{minimize} & \sum_{i=1}^N \ell_h\left[y_i\left(a^Tx_i - b\right) \right]
\end{array}
$$
- unconstrained (non-differentiable) convex problem
- in CVXPY:

    ```python
    def hinge(z):
        return pos(1-z)

    r = mul_elemwise(y, X*a - b)
    obj = Minimize(sum_entries(hinge(r)))
    Problem(obj).solve()
    ```

## Why Hinge Loss?
- "0-1" loss: $\ell_{0-1}(z) = \begin{cases} 0 & z  \geq 1 \\ 1 & z < 1\end{cases}$
- can't solve nonconvex, combinatorial problem to minimize (discrete) number of violations with
$$
\begin{array}{ll}
\mbox{minimize} & \sum_{i=1}^N \ell_{0-1}\left[y_i\left(a^Tx_i - b\right) \right]
\end{array}
$$
- hinge loss gives a **convex** approximation to 0-1 loss

## Why Hinge Loss?
\centering
\includegraphics[width=0.55\textwidth]{fig/hinge_01.pdf}

- but not the **only** convex approximation

## Hinge Loss SVC
- can rewrite SVC as the **unconstrained** problem
$$
\begin{array}{ll}
\mbox{minimize} & \|a\|_2 + \rho\sum_{i=1}^N \ell_h\left[y_i\left(a^Tx_i - b\right) \right]
\end{array}
$$
- completely **equivalent** to the SVC formulation from before
- common form for classification problems:
$$
\begin{array}{ll}
\mbox{minimize} & r(a) + \rho \sum_{i=1}^N \ell\left[y_i\left(a^Tx_i - b\right) \right]
\end{array}
$$
    - $\ell$ is a **loss function** (fit to data)
    - $r$ is a **regularizer** (prior on parameters)
- **mix and match** regularizers and loss functions for different types of classification

## Logistic Loss
- **logistic loss** is an alternative to hinge loss:
$$
\ell_L(z) = \log(1 + \exp(-z))
$$
- convex, but not immediately obvious (2nd derivative test)

\centering
\includegraphics[width=0.45\textwidth]{fig/log_hinge.pdf}

## Logistic Regression
- get classic **logistic regression** with
$$
\begin{array}{ll}
\mbox{minimize} & r(a) + \rho \sum_{i=1}^N \ell\left[y_i\left(a^Tx_i - b\right) \right]
\end{array}
$$
when:
    - $r(a) \equiv 0$
    - $\ell(z) = \ell_L(z)$
- nice probabilistic interpretation
- **regularized** logistic regression when $r(a)$ is $\|a\|_2$ or $\|a\|_1$ (sparsity)

## Logistic Loss in CVXPY
- $\ell_L(z) = \log(1 + \exp(-z))$ doesn't follow convex composition rules
- to represent in CVXPY, use existing convex atom **log-sum-exp**:
$$
f(x) = \log\left(e^{x_1} + \cdots + e^{x_n} \right)
$$
- convexity follows from Hessian argument
- $$\ell_L(z) = \log(1 + \exp(-z)) =  \log\left(e^{0} + e^{-z} \right)$$

```python
def logistic(x):
    elems = []
    for xi in x:
        elems += [cvx.log_sum_exp(cvx.vstack(0, xi))]
    
    return cvx.vstack(*elems)
```

## Logistic Regression in CVXPY
```python
a = Variable(n)
b = Variable()

r = mul_elemwise(y, X*a - b)
obj = Minimize(sum_entries(logistic(r)))
Problem(obj).solve()
```

## Logistic Regression in CVXPY
\centering
\includegraphics[width=0.65\textwidth]{fig/logistic_reg.pdf}

## SpVC (for comparison)
\centering
\includegraphics[width=0.65\textwidth]{fig/sparse.pdf}

## Other Loss Functions
- many loss functions are available to modeler
- hard loss: $\ell_\mathrm{hard}(z) =
\begin{cases}
0 & z \geq 1 \\
+\infty & z < 1 \\
\end{cases}$
- exponential loss: $\ell_\mathrm{exp}(z) = \exp(-z)$
- quadratic loss: $\ell_2(z) = (1-z)_+^2$

## Other Loss Functions
\centering
\includegraphics[width=0.65\textwidth]{fig/losses.pdf}

## Unified Models
- many classification models fall into the form
$$
\begin{array}{ll}
\mbox{minimize} & r(a) + \rho \sum_{i=1}^N \ell\left[y_i\left(a^Tx_i - b\right) \right]
\end{array}
$$
- unified way to think about many of these models

## Unified Models
- linear separator feasibility problem: $\ell_\mathrm{hard}$, $r \equiv 0$
- MMC: $\ell_\mathrm{hard}$, $r(a) = \|a\|_2$
- SpVC: $\ell_h$, $r \equiv 0$
- SVC: $\ell_h$, $r(a) = \|a\|_2$
- Logistic regression: $\ell_L$, $r \equiv 0$
- "boosting": $\ell_{\mathrm{exp}}$, $r \equiv 0$
- many other options for modeling
    - loss for max violation instead of sum
    - sparse $a$ for feature selection
    - one-sided Huber loss for outliers?

# Nonlinear Separators
## Nonlinear Separators
\centering
\includegraphics[width=0.65\textwidth]{fig/non_lin_sep.pdf}

## Nonlinear Separators
- don't expect a linear separator to work
- to get nonlinear separators, we need to generalize our classification
function
$$
f(x) = a^T x + b
$$
- consider polynomials of $x \in \mathbf{R}^n$ of degree $d$:
$$
f(x) = \sum_{j_1 + \cdots + j_n \leq d} a_{j_1 \cdots j_n} x_1^{j_1} \cdots x_n^{j_n}
$$
- a little messy, but **still linear** in decision variable $a$

## Support Vector Machines
- follow the same setup as before: data $x_i$ with labels $y_i \in \lbrace +1,-1 \rbrace$
- positive and negative examples on opposite "sides" of the classification function
\begin{align*}
f_a(x_i) > 0 \mbox{ if } y_i = +1\\
f_a(x_i) < 0 \mbox{ if } y_i = -1
\end{align*}
which we simplify to
$$
y_if_a(x_i) > 0 
$$

## Support Vector Machines
- **quantify** our dislike of violations with **any** loss function we like
$$
\ell\left[y_i f_a(x_i)\right]
$$
- add regularization to get the same general set up as before:
$$
\begin{array}{ll}
\mbox{minimize}_a & r(a) + \rho \sum_{i=1}^N \ell\left[y_i f_a(x) \right]
\end{array}
$$
- not really different from linear classification problem before
- we've just expanded the number of **features** for each point by considering
polynomials
- **support vector machine** is usually $\ell_h$, $r(a) = \|a\|_2$

# Multiclass SVM
## Multiclass SVM
- what if you have more than 2 labels?
- example: handwritten digit classification

\centering
\includegraphics[width=0.55\textwidth]{fig/mnistExamples.png}

## Multiclass SVM Approaches
- one-vs-one
    - for each pair of classes, train an SVM
    - $\frac{K(K-1)}{2}$ problems!
    - for a new observation, choose most frequently predicted label among all SVMs
- one-vs-all
    - train $K$ SVMs: one single class vs. all others grouped
    - for new observation, choose label furthest away from separating hyperplane

## Single-model Multiclass SVM
- TODO
