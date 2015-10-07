% CME 252: Support Vector Machines
% AJ Friend \
  ICME, Stanford University

## Support Vector Machines
- find a hyperplane to separate data points into two classes
- use hyperplane to classify new (unseen) points

\centering
\includegraphics[width=0.6\textwidth]{fig/first.pdf}

## Scenarios
- assume data falls into one category:
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

## Linearly Separable Problem
- data: $x_i \in \reals^n$ with labels $y_i \in \lbrace +1, -1 \rbrace$ for $i = 1, \ldots, m$
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
- for all $i$, rewrite constraints as
$$
y_i\left(a^Tx_i - b\right) \geq 1
$$
- get **feasibility** problem
$$
\begin{array}{ll}
\mbox{minimize} & 0 \\
\mbox{subject to} & y_i\left(a^Tx_i - b\right) \geq 1 \mbox{ for } i = 1, \ldots, m
\end{array}
$$
with variables $a \in \reals^n$, $b \in \reals$
- has many potential separators

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