% CME 252: Homework 0 \
  Due: Not Graded
% AJ Friend

We'll use this homework to get everyone set up with
the Python environment you'll use for the whole class,
and do a test run for homework submission.

# Part 1: Python Environment
We'll want to get the following things working:

- Python interpreter
- "SciPy stack" python packages:
    - `numpy`
    - `scipy`
    - `matplotlib`
    - `ipython`
- `cvxpy`

It doesn't matter **how** you get these working, as long as you do. I'll outline **one** possible path to get these working below.

As a quick test, you can skip this part if you can download, open on your own computer, and run all the cells in the example [Time Series Smoothing notebook](http://nbviewer.ipython.org/github/icme/cme252-optimization/blob/master/lectures/intro/smooth.ipynb). If you have any trouble,
this homework should help. We can also iron out installation issues
in class, in office hours, or on [Piazza](http://piazza.com/stanford/fall2015/cme252).

## Step 1. Install Anaconda Python distribution
Anaconda is a Python distribution that includes Python itself, and many useful packages (including the ones we need: `numpy`, `scipy`, `matplotlib`, and `ipython`). Installing Anaconda is probably the easiest way (especially for beginners) to get a working setup.

- Go to <https://www.continuum.io/downloads>, download the installer for **Python 2.7** for your operating system, and run it. The code we use in this class will probably work for Python 3.4 (and you can complain to me if it doesn't), but I'll be using 2.7, so that's your safest bet.
- Open a **new** terminal window and run `conda info` to ensure that everything is working.

## Step 2. Install `cvxpy`
- Run `pip install cvxpy==0.2.28`. We're avoiding the very latest version of `cvxpy` because it seems to be causing installation issues. If you can get the latest version working with `pip install cvxpy`, that's great, too.

## Step 3. Check that everything is working
- Download the [Time Series Smoothing notebook](http://nbviewer.ipython.org/github/icme/cme252-optimization/blob/master/lectures/intro/smooth.ipynb) to some directory on your computer.
- In a terminal window, and in the same directory where you downloaded the notebook, type `ipython notebook`. This should open up a new browser window.
- Locate `smooth.ipynb` in the list of files and open it.
- You should be looking at an IPython (recently renamed to "Jupyter") notebook. It's just an interactive Python interpreter that allows you to split code into cells and execute them independently. It also allows for inline plotting of images, and inclusion of descriptive text (and LaTex for math).
I find it useful both for experimenting with code, and for explaining concepts. I'll give many examples using notebooks throughout the class.
- Execute the cells, from first to last, to make sure that `numpy`, `matplotlib`, and `cxvpy` are all working. On my computer, I execute a cell with `Shift-Enter`. A cell is busy computing an answer if there is an asterisk (\*) next to it.
- If you got pretty plots and no error messages, then everything works! If you have any trouble, please ask on Piazza, in class, or in office hours.

# Part 2: Homework Submission

Follow this Dropbox File request link: <https://www.dropbox.com/request/KoUUxj6Oy9j2gNh98hZ8>
