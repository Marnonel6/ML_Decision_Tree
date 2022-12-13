## Instructions

The coding and free response questions are due on October 13 at 11:59pm CDT.
There are 22 points possible for this assignment, but it's graded out of 20.
If you earn an average of 19 points on all five assignments, you'll get an A.
With the extra credit from here (and participation), it's possible to get an A
without completing the fifth assignment.

### Setup (2 points)

These 2 points are extra credit, but have an earlier deadline of October 6.
All you need to do for these points is pass the `test_setup` test case, which
just requires putting your NetID in the `netid` file and creating three PDF
files titled `XXX_qYYY.pdf` where `XXX` is replaced with your netid, and `YYY`
ranges from 1 to 3 (for the three free-response questions). The content of these
PDFs won't be graded, this is just to ensure that you can set up your
repository to be autograded.

Your final submission must also pass the `test_setup` test, or you will lose
these points.

### Coding (16 points)

You need to write code in every function in `src/` that raises a
`NotImplementedError` exception. Your code is graded automatically using the
test cases in `tests/`.  To see what your grade is going to be, you can run
`python -m pytest`; make sure you have installed the packages from
`requirements.txt` first. If the autograder says you get 100/100, it means you
get all 16 points.

For the numpy practice problems in `src/numpy_practice.py`, pay attention
to the docstring. The tests will make you implement it using certain
numpy functions, and will expect you to write each function in only
one or two lines of code.

The tests build on and sometimes depend on each other. We suggest that you
implement them in the order they appear in `tests/rubric.json`. That file
also allows you to see how many points each test is worth and which other
tests it may depend on. 

### Free response (4 points)

There are three free response questions. Your answer to each should be in its
own PDF file, titled `XXX_qYYY.pdf`, where `XXX` is replaced with your NetID
and `YYY` is the number of the question. So if your netid were `xyz0123`, the
answer to question 1 should be in a file named `xyz0123_q1.pdf`.  For questions
with multiple parts, put all parts in the single PDF and just clearly label
where each part begins.  Please *do not put your name in these PDFs* -- we will
grade your work anonymously using only your netid in the filenames.

## Free response questions

### Question 1 (1 point)

Suppose you have a deterministic function `f(X)` where `X` is an array of
`d` Boolean values, and `f` outputs a Boolean value that is some function of
those inputs. The function can be arbitrarily complicated, but for the same
input `x`, it always outputs the same value `f(x)`.

Suppose we collect a *lot* of data sampled from this function, such that we see
each of the possible `2 ^ d` values of `x` at least once. Assume our data
collection doesn't have any noise. 

Can a decision tree be built to perfectly represent `f(X)`? If so, give an
explanation or simple proof of why such a decision tree exists. If not, provide
an explanation or counterexample of a function that can't be represented by a
decision tree.

### Question 2 (1 point)

Suppose we want to learn a binary function `f(X)` that classifies whether a
picture contains a cat (`f(X) = 1`) or not (`f(X) = 0`). Our input
dataset is composed of 200 images, each represented as an array of 1024
pixels. Say we split the data into a training set of 100 images and a test
set of 100 images. Rather than using a machine learning algorithm to choose
our hypothesis `h(X)` that approximates the true function `f(X)`, we just decide
to randomly pick hypotheses until we find a good one.

We'll say two hypotheses `g(X)` and `h(X)` are *distinct* if they provide a
different labeling of the data. So if we only had one image, there's only
two distinct hypotheses: either you label the image as 1 or 0.

a.) How many distinct hypotheses are there for the entire dataset of 200
images?

b.) If you randomly choose one such hypothesis, what is the probability
that its labeling perfectly matches the 100 training images?

c.) Suppose we randomly choose a hypothesis and find out it does in fact
perfectly label the 100 training images. What is the probability that it
also perfectly labels the 100 test images?

### Question 3 (2 points)

The ID3 algorithm we implemented is relatively simple, in that it has
no hyperparameters (i.e., decisions about the algorithm that need to be
made before running it on a given dataset). Read the [scikit-learn
DecisionTreeClassifier documentation](
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
and read through some of the hyperparameters that the model allows
you to adjust. Pick two of the following hyperparameters:

- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `max_leaf_nodes`
- `min_impurity_decrease`

For each hyperparameter you choose, give a one-sentence explanation of what
it does to affect the kinds of trees that the algorithm learns.
Then, describe whether a small or large value for this hyperparameter
is more likely to cause the tree to *overfit*.

An example answer to this question for the `max_depth` hyperparameter,
(which you aren't allowed to use), might look like:

> The `max_depth` argument controls the maximum depth (longest path from the
> root to any leaf) of the tree. Using a large value of `max_depth` is more
> likely to overfit, because if the tree gets so deep that each leaf only
> corresponds to a single training example, we wouldn't expect it to generalize
> well to new unseen test data.
