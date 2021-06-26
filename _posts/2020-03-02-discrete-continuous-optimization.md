---
title: Discrete and Continuous Optimization Algorithms
date: 2020-03-02
permalink: /posts/2020/03/discrete-continuous-optimization/
tags:
  - Optimization
---

Optimization is a key process in machine learning, from which we can approach inference and learning. It allows us to decouple the mathematical specification of what we want to compute from the algorithms for how to compute it.

In total generality, optimization problems ask that you find the $x$ that lives in a constraint set $C$ that makes the function $F(x)$ as small as possible.

There are two types of optimization problems we'll consider: discrete optimization problems (mostly for inference) and continuous optimization problems (mostly for learning).

The following code is **based** on a lecture from Stanford by Percy Liang in **Stanford CS221: AI** course.

## Discrete Optimization

The process basically requires us to find the best discrete object
$$\min\limits_{p \in Paths} Cost(p)$$
and the **algorithmic** tool we'll be using is **Dynamic Programming**.

### Example Problem: Computing Edit Distance

Suppose we take as an **input**, two strings $s$ and $t$, we need to **output** the minimum number of character insertions, deletions, and substitutions it takes to change $s$ into $t$. For example:

| s        | t           | output |
| -------- | ----------- | ------ |
| "cat"    | "cat"       | 0      |
| "cat"    | "dog"       | 3      |
| "cat"    | "cats"      | 1      |
| "cat"    | "at"        | 1      |
| "a cat!" | "the cats!" | 4      |

### Solution

We should think of breaking down the problem into subproblems.

**Observation 1**: inserting into is equivalent to deleting a letter from (ensures subproblems get smaller).

**Observation 2**: perform edits at the end/start of strings.

Consider the last letter of $s$ and $t$. If these are the same, then we don't need to edit these letters, and we can proceed to the second-to-last letters. If they are different, then we have three choices.

**(i)** We can substitute the last letter of $s$ with the last letter of $t$.

**(ii)** We can delete the last letter of $s$.

**(iii)** We can insert the last letter of $t$ at the end of $s$.

In each of those cases, we can reduce the problem into a smaller problem. We simply try all of them and take the one that yields the minimum cost!

We can express this more formally with a mathematical recurrence. Before writing down the actual recurrence, the first step is to express the quantity that we wish to compute.

In this case: let $d(m, n)$ be the edit distance between the first $m$ letters of $s$ and the first $n$ letters of $t$. Then we have

$$
d(m,n) = \left\{
\begin{array}{ll}
      m & n = 0 \\
      n & m = 0 \\
      d(m-1, n-1) & s_m = t_n \\
      1 + \min\{d(m-1,n-1), d(m-1,n), d(m,n-1)\} & otherwise \\
\end{array}
\right.
$$

```python
def computeEditDistance(s, t):
    def recurse(m, n):
        """
        Return the minimum edit distance between:
        - first m letters of s
        - first n letters of t
        """
        if m == 0:  # Base case
            result = n
        elif n == 0:  # Base case
            result = m
        elif s[m - 1] == t[n - 1]:  # Last letter matches
            result = recurse(m - 1, n - 1)
        else:
            subCost = 1 + recurse(m - 1, n - 1)
            delCost = 1 + recurse(m - 1, n)
            insCost = 1 + recurse(m, n - 1)
            result = min(subCost, delCost, insCost)
        return result

    return recurse(len(s), len(t))

print(computeEditDistance('a cat!', 'the cats!'))
```

    4

Once we have the recurrence, we can code it up. However, the straightforward implementation will take exponential time. But we can memoize the results to make it quadratic time (in this case, $O(nm)$). The end result is the dynamic programming solution: recurrence + memoization.

```python
def computeEditDistance(s, t):
    cache = {}  # (m, n) => result
    def recurse(m, n):
        """
        Return the minimum edit distance between:
        - first m letters of s
        - first n letters of t
        """
        if (m, n) in cache:
            return cache[(m, n)]
        if m == 0:  # Base case
            result = n
        elif n == 0:  # Base case
            result = m
        elif s[m - 1] == t[n - 1]:  # Last letter matches
            result = recurse(m - 1, n - 1)
        else:
            subCost = 1 + recurse(m - 1, n - 1)
            delCost = 1 + recurse(m - 1, n)
            insCost = 1 + recurse(m, n - 1)
            result = min(subCost, delCost, insCost)
        cache[(m, n)] = result
        return result

    return recurse(len(s), len(t))

print(computeEditDistance('a cat!' * 10, 'the cats!' * 10))
```

    40

## Continuous Optimization

Find the best vector of real numbers
$$\min\limits_{\textbf{w} \in \mathbb{R}^d} TrainingError(\textbf{w})$$
and the **algorithmic** tool we'll be using is **Gradient Descent**.

### Example Problem: Finding the Least Squares Line

Suppose we take as an **input**, a set of pairs $\{(x_1, y_1),...,(x_n, y_n)\}$, we find, as an **output**, $w \in \mathbb{R}$ that minimizes the squared error
$$F(w) = \sum_{i=1}^{n} (x_i w-y_i)^2$$
For example:

|    pairs     |  w  |
| :----------: | :-: |
|    (2,4)     |  2  |
| (2,4), (4,2) |  ?  |

### Solution

We'll implement Linear regression, an important problem in machine learning. Posit a linear relationship $y=wx$. Now we get a set of training examples, each of which is a $(x_i, y_i)$ pair. The goal is to find the slope that best fits the data.

We would like an algorithm for optimizing general types of $F(w)$. So let's abstract away from the details. Start at a guess of $w$ (say $w=0$), and then iteratively update $w$ based on the derivative (gradient if $w$ is a vector) of $F(w)$. The algorithm we will use is called **gradient descent**.

If the derivative $F'(w) < 0$, then increase $w$; if $F'(w) > 0$, decrease $w$; otherwise, keep $w$ still. This motivates the following update rule, which we perform over and over again: $w \leftarrow w - \eta F'(w)$, where $\eta > 0$ is a **step size** that controls how aggressively we change $w$.

If $\eta$ is too big, then $w$ might bounce around and not converge. If $\eta$ is too small, then $w$ might not move very far to the optimum.

Now to specialize to our function, we just need to compute the derivative, which is an elementary calculus exercise:
$$ F'(w) = \sum\_{i=1}^{n} 2(x_i w-y_i)x_i$$

Finally the resultant code looks like the following

```python
points = [(2, 4), (4, 2)]

def F(w):
    return sum((w * x - y)**2 for x, y in points)

def dF(w):
    return sum(2*(w * x - y) * x for x, y in points)

# Gradient descent
w = 0
eta = 0.01
for t in range(100):
    value = F(w)
    gradient = dF(w)
    w = w - eta * gradient
    print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))
```

    iteration 0: w = 0.32, F(w) = 20
    iteration 1: w = 0.512, F(w) = 11.807999999999998
    iteration 2: w = 0.6272, F(w) = 8.858880000000001
    iteration 3: w = 0.69632, F(w) = 7.7971968
    iteration 4: w = 0.737792, F(w) = 7.4149908479999995
    iteration 5: w = 0.7626752, F(w) = 7.27739670528
    iteration 6: w = 0.77760512, F(w) = 7.227862813900801
    iteration 7: w = 0.786563072, F(w) = 7.210030613004288
    iteration 8: w = 0.7919378432, F(w) = 7.203611020681545
    iteration 9: w = 0.79516270592, F(w) = 7.201299967445356
    iteration 10: w = 0.797097623552, F(w) = 7.200467988280327
    iteration 11: w = 0.7982585741311999, F(w) = 7.200168475780918
    iteration 12: w = 0.79895514447872, F(w) = 7.200060651281129
    iteration 13: w = 0.799373086687232, F(w) = 7.200021834461207
    iteration 14: w = 0.7996238520123392, F(w) = 7.200007860406035
    iteration 15: w = 0.7997743112074035, F(w) = 7.200002829746172
    iteration 16: w = 0.799864586724442, F(w) = 7.200001018708621
    iteration 17: w = 0.7999187520346652, F(w) = 7.200000366735104
    iteration 18: w = 0.7999512512207991, F(w) = 7.2000001320246385
    iteration 19: w = 0.7999707507324795, F(w) = 7.200000047528869
    iteration 20: w = 0.7999824504394877, F(w) = 7.200000017110394
    iteration 21: w = 0.7999894702636926, F(w) = 7.200000006159741
    iteration 22: w = 0.7999936821582155, F(w) = 7.200000002217507
    iteration 23: w = 0.7999962092949293, F(w) = 7.200000000798303
    iteration 24: w = 0.7999977255769576, F(w) = 7.200000000287389
    iteration 25: w = 0.7999986353461745, F(w) = 7.200000000103461
    iteration 26: w = 0.7999991812077047, F(w) = 7.200000000037246
    iteration 27: w = 0.7999995087246229, F(w) = 7.200000000013408
    iteration 28: w = 0.7999997052347737, F(w) = 7.2000000000048265
    iteration 29: w = 0.7999998231408643, F(w) = 7.200000000001737
    iteration 30: w = 0.7999998938845185, F(w) = 7.2000000000006255
    iteration 31: w = 0.7999999363307111, F(w) = 7.200000000000226
    iteration 32: w = 0.7999999617984267, F(w) = 7.200000000000081
    iteration 33: w = 0.799999977079056, F(w) = 7.200000000000029
    iteration 34: w = 0.7999999862474336, F(w) = 7.20000000000001
    iteration 35: w = 0.7999999917484601, F(w) = 7.200000000000004
    iteration 36: w = 0.799999995049076, F(w) = 7.200000000000001
    iteration 37: w = 0.7999999970294456, F(w) = 7.199999999999999
    iteration 38: w = 0.7999999982176673, F(w) = 7.2
    iteration 39: w = 0.7999999989306004, F(w) = 7.2
    iteration 40: w = 0.7999999993583602, F(w) = 7.2
    iteration 41: w = 0.7999999996150161, F(w) = 7.199999999999999
    iteration 42: w = 0.7999999997690097, F(w) = 7.199999999999999
    iteration 43: w = 0.7999999998614058, F(w) = 7.2
    iteration 44: w = 0.7999999999168435, F(w) = 7.2
    iteration 45: w = 0.7999999999501061, F(w) = 7.199999999999999
    iteration 46: w = 0.7999999999700637, F(w) = 7.200000000000001
    iteration 47: w = 0.7999999999820382, F(w) = 7.199999999999999
    iteration 48: w = 0.7999999999892229, F(w) = 7.2
    iteration 49: w = 0.7999999999935338, F(w) = 7.2
    iteration 50: w = 0.7999999999961203, F(w) = 7.199999999999999
    iteration 51: w = 0.7999999999976721, F(w) = 7.2
    iteration 52: w = 0.7999999999986033, F(w) = 7.199999999999999
    iteration 53: w = 0.7999999999991619, F(w) = 7.200000000000001
    iteration 54: w = 0.7999999999994972, F(w) = 7.200000000000001
    iteration 55: w = 0.7999999999996984, F(w) = 7.200000000000001
    iteration 56: w = 0.7999999999998191, F(w) = 7.200000000000001
    iteration 57: w = 0.7999999999998915, F(w) = 7.2
    iteration 58: w = 0.7999999999999349, F(w) = 7.199999999999999
    iteration 59: w = 0.799999999999961, F(w) = 7.199999999999999
    iteration 60: w = 0.7999999999999766, F(w) = 7.2
    iteration 61: w = 0.799999999999986, F(w) = 7.199999999999999
    iteration 62: w = 0.7999999999999916, F(w) = 7.199999999999999
    iteration 63: w = 0.7999999999999949, F(w) = 7.2
    iteration 64: w = 0.7999999999999969, F(w) = 7.2
    iteration 65: w = 0.7999999999999982, F(w) = 7.199999999999999
    iteration 66: w = 0.7999999999999989, F(w) = 7.200000000000001
    iteration 67: w = 0.7999999999999994, F(w) = 7.2
    iteration 68: w = 0.7999999999999996, F(w) = 7.2
    iteration 69: w = 0.7999999999999997, F(w) = 7.2
    iteration 70: w = 0.7999999999999998, F(w) = 7.199999999999999
    iteration 71: w = 0.7999999999999999, F(w) = 7.2
    iteration 72: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 73: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 74: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 75: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 76: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 77: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 78: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 79: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 80: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 81: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 82: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 83: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 84: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 85: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 86: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 87: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 88: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 89: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 90: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 91: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 92: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 93: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 94: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 95: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 96: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 97: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 98: w = 0.7999999999999999, F(w) = 7.200000000000001
    iteration 99: w = 0.7999999999999999, F(w) = 7.200000000000001

From which we find that the best value of $w$ to minimize $F(w)$ is when $w \approx 0.8$.
