---
layout: post
title: "Broadcasting"
date: 2022-04-24 10:09:00 +0100
categories: broadcasting
#permalink: /seq2seq
#categories: encoder-decoder keras
#permalink: CoRec
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Introduction

Broadcasting is the approach that Numpy and other libraries follow when performing arithmetical operations with arrays of different dimensions. Broadcast arrays are treated as if they had the same shapes: e.g. $$3*[1,2,3]$$ yields $$[3,6,9]$$ as if the scalar element were a vector of size 3. It allows for vectorization of operations, which translate to faster loops in C, without useless redundancy of data.

# Main rule

Broadcasting can only happen when, while analyzing two arrays' dimensions from the **rightmost** one, two conditions are satisfied:

- one of the dimensions is one
- they are equal

```python
# x      (4d array):  12 x 1 x 4 x 5
# y      (3d array):      6 x 4 x 1
# Result (4d array):  12 x 6 x 4 x 5

x = np.zeros((12, 1, 4, 5))
y = np.ones((6, 4, 1))

x+y
```

When one dimension is one, the greater one is used, and the smaller one is stretched to behave like an array of repeated elements of the size equal to the greater dimension. E.g. in the previous example, each `y[:,:,0]` is broadcast (repeated) to each of the 5 repetitions of the rightmost dimension, thanks to `x`. This expansion of dimension is mostly an abstraction, since internally NumPy doesn't actually make copies of the same data, but handles it efficiently.

If these conditions are not met, an error occurs, e.g. NumPy throws a _ValueError: operands could not be broadcast together_ exception.

```python
# x      (1d array):  3
# y      (1d array):  4
# Result ():  ?

x = np.array([1,2,3])

y = np.array([3,4,5,6])

x+y
```

```python
# x      (2d array):  2 x 3
# y      (2d array):  3 x 3
# Result ():  ?

x = np.array([[1,2,3], [4,5,6]])

y = np.array([[1,2,3], [4,5,6], [7,8,9]])

x*y
```

# Conclusion

Broadcasting semantics is used everywhere, and its rules are not implemented only in NumPy, but in other libraries that follow the same rationale, like PyTorch with tensors.
