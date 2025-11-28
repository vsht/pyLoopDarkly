pyLoopDarkly is an attempt to write a useful library for manipulating
multiloop integrals and topologies that does not require commerical software
such as Mathematica and Maple.

Currently it is still in a very early development stage

# Motto

" ... I see only private codes. Collaborators only, can't share with anyone else. I hope, for physics's sake, we can do better. Because, if everyone hides their codes, the way I myself do, then we will never advance and fail to discover new physics, and we'll end up ignorant this way, understanding very little and still getting it wrong."

# Installing
pyLoopDarkly is not on PyPy yet. But you can do

```bash
pip install git+https://github.com/vsht/pyLoopDarkly
```
# Examples

```python
import sympy as sp
p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = sp.symbols('p1:11')
q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = sp.symbols('q1:11')
k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = sp.symbols('k1:11')
m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = sp.symbols('m1:11')
import pyLoopDarkly as pld

# U,F polynomails for a massive 1L bubble
pld.feynman_prepare([pld.fcsp(k1)-m1**2,pld.fcsp(k1+p1)-m2**2],[k1])
# (x1 + x2, m1**2*x1**2 + m1**2*x1*x2 + m2**2*x1*x2 + m2**2*x2**2 - x1*x2*fcsp(p1, p1))

# U,F polynomails for a massless 2L bubble
pld.feynman_prepare([pld.fcsp(k1),pld.fcsp(k2),pld.fcsp(q1-k1-k2),pld.fcsp(q1-k1),pld.fcsp(q1-k2)],[k1,k2])
# (x1*x2 + x1*x3 + x1*x5 + x2*x3 + x2*x4 + x3*x4 + x3*x5 + x4*x5, 
# -x1*x2*x3*fcsp(q1, q1) - x1*x2*x4*fcsp(q1, q1) - x1*x2*x5*fcsp(q1, q1) - 
# x1*x3*x4*fcsp(q1, q1) - x1*x4*x5*fcsp(q1, q1) - x2*x3*x5*fcsp(q1, q1) - 
# x2*x4*x5*fcsp(q1, q1) - x3*x4*x5*fcsp(q1, q1))
```
