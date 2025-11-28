
import pytest
import sympy as sp
import pyLoopDarkly as pld


def test_fcsp_symmetry():
    a,b = sp.symbols('a b')

    assert pld.fcsp(b,a) == pld.fcsp(a,b)

def test_fcsp_expansion():
    p1,p2,k1,k2 = sp.symbols('p1 p2 k1 k2')

    assert pld.fcsp_expand(2*p1+p2,k1+k2) == 2*pld.fcsp(k1, p1) + pld.fcsp(k1, p2) + 2*pld.fcsp(k2, p1) + pld.fcsp(k2, p2)


def test_feynman_prepare_tadpole_1l():
    k1,m1,x1 = sp.symbols('k1 m1 x1')

    assert pld.feynman_prepare([pld.fcsp(k1)-m1**2],[k1]) == (x1, m1**2*x1**2)

def test_feynman_prepare_bubble_1l():
    k1, p1, m1, m2, x1, x2 = sp.symbols('k1 p1 m1 m2 x1 x2')

    assert pld.feynman_prepare([pld.fcsp(k1)-m1**2,pld.fcsp(k1+p1)-m2**2],[k1]) == (x1 + x2,
    m1**2*x1**2 + m1**2*x1*x2 + m2**2*x1*x2 + m2**2*x2**2 - x1*x2*pld.fcsp(p1, p1))


def test_feynman_prepare_bubble_massless_2l():
    k1, k2, q1, x1,x2,x3,x4,x5 = sp.symbols('k1 k2 q1 x1 x2 x3 x4 x5')

    assert pld.feynman_prepare([pld.fcsp(k1),pld.fcsp(k2),pld.fcsp(q1-k1-k2),pld.fcsp(q1-k1),pld.fcsp(q1-k2)],[k1,k2]) == (x1*x2 + x1*x3 + x1*x5 + x2*x3 + x2*x4 + x3*x4 + x3*x5 + x4*x5,
 -x1*x2*x3*pld.fcsp(q1, q1) - x1*x2*x4*pld.fcsp(q1, q1) - x1*x2*x5*pld.fcsp(q1, q1) - x1*x3*x4*pld.fcsp(q1, q1) - x1*x4*x5*pld.fcsp(q1, q1) - x2*x3*x5*pld.fcsp(q1, q1) - x2*x4*x5*pld.fcsp(q1, q1) - x3*x4*x5*pld.fcsp(q1, q1))
