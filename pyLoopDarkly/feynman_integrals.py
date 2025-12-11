"""
Tools for Feynman-parametric integrals
-------

Implementation routines for deriving and manipulating
Feynman-parametric integrals

"""

import math
import sympy as sp

class fcsp(sp.Function): # pylint: disable=invalid-name
    """Defines a symbolic scalar product of two 4-vectors

    fcsp(p1,q1) corresponds to p1.q1
    fcsp(p1) evaluates to p1.p1

    The function is symmetric in its arguments. However, linear
    combinations of 4-vectors are not expanded automatically. For
    that you need to replace fcsp with fcsp_expand.
    """
    @classmethod
    def eval(cls, *args):
        if len(args)==1:
            return cls(args[0], args[0])
        if len(args)==2:
            if sp.default_sort_key(args[0]) > sp.default_sort_key(args[1]):
                return cls(args[1], args[0])
        else:
            raise TypeError("Wrong number of arguments.")
        return None


class fcsp_expand(fcsp): # pylint: disable=invalid-name
    """Defines the expansion of a symbolic scalar product of two 4-vectors

       For example, fcsp_expand(2*p1-p2,p3) evaluates to 2*fcsp(p1,p3)-fcsp(p2,p3)
    """
    @classmethod
    def eval(cls, *args):
        if len(args)!=2:
            raise TypeError("Wrong number of arguments.")
        a,b=args[0],args[1]

        if isinstance(a, sp.Add) or isinstance(b, sp.Add):
            if isinstance(a, sp.Add):
                s1, s2 = a.as_two_terms()
                return(cls(s1,b)+cls(s2,b))
            if isinstance(b, sp.Add):
                s1, s2 = b.as_two_terms()
                return(cls(a,s1)+cls(a,s2))
        elif isinstance(a, sp.Mul):
            s1, s2 = a.as_two_terms()
            if s1.is_Integer and not s2.is_Integer:
                return s1*cls(s2,b)
        elif isinstance(b, sp.Mul):
            s1, s2 = b.as_two_terms()
            if s1.is_Integer and not s2.is_Integer:
                return s1*cls(a,s2)
        else:
            return fcsp(a,b)

def propagators_join(props : list,x='x') -> sp.core.expr:
    """Joins propagators of loop integrals by introducing Feynman parameters x_i

    Args:
        props (list): List of inverse propagators written in
        terms of fcsp containers for scalar products, e.g.
        [fcsp(k1,k1) - m1**2]

        x (str, optional): Suffix of the Feynman parameters. Defaults to 'x'.

    Raises:
        TypeError: if props is not a list
        TypeError: if x is not a str

    Returns:
        sp.core.expr: Returns a sum of propagators joined using Feynman parameters
    """
    if not isinstance(props, list):
        raise TypeError("props must be a list.")

    if not isinstance(x,str):
        raise TypeError("x must be a str.")

    res = 0
    for i,prop in enumerate(props):
        res+=sp.symbols(x+str(i+1))*prop

    return res.subs(fcsp,fcsp_expand)



def build_uf(old_u : sp.core.expr, old_f : sp.core.expr,
    lmom : sp.core.symbol.Symbol) -> (sp.core.expr,sp.core.expr):
    """Iteratively constructs Symanzik polynomials U and F by eliminating
    the current loop momentum lmom

    Args:
        old_u (sp.core.expr): U polynomial from a previous iteration
        old_f (sp.core.expr): F polynomial from a previous iteration
        lmom (sp.core.symbol.Symbol): current loop momentum to be eliminated

    Returns:
        (sp.core.expr,sp.core.expr): Returns a tuple of new U and F polynomials free of loop
        momentum lmom
    """
    a,b,c = sp.Wild('a'),sp.Wild('b'),sp.Wild('c')
    loopmark = sp.symbols('loopmark')
    half_sp = sp.Function('half_sp')

    # lmom.p -> loopmark*p
    tmp =old_f.replace(
    lambda arg: arg.func == fcsp and arg.args.count(lmom)==1,
    lambda arg: loopmark*half_sp(arg.args[0])*half_sp(arg.args[1]) #pylint: disable=not-callable
    ).subs(half_sp(lmom),1) # pylint: disable=not-callable

    # lmom.lmom -> loopmark^2
    tmp=tmp.subs(fcsp(lmom,lmom),loopmark**2)

    # At this point there is no lmom anymore

    if tmp.has(lmom):
        print(tmp)
        raise ValueError('Failed to eliminate the loop momentum ' + str(lmom))

    tmp = sp.Poly(tmp,loopmark)
    # la is proportional to terms containing lmom.lmom
    la = tmp.coeff_monomial(loopmark**2)
    # num is proportional to terms containing lmom.p
    num = tmp.coeff_monomial(loopmark)

    # now we square num, elmininating lmom
    num_num, num_den = num.as_numer_denom()

    num_num = sp.expand(num_num**2).replace(half_sp(a)**2, #pylint: disable=not-callable
        fcsp(a,a)).replace(c*half_sp(a)*half_sp(b),c*fcsp(a,b)) #pylint: disable=not-callable

    num = num_num/(num_den**2)

    j = tmp.coeff_monomial(1)


    return (sp.expand(sp.together(la*old_u)),sp.together(j-num/(4*la)))


def feynman_prepare(props: list, lmoms : list, **kwargs) -> (sp.core.expr,sp.core.expr):
    """Calculates Symanzik polynomials U and F from the given list of inverse
    propagators props and loop momenta lmoms

    Args:
        props (list): list of inverse propagators, e.g.
            props=[fcsp(k1,k1) - m1**2,fcsp(k1-p1,k1-p1) - m2**2,fcsp(k1-p2,k1-p2) - m3**2]
        lmoms (list): list of loop momemnta, e.g. [k1]
        fp_suffix (str, optional): Suffix of the Feynman parameters. Defaults to 'x'.

    Raises:
        TypeError: if props is not a list
        TypeError: if lmoms is not a list

    Returns:
        (sp.core.expr,sp.core.expr): Returns a tuple of the final U and F polynomials
    """
    if not isinstance(lmoms, list):
        raise TypeError('lmoms shoud be a list')
    if not isinstance(props, list):
        raise TypeError('props shoud be a list')

    fp_suffix=kwargs.get('fp_suffix','x')

    u_poly, f_poly = 1, propagators_join(props,x=fp_suffix)

    for l in lmoms:
        u_poly,f_poly=build_uf(u_poly, f_poly, l)

    fin_u,fin_f = u_poly, -sp.simplify(sp.together(u_poly*f_poly))
    return fin_u,sp.expand(fin_f)


def feynman_parametrize(props: list, pows: list, lmoms : list, **kwargs) -> (sp.core.expr,sp.core.expr,list):
    """Calculates Feynman parametrization of the given loop integral returning an integral
    in the Feynman parameters x_i

    Args:
        props (list): list of inverse propagators, e.g.
            props=[fcsp(k1,k1) - m1**2,fcsp(k1-p1,k1-p1) - m2**2,fcsp(k1-p2,k1-p2) - m3**2]
        lmoms (list): list of loop momemnta, e.g. [k1]
    Raises:
        TypeError: if props is not a list
        TypeError: if lmoms is not a list
        TypeError: if pows is not a list

    Returns:
        _type_: _description_
    """
    if not isinstance(lmoms, list):
        raise TypeError('lmoms shoud be a list')
    if not isinstance(props, list):
        raise TypeError('props shoud be a list')
    if not isinstance(pows, list):
        raise TypeError('pows shoud be a list')


    fp_suffix       = kwargs.get('fp_suffix','x')
    dim_d           = kwargs.get('dim',sp.symbols('d'))
    replace_dim     = kwargs.get('replace_dim',[])
    prefactor       = kwargs.get('prefactor','Multiloop1')
    extra_prefactor = kwargs.get('extra_prefactor',1)

    n_props         = len(pows)
    n_loops         = len(lmoms)
    f_pow           = sum(pows) - n_loops*dim_d/2
    inverse_measure = (sp.I*sp.pi**(dim_d/2))**n_loops

    prop_powers_hat         = [max(0,-math.floor(x)) for x in pows]
    prop_powers_tilde_raw   = [x+y for x,y in zip(pows,prop_powers_hat)]
    xpars_raw               = [sp.symbols('x'+str(i+1)) for i in range(n_props)]

    prefactor_val = 1
    if prefactor=='Unity':
        prefactor_val = inverse_measure
    elif prefactor=='Textbook':
        prefactor_val = inverse_measure/(2*sp.pi)**(dim_d*n_loops)
    elif prefactor=='Multiloop1':
        pass
    elif prefactor=='Multiloop2':
        prefactor_val = sp.exp(n_loops*sp.EulerGamma*(4-dim_d)/2)
    else:
        raise TypeError("Invalid value of the prefactor option.")

    # prop_powers_tilde contains no zero powers
    prop_powers_tilde = [x for x in prop_powers_tilde_raw if x!=0]

    #x_i linked to zero powers in prop_powers_tilde_raw
    zero_den_vars=[x[1] for x in zip(prop_powers_tilde_raw,xpars_raw) if x[0]==0]

    # xpars contains no x_i related to zero powers
    xpars=[x[1] for x in zip(prop_powers_tilde_raw,xpars_raw) if x[0]!=0]

    #x_i linked to nonzero powers in prop_powers_hat
    num_vars=[x[1] for x in zip(prop_powers_hat,xpars_raw) if x[0]!=0]

    prop_powers_hat = [x for x in prop_powers_hat if x!=0]

    fp_pref = 1
    for x,n in zip(xpars,prop_powers_tilde):
        fp_pref*= x**(n-1)


    u,f = feynman_prepare(props, lmoms, fp_suffix=fp_suffix)
    fp_int = u**(f_pow-dim_d/2)/f**f_pow

    # here we do the differentiation using num_vars and prop_powers_hat
    for x,n in zip(num_vars,prop_powers_hat):
        fp_int=(-1)**n*sp.diff(fp_int,x,n)

    for x in zero_den_vars:
        u,f,fp_int = u.subs(x,0), f.subs(x,0),fp_int.subs(x,0)

    fp_int = fp_pref*fp_int

    pref = extra_prefactor*sp.gamma(f_pow)/math.prod(map(sp.gamma,prop_powers_tilde))

    pref = pref*(-1)**sum(pows)*prefactor_val


    if replace_dim!=[]:
        fp_int=fp_int.subs(replace_dim[0],replace_dim[1])
        pref=pref.subs(replace_dim[0],replace_dim[1])

    fp_int = sp.powsimp(sp.factor(fp_int))

    return (fp_int,pref,xpars)
