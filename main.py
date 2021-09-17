from sympy import *
import sys


def gradSimple(p_exp, ppas, ppt, tolerance):
    variables = p_exp.free_symbols
    grad = []
    XK1 = []
    vec = []
    for var in variables:
        XK1.append(0)
        grad.append(p_exp.diff(var))
    print(variables)
    print(ppt)
    vec = list(zip(variables, ppt))
    print(vec)
    cond = Matrix(grad).subs(vec).norm()
    for i in range(len(XK1)):
        XK1[i] = (ppt[i] - ppas * ((grad[i].subs(vec))/cond)).evalf()
    print(XK1)
    u = Matrix(XK1)
    cond = u.norm()
    j = 0
    while cond > tolerance:
        for i in range(len(XK1)):
            XK1[i] = (vec[i][1] - ppas * (grad[i].subs(vec)/cond)).evalf()
        vec = list(zip(variables, XK1))
        print(vec)
        u = Matrix(XK1)
        cond = u.norm()
        print(cond)
        j += 1
    print(XK1)
    return XK1


def gradPOpti(p_exp, ppt, tolerance):
    variables = p_exp.free_symbols
    size = len(variables)
    grad = []
    for var in variables:
        grad.append(p_exp.diff(var))
    vec = list(zip(variables, ppt))
    expas = expPas(ppt, grad, vec, size) # debut du calcul du pas opti
    vec2 = list(zip(variables, expas))
    p = Symbol('p')
    pas = solve(p_exp.subs(vec2), p)         #fin calcul du pas opti
    XK1 = Xk(vec, pas, grad, size)
    vec = list(zip(variables, XK1))
    u = Matrix(XK1)                          #calcul condition d'arret
    cond = u.norm()
    while cond > tolerance:
        expas = expPas(ppt, grad, vec, size)
        vec2 = list(zip(variables, expas))
        pas = solve(p_exp.subs(vec2), p)
        XK1 = Xk(vec, pas, grad, size)
        vec = list(zip(variables, XK1))
        u = Matrix(XK1)
        cond = u.norm()
    return XK1


def expPas(ppnt, pgrad, pvec, pdim):
    """
    expPas renvoie l'expression de phi en fonction de p soit un vecteur de même dimension que pvec. Cette expression
    sert par la suite à calculer le pas optimal

    :param ppnt: [val1, .. , valn] point à partir duquel le pas optimal
    :param pgrad: [derivepartielEnVar1, .. , derivepartielEnVarn] gradient de la fonction d'origine
    :param pvec: [(var1, val1), ...,(varn, valn)] tableau qui associe au valeur de ppnt le nom de la variable à laquel
    elle est associé
    :param pdim: n dimension du vecteur de la fonction
    :return: expression de phi de p
    """
    return [parse_expr(str(ppnt[i]) + " - p * " + str(pgrad[i].subs(pvec))) for i in range(pdim)]


def Xk(pvec, ppas, pgrad, pdim):
    """
    renvoie le point X k+1 pour une regression
    :param pvec: [(var1, val1), ...,(varn, valn)] tableau qui associe au valeur de ppnt le nom de la variable à laquel
    :param ppas: indique le pas necessaire pour l'iteration suivante
    :param pgrad: [derivepartielEnVar1, .. , derivepartielEnVarn] gradient de la fonction d'origine
    :param pdim: n dimension du vecteur de la fonction
    :return: [val1, .. , valn] point à l'iteration k+1 soit Xk+1
    """
    return [pvec[i][1] - ppas[0] * pgrad[i].subs(pvec) for i in range(pdim)]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage : python main.py expression pas")
    else:
        f = parse_expr(sys.argv[1])
        res = gradSimple(f, 0.25, [10, 10], 0.5)
        variables = f.free_symbols
        vec = list(zip(variables, res))

        print(f.subs(vec))
        print("resultat : {}".format(res))
