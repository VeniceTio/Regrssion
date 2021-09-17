from sympy import *
import sys


def gradSimple(p_exp, ppas, ppt, tolerance):
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    cond = Matrix(grad).subs(vec).norm()
    XK1 = Xk(vec, ppas, grad, size, pmod=1, pcond=cond)
    print(XK1)
    cond = Matrix(XK1).norm()
    while cond > tolerance:
        XK1 =Xk(vec, ppas, grad, size, pmod=1, pcond=cond)
        vec = list(zip(variables, XK1))
        cond = Matrix(XK1).norm()
    return XK1


def gradPOpti(p_exp, ppt, tolerance):
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    expas = expPas(ppt, grad, vec, size) # debut du calcul du pas opti
    pas = pasOpti(p_exp, list(zip(variables, expas)), p)
    XK1 = Xk(vec, pas, grad, size)
    cond = Matrix(XK1).norm()
    while cond > tolerance:                     #calcul condition d'arret
        vec = list(zip(variables, XK1))
        expas = expPas(ppt, grad, vec, size)
        pas = pasOpti(p_exp, list(zip(variables, expas)), p)
        XK1 = Xk(vec, pas, grad, size)
        cond = Matrix(XK1).norm()
    return XK1


def initForGrad(pexp, ppoint):
    variables = pexp.free_symbols
    return variables, len(variables), Symbol('p'), [pexp.diff(var) for var in variables], list(zip(variables, ppoint))


def pasOpti(pexprpas, pvec, pp): #TODO: selction du choix parmi ceux sortie
    pas = solve(pexprpas.subs(pvec), pp)         #fin calcul du pas opti
    return pas

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


def Xk(pvec, ppas, pgrad, pdim, pmod=0, pcond=1):
    """
    renvoie le point X k+1 pour une regression
    :param pvec: [(var1, val1), ...,(varn, valn)] tableau qui associe au valeur de ppnt le nom de la variable à laquel
    :param ppas: indique le pas necessaire pour l'iteration suivante
    :param pgrad: [derivepartielEnVar1, .. , derivepartielEnVarn] gradient de la fonction d'origine
    :param pdim: n dimension du vecteur de la fonction
    :return: [val1, .. , valn] point à l'iteration k+1 soit Xk+1
    """
    if pmod == 0:
        res = [pvec[i][1] - ppas[0] * pgrad[i].subs(pvec) for i in range(pdim)]
    else:
        res = [(pvec[i][1] - ppas * (pgrad[i].subs(pvec)/pcond)).evalf() for i in range(pdim)]
    return res


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage : python main.py expression pas")
    else:
        f = parse_expr(sys.argv[1])
        res = gradPOpti(f, [10, 10], 0.5)
        variables = f.free_symbols
        vec = list(zip(variables, res))

        print(f.subs(vec))
        print("resultat : {}".format(res))
