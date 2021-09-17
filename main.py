import string
from sympy import *
import sys


def gradSimple(p_exp, ppas, ppt, tolerance):
    """
    Effectue une descente de gradient simple a pas fixe
    :param p_exp: expression de la fonction sur laquel effectuer une descente de gradient
    :param ppas: float le pas fixe pour effectuer les differente itération
    :param ppt: [val1, .. , valn] point de depart de la descente de gradient
    :param tolerance: float seuil à partir duquel on decidra que l'aproximation est suffisante (par rapport a la norme
du point trouvé
    :return: [val1, .. , valn] point au plus proche du minimum local
    """
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    cond = Matrix(grad).subs(vec).norm()
    XK1 = Xk(vec, ppas, grad, size, pmod=1, pcond=cond)
    cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
    print(cond)
    while cond > tolerance:
        XK1 = Xk(vec, ppas, grad, size, pmod=1, pcond=cond)
        vec = list(zip(variables, XK1))
        cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
        print(cond)
    return XK1


def gradPOpti(p_exp, ppt, tolerance):
    """
    Effectue une descente de gradient simple a pas optimisé
    :param p_exp: expression de la fonction sur laquel effectuer une descente de gradient
    :param ppt: [val1, .. , valn] point de depart de la descente de gradient
    :param tolerance: float seuil à partir duquel on decidra que l'aproximation est suffisante (par rapport a la norme
du point trouvé
    :return: [val1, .. , valn] point au plus proche du minimum local
    """
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    expas = expPas(ppt, grad, vec, size)
    pas = pasOpti(p_exp, list(zip(variables, expas)), p)
    XK1 = Xk(vec, pas, grad, size)
    cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
    while cond > tolerance:
        vec = list(zip(variables, XK1))
        expas = expPas(ppt, grad, vec, size)
        pas = pasOpti(p_exp, list(zip(variables, expas)), p)
        if pas == -1:
            break
        XK1 = Xk(vec, pas, grad, size)
        cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
    return XK1


def gradFletcher(p_exp, ppt, tolerance):  # TODO adapter pour le moment seulement gradient a pas opti
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    expas = expPas(ppt, grad, vec, size)  # debut du calcul du pas opti
    pas = pasOpti(p_exp, list(zip(variables, expas)), p)
    XK1 = Xk(vec, pas, grad, size)
    cond = Matrix(XK1).norm()
    while cond > tolerance:  # calcul condition d'arret
        vec = list(zip(variables, XK1))
        expas = expPas(ppt, grad, vec, size)
        pas = pasOpti(p_exp, list(zip(variables, expas)), p)
        XK1 = Xk(vec, pas, grad, size)
        cond = Matrix(XK1).norm()
    return XK1


def gradPolak(p_exp, ppt, tolerance):  # TODO adapter pour le moment seulement gradient a pas opti
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    expas = expPas(ppt, grad, vec, size)  # debut du calcul du pas opti
    pas = pasOpti(p_exp, list(zip(variables, expas)), p)
    XK1 = Xk(vec, pas, grad, size)
    cond = Matrix(XK1).norm()
    while cond > tolerance:  # calcul condition d'arret
        vec = list(zip(variables, XK1))
        expas = expPas(ppt, grad, vec, size)
        pas = pasOpti(p_exp, list(zip(variables, expas)), p)
        XK1 = Xk(vec, pas, grad, size)
        cond = Matrix(XK1).norm()
    return XK1


def initForGrad(pexp, ppoint):
    """
    fonction permettant d'initialiser quelque variable necessaire au differente fonction de gradient
    :param pexp: expression de la fonction
    :param ppoint: premier point X0
    :return: liste des variable de pexp, nombre de variable, symbol p du pas, initialisation d'un gradient correspondant à pexp, [[(var1, val1), ...,(varn, valn)]
    """
    variables = pexp.free_symbols
    return variables, len(variables), Symbol('p'), [pexp.diff(var) for var in variables], list(zip(variables, ppoint))


def pasOpti(pexprpas, pvec, pp):
    """
    fonction permettant de calculer le pas optimal pour Xk+1
    :param pexprpas: expression de f
    :param pvec: vecteur trouver pour phi de p
    :param pp: symbole du pas
    :return: pas optimisé sinon -1 si fonction a échoué
    """
    pas = solve(pexprpas.subs(pvec), pp)  # fin calcul du pas opti
    res = -1
    for e in pas:
        if e > 0:
            res = e
            break
    return res


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
        res = [pvec[i][1] - ppas * pgrad[i].subs(pvec) for i in range(pdim)]
    else:
        res = [(pvec[i][1] - ppas * (pgrad[i].subs(pvec) / pcond)).evalf() for i in range(pdim)]
    return res


def printUsage(poption: string):
    chaine = "Usage : python main.py \"expression\" " + poption + "tolerance point"
    if poption == "-S":
        chaine += " pas"
        chaine += "\nExemple: python main.py \"(x - y)**2 + x**3 + y**3\" -S 0.5 \"10 10\" 0.25" + \
                  "\nWarning: le pas doit etre strictement superieur à zero"
    else:
        chaine += "\nExemple: python main.py \"(x - y)**2 + x**3 + y**3\" " + poption + " 0.5 \"10 10\""
    return chaine


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage : python main.py \"expression\" <arg>")
    else:
        res = -1
        f = parse_expr(sys.argv[1])
        if sys.argv[2] == "-S":
            if len(sys.argv) == 6:
                res = gradSimple(f, float(sys.argv[5]), list(map(float, list(sys.argv[4].split(" ")))),
                                 float(sys.argv[3]))
            else:
                print(printUsage(sys.argv))
        elif sys.argv[2] == "-O":
            if len(sys.argv) == 5:
                res = gradPOpti(f, list(map(float, list(sys.argv[4].split(" ")))), float(sys.argv[3]))
            else:
                print(printUsage(sys.argv))
        elif sys.argv[2] == "-F":
            if len(sys.argv) == 5:
                res = gradFletcher(f, list(map(float, list(sys.argv[4].split(" ")))), float(sys.argv[3]))
            else:
                print(printUsage(sys.argv))
        elif sys.argv[2] == "-P":
            if len(sys.argv) == 5:
                res = gradPolak(f, list(map(float, list(sys.argv[4].split(" ")))), float(sys.argv[3]))
            else:
                print(printUsage(sys.argv))
        else:
            print("Usage : python main.py \"expression\" <arg> tolerance point")
        if res != -1:
            variables = f.free_symbols
            vec = list(zip(variables, res))

            print("#######    RESULT    #######")
            print("point au plus proche du minimum local {}".format(f.subs(vec)))
            print("approximation trouvé : {}".format(vec))
