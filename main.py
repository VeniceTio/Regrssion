"""
TP DESCENTE DE GRADIENT
Author:  LOUAZEL Yoann, OURO-AGORO Shrafdine, SERGENT Olaf-Marie

Tout les algorithme semble fonctionner sauf la variante Polak-ribieres
seul les points de type (10,10) (8, 8) ou x = y semble permetre la convergence.
ce probleme semble provennir de l'implementation de phi qui n'a pas été implémenté celon la dérivé de phi
"""
import string
from sympy import *
import sys


def gradSimple(p_exp, ppas, ppt, tolerance, pverbose=0):
    """
    Effectue une descente de gradient simple a pas fixe
    :param p_exp: expression de la fonction sur laquel effectuer une descente de gradient
    :param ppas: float le pas fixe pour effectuer les differente itération
    :param ppt: [val1, .. , valn] point de depart de la descente de gradient
    :param tolerance: float seuil à partir duquel on decidra que l'aproximation est suffisante (par rapport a la norme
du point trouvé
    :param pverbose: int par defaut à 0 et si different de 0 alors le mode verbose est activé
    :return: [val1, .. , valn] point au plus proche du minimum local
    """
    j = 1
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    cond = Matrix(grad).subs(vec).norm()
    XK1 = Xk(vec, ppas, grad, size, pmod=1, pcond=cond)
    cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
    while cond > tolerance:
        if pverbose == 1:
            print("X{} : {}".format(j, XK1))
            j += 1
        XK1 = Xk(vec, ppas, grad, size, pmod=1, pcond=cond)
        vec = list(zip(variables, XK1))
        cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
    return XK1


def gradPOpti(p_exp, ppt, tolerance, pverbose=0):
    """
    Effectue une descente de gradient a pas optimisé
    :param p_exp: expression de la fonction sur laquel effectuer une descente de gradient
    :param ppt: [val1, .. , valn] point de depart de la descente de gradient
    :param tolerance: float seuil à partir duquel on decidra que l'aproximation est suffisante (par rapport a la norme
du point trouvé
    :param pverbose: int par defaut à 0 et si different de 0 alors le mode verbose est activé
    :return: [val1, .. , valn] point au plus proche du minimum local
    """
    j = 1
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    expas = expPas(ppt, grad, vec, size)
    pas = pasOpti(p_exp, list(zip(variables, expas)), p)
    XK1 = Xk(vec, pas, grad, size)
    vec = list(zip(variables, XK1))
    cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
    while cond > tolerance:
        if pverbose == 1:
            print("X{} : {}".format(j, XK1))
            j += 1
        expas = expPas(ppt, grad, vec, size)
        pas = pasOpti(p_exp, list(zip(variables, expas)), p)
        if pas == -1:
            break
        XK1 = Xk(vec, pas, grad, size)
        vec = list(zip(variables, XK1))
        cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
    return XK1


def gradFletcher(p_exp, ppt, tolerance, pverbose=0):
    """
        Effectue une descente de gradient celon la variante de Fletcher Reeves
        :param p_exp: expression de la fonction sur laquel effectuer une descente de gradient
        :param ppt: [val1, .. , valn] point de depart de la descente de gradient
        :param tolerance: float seuil à partir duquel on decidra que l'aproximation est suffisante (par rapport a la norme
    du point trouvé
        :param pverbose: int par defaut à 0 et si different de 0 alors le mode verbose est activé
        :return: [val1, .. , valn] point au plus proche du minimum local
        """
    j = 1
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    expas = expPas(ppt, grad, vec, size)
    pas = pasOpti(p_exp, list(zip(variables, expas)), p)
    XK1 = Xk(vec, pas, grad, size)
    vecm1 = vec
    vec = list(zip(variables, XK1))
    Bk = (Matrix([grad[i].subs(vec) for i in range(size)]).norm() ** 2) / (
                Matrix([grad[i].subs(vecm1) for i in range(size)]).norm() ** 2)
    dk1 = [- grad[i].subs(vec) - Bk * grad[i].subs(vecm1) for i in range(size)]
    while Matrix([grad[i].subs(vec) for i in range(size)]).norm() > tolerance:
        if pverbose == 1:
            print("###\nX{0} : {1}\n B{0} : {2}\n D{3} : {4}".format(j, XK1, Bk, j+1, dk1))
            j += 1
        expas = expPas(ppt, grad, vec, size)
        pas = pasOpti(p_exp, list(zip(variables, expas)), p)
        XK1 = Xk(vec, pas, dk1, size, pmod=2)
        vecm1 = vec
        vec = list(zip(variables, XK1))
        Bk = (Matrix([grad[i].subs(vec) for i in range(size)]).norm() ** 2) / (
                Matrix([grad[i].subs(vecm1) for i in range(size)]).norm() ** 2)
        dk1 = [- grad[i].subs(vec) - Bk * grad[i].subs(vecm1) for i in range(size)]
    return XK1


def gradPolak(p_exp, ppt, tolerance, pverbose=0):
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)

    # point d'arrêt
    pointArret = solve(grad, variables)

    cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()

    while Matrix([grad[i].subs(vec) for i in range(size)]).norm() > tolerance:
        expas = expPas(ppt, grad, vec, size)
        pas = pasOpti(p_exp, list(zip(variables, expas)), p)
        XK1 = Xk(vec, pas, grad, size)

        # point correspondant
        pXK1 = []
        i = 0
        for variable in variables:
            pXK1.append((variable, XK1[i]))
            i += 1

        # calcul de beta
        g = [p_exp.diff(var).subs(pXK1) for var in variables]
        h = [(p_exp.diff(var).subs(pXK1) - p_exp.diff(var).subs(vec))
             for var in variables]
        k = [p_exp.diff(var).subs(vec) for var in variables]
        beta = Matrix(g).transpose() * Matrix(h) / \
               Matrix(k).norm() * Matrix(k).norm()

        # direction suivante
        dK1 = [((-1) * p_exp.diff(var).subs(pXK1) - beta[0] * p_exp.diff(var).subs(vec))
               for var in variables]

        # setting prochaine itération
        vec = pXK1
        print(vec)
    return vec


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
    elif pmod == 2:
        res = [pvec[i][1] + ppas * pgrad[i] for i in range(pdim)]
    else:
        res = [(pvec[i][1] - ppas * (pgrad[i].subs(pvec) / pcond)).evalf() for i in range(pdim)]
    return res


def printUsage(poption):
    """
    construit la phrase renvoyé en cas d'erreur de saisi
    :param poption: option d'usage saisie
    :return: chaine renvoyé spécifique à chaque option possible
    """
    chaine = "Usage : python main.py \"expression\" " + str(poption) + " tolerance point"
    if poption == "-S":
        chaine += " pas"
        chaine += "\nExemple: python main.py \"(x - y)**2 + x**3 + y**3\" -S 0.5 \"10 10\" 0.25" + \
                  "\nWarning: le pas doit etre strictement superieur à zero"
    else:
        chaine += "\nExemple: python main.py \"(x - y)**2 + x**3 + y**3\" " + str(poption) + " 0.5 \"10 10\""
    return chaine


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage : python main.py \"expression\" <arg> \n - argument possible: -S -O -F -P")
    else:
        res = -1
        ver = 0
        f = parse_expr(sys.argv[1])
        if sys.argv[2] == "-S":
            if len(sys.argv) == 6 or len(sys.argv) == 7:
                if len(sys.argv) == 7 and sys.argv[6] == "-v":
                    ver = 1
                res = gradSimple(f, float(sys.argv[5]), list(map(float, list(sys.argv[4].split(" ")))),
                                 float(sys.argv[3]), ver)
            else:
                print(printUsage(sys.argv[2]))
        elif sys.argv[2] == "-O":
            if len(sys.argv) == 5 or len(sys.argv) == 6:
                if len(sys.argv) == 6 and sys.argv[5] == "-v":
                    ver = 1
                res = gradPOpti(f, list(map(float, list(sys.argv[4].split(" ")))), float(sys.argv[3]), ver)
            else:
                print(printUsage(sys.argv[2]))
        elif sys.argv[2] == "-F":
            if len(sys.argv) == 5 or len(sys.argv) == 6:
                if len(sys.argv) == 6 and sys.argv[5] == "-v":
                    ver = 1
                res = gradFletcher(f, list(map(float, list(sys.argv[4].split(" ")))), float(sys.argv[3]), ver)
            else:
                print(printUsage(sys.argv[2]))
        elif sys.argv[2] == "-P":
            if len(sys.argv) == 5 or len(sys.argv) == 6:
                if len(sys.argv) == 6 and sys.argv[5] == "-v":
                    ver = 1
                res = gradPolak(f, list(map(float, list(sys.argv[4].split(" ")))), float(sys.argv[3]), ver)
            else:
                print(printUsage(sys.argv[2]))
        else:
            print("Usage : python main.py \"expression\" <arg> tolerance point")
        if res != -1:
            variables = f.free_symbols
            vec = list(zip(variables, res))
            print("#######    RESULT    #######")
            print("point au plus proche du minimum local {}".format(f.subs(vec)))
            print("approximation trouvé : {}".format(vec))
