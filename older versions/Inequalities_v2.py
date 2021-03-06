# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:35:53 2020

@author: boraulu
"""

import importlib
from igraph import *
import numpy as np
import time
import mosek
from itertools import *
import sympy as sy
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix


def ReloadInflation():
    import importlib
    try:
        importlib.reload(inflation)
    except NameError:
        import InitializeInflation as inflation


DataInstrumental = [0.5, 0, 0, 0, 0, 0.5, 0, 0]
TriangleData = [0.12199995751046305, 0.0022969343799089472, 0.001748319476328954, 3.999015242496535e-05,
                0.028907881434196828, 0.0005736087488455967, 0.0003924033706699725, 1.1247230369521505e-05,
                0.0030142577390317635, 0.09234476010282468, 4.373922921480586e-05, 0.0014533921021948346,
                0.0007798079722868244, 0.024091567451515063, 1.1247230369521505e-05, 0.0003849052170902915,
                0.020774884184769502, 0.000396152447459813, 0.0003049249122403608, 4.998769053120669e-06,
                0.10820335492385, 0.0020794879260981982, 0.0015546171755205281, 2.4993845265603346e-05,
                0.0006260958239033638, 0.020273757587194154, 7.498153579681003e-06, 0.0003374169110856452,
                0.0028942872817568676, 0.08976414557915113, 2.624353752888351e-05, 0.0012984302615480939,
                0.002370666223442477, 4.7488306004646356e-05, 0.0999928767540993, 0.001957018084296742,
                0.0006198473625869629, 8.747845842961171e-06, 0.02636975644747481, 0.0005198719815245496,
                1.4996307159362007e-05, 0.000403650601039494, 0.0005498645958432735, 0.017359475229224805,
                7.123245900696953e-05, 0.002346922070440154, 0.0033754188031197316, 0.10295964618712641,
                0.00038740460161685187, 7.498153579681003e-06, 0.01608353942841575, 0.000306174604503641,
                0.0021319750011559654, 4.248953695152569e-05, 0.09107007399427891, 0.001860791780024169,
                5.998522863744803e-05, 0.0018395470115484063, 0.002570616985567304, 0.0766411271224461,
                1.874538394920251e-05, 0.00048238121362614454, 0.0006410921310627258, 0.020223769896662948]

inflation_order = 2
card = 4
TriangleGraph = Graph.Formula("X->A:C,Y->A:B,Z->B:C")
EvansGraph = Graph.Formula("U3->A:C:D,U2->B:C:D,U1->A:B,A->C,B->D")
InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")

ReloadInflation()

try:
    importlib.reload(inflation)
except NameError:
    import InitializeInflation as inflation

obs_count, num_vars, exp_set, group_elem, det_assumptions, names = inflation.LearnInflationGraphParameters(
    TriangleGraph, inflation_order)

# valid_column_orbits=inflation.ValidColumnOrbitsFromGraph(TriangleGraph,inflation_order,card)
# SpMatrix=inflation.SparseInflationMatrix(obs_count, num_vars, valid_column_orbits, exp_set, inflation_order, card)

SpMatrix = inflation.InflationMatrixFromGraph(TriangleGraph, 2, 4)
b = inflation.FindB(TriangleData, 2)

b = inflation.FindB(TriangleData, inflation_order)

Sol, epsilon = inflation.InflationLP(SpMatrix, b)


def Compatibility(Sol, b, epsilon):
    y = np.array(Sol)

    Comp = y.T.dot(b)

    if Comp + abs(epsilon) < 0:
        Comp = False
        print('Distribution Compatibility Status: INCOMPATIBLE')
    else:
        Comp = True
        print('Distribution Compatibility Status: COMPATIBLE')

    return Comp


def Epsilon(y, n):
    ymax = max(y)
    denom = 1000 * n
    epsilon = ymax / denom
    print(denom)
    return epsilon


def ValidityCheck(y, SpMatrix, yreal):
    # DO NOT LEAVE SPARSITY!!
    checkY = y.T * SpMatrix
    Yreal = yreal.T * SpMatrix
    if np.amin(checkY) >= np.amin(Yreal):

        Validity = True

    else:

        Validity = False

    return Validity


def yZeroFilter(Sol, n):
    y = np.array(Sol)
    y[np.abs(y) < Epsilon(y, n)] = 0

    return y


def yRoundFilter(y, n):
    epsilon = Epsilon(y, n)
    order = -np.floor(np.log10(epsilon))

    y2 = np.rint(y * (10 ** order)).astype(np.int)

    uy = np.unique(np.abs(y2)).astype(np.uint)
    GCD = np.gcd.reduce(uy)

    y2 = y2 / GCD

    return y2.astype(np.int)


def Inequality(Graph, inflation_order, card, Sol, b, SpMatrix):
    y = np.array(Sol)
    yreal = np.array(Sol)
    n = 1

    epsilon = Epsilon(y, n)

    y = yZeroFilter(Sol, n)

    while ValidityCheck(y, SpMatrix, yreal) == False:
        n = n * 10
        y = yZeroFilter(Sol, n)

    y = yRoundFilter(y, n)

    while ValidityCheck(y, SpMatrix, yreal) == False:
        n = n * 10
        y = yRoundFilter(y, n)

    Comp = Compatibility(Sol, b, epsilon)

    if Comp == False:

        obs_count, num_vars, exp_set, group_elem, det_assumptions, names = inflation.LearnInflationGraphParameters(
            Graph, inflation_order)
        G = inflation.GenerateEncodingMonomialToRow(card ** obs_count, inflation_order)

        st = ''.join(str(i) for i in np.arange(card))

        V = product(st, repeat=obs_count)

        ListingProbs1 = ['P(' + ''.join(v) + ')' for v in V]

        ListingProbs2 = np.array([sy.symbols(v) for v in ListingProbs1])

        preb = ListingProbs2
        bs = preb
        for i in range(1, inflation_order):
            bs = np.kron(preb, bs)

        ListingProbs3 = []
        for i in np.unique(G):

            indecies = np.where(G == i)[0]
            elem = bs[indecies[0]]

            for j in range(1, len(indecies)):
                elem = elem + bs[indecies[j]]
            ListingProbs3.append(elem)
        ListingProbs3 = np.array(ListingProbs3)

        y = y.astype(np.int)

        InequalityAsArray = y.T.dot(ListingProbs3)
        # InequalityAsString=str(InequalityAsArray)+'>=0'

        divisor = sy.content(InequalityAsArray[0])

        Inequality = InequalityAsArray[0] / divisor

        InequalityAsString = str(Inequality) + '>=0'

        return Inequality, InequalityAsString
    else:
        return print('Compatibility Error: The input distribution is compatible with given inflation order test.')


Graph = TriangleGraph
IA, IS = Inequality(Graph, inflation_order, card, Sol, b, SpMatrix)

# print(SOL)
print(IS)

"""
a = Ineq
a = r'{}'.format(a)
ax = axes() #left,bottom,width,height
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
text(0,0.5,r'$%s$' %a,size=10,color="black")
"""
