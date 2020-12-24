#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleanup solver output to yield a readable inequality.
"""

import numpy as np
from numba import njit
import sympy as sy
from scipy.sparse import csr_matrix
from collections import defaultdict
import json
from .quickgraph import LearnSomeInflationGraphParameters
from .inflationmatrix import Generate_b_and_counts



def WitnessDataTest(y,b,tol):
    IncompTest=(np.dot(y,b) < tol)
    if IncompTest:
        print('Distribution Compatibility Status: INCOMPATIBLE')
    else:
        print('Distribution Compatibility Status: COMPATIBLE')
    return IncompTest

@njit
def Epsilon(y,n):
    ymax=np.amax(np.abs(y))
    denom=1000*n
    return ymax/denom

def ValidityCheck(y,SpMatrix):
    #Smatrix=SpMatrix.toarray()    #DO NOT LEAVE SPARSITY!!
    checkY=csr_matrix(y.ravel()) * SpMatrix     
    return (checkY.min() >= 0)
    
@njit
def yZeroFilter(yRaw,n):
    y=yRaw.copy()
    y[np.abs(y) < Epsilon(y,n)] = 0;
    return y

def yRoundFilter(y,n):
    order=-np.floor(np.log10(Epsilon(y,n)))
    y2=np.rint(y*(10**order)).astype(np.int);
    uy=np.unique(np.abs(y2))
    GCD=np.gcd.reduce(uy)
    y2=y2/GCD
    return y2.astype(np.int);   
    

def Inequality(Graph,inflation_order,card,SpMatrix,b,Sol):
    yRaw=np.array(Sol['x']).ravel()
    if WitnessDataTest(yRaw,b,Sol['gap']):
        n=1
        y=yZeroFilter(yRaw,n)
        y=yRoundFilter(y,n)
        while ValidityCheck(y,SpMatrix) == False:
            n=n*10
            y=yZeroFilter(yRaw,n)
            y=yRoundFilter(y,n)
            
        #print(n)
        obs_count,num_vars,names=LearnSomeInflationGraphParameters(Graph,inflation_order)
        
        symbolnames=['P('+''.join([''.join(str(i)) for i in idx])+')' for idx in np.ndindex(tuple(np.full(obs_count,card,np.uint8)))]
        symbols=np.array(sy.symbols(symbolnames))
        symb,counts=Generate_b_and_counts(symbols, inflation_order)
        symbtostring= np.array([str(term).replace('*P','P') for term in symb])
        
        yRaw=np.multiply(yRaw,counts)
        
        y3=np.multiply(y,counts)
        uy=np.unique(np.abs(y3))
        GCD=np.gcd.reduce(uy)
        y3=y3/GCD
        y3=y3.astype(np.int)
        #ySparse=coo_matrix(y3)
        
        #print('Now to make things human readable...')
        
        indextally=defaultdict(list)
        #[indextally[str(y3[i])].append(i) for i in np.nonzero(y3)]
        [indextally[str(val)].append(i) for i,val in enumerate(y3) if val != 0]
        
        symboltally=defaultdict(list)
        for i, vals in indextally.items():
            symboltally[i] = symbtostring.take(vals).tolist() 
        
        final_ineq_WITHOUT_ZEROS=np.multiply(y3[np.nonzero(y3)],symb[np.nonzero(y3)])
        Inequality_as_string='0≤'+"+".join([str(term) for term in final_ineq_WITHOUT_ZEROS]).replace('*P','P')
        Inequality_as_string=Inequality_as_string.replace('+-','-')
        
        print("Writing to file: 'inequality_output.json'")
        
        returntouser={
         'Order of variables': names,
         'Raw rolver output': yRaw.tolist(),
         'Inequality as string': Inequality_as_string,
         'Coefficients grouped by index': indextally,
         'Coefficients grouped by symbol': symboltally,
        # 'b_vector_position': idx.tolist(),
         'Clean solver output': y3.tolist(),
         'Symolic association': symbtostring.tolist()
        };
        f = open('inequality_output.json', 'w');
        print(json.dumps(returntouser), file=f)
        f.close()
        return returntouser
    else:
        return print('Compatibility Error: The input distribution is compatible with given inflation order test.')

