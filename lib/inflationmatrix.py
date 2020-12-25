#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:16:04 2020

@author: boraulu
"""
#import sys
#import pathlib 
#parent_directory=pathlib.Path(__file__).resolve().parent
#import importlib
#importlib.import_module('strategies', parent_directory)

from functools import lru_cache
from itertools import permutations

import numpy as np
from scipy.sparse import coo_matrix

from .graphs import LearnInflationGraphParameters
from .strategies import ValidColumnOrbits
from .utilities import PositionIndex, MoveToBack, GenShapedColumnIntegers


@lru_cache(maxsize=16)
def GenerateEncodingMonomialToRow(original_cardinality_product,inflation_order): #I should make this recursive, as called by both A and b construction.
    monomial_count=int(original_cardinality_product**inflation_order)
    permutation_count=int(np.math.factorial(inflation_order))
    MonomialIntegers=np.arange(0,monomial_count,1,np.uint)
    new_shape=np.full(inflation_order,original_cardinality_product)
    MonomialIntegersPermutations=np.empty([permutation_count,monomial_count],np.uint)
    IndexPermutations=list(permutations(np.arange(inflation_order)))
    MonomialIntegersPermutations[0]=MonomialIntegers
    MonomialIntegers=MonomialIntegers.reshape(new_shape)
    for i in np.arange(1,permutation_count):
        MonomialIntegersPermutations[i]=np.transpose(MonomialIntegers,IndexPermutations[i]).flat
    return PositionIndex(np.amin(
        MonomialIntegersPermutations,axis=0))

def GenerateEncodingColumnToMonomial(card,num_var,expr_set):
    initialshape=np.full(num_var,card,np.uint)
    ColumnIntegers=GenShapedColumnIntegers(tuple(initialshape))
    ColumnIntegers=ColumnIntegers.transpose(MoveToBack(num_var,np.array(expr_set))).reshape((-1,card**len(expr_set)))
    EncodingColumnToMonomial=np.empty(card**num_var,np.uint32)
    EncodingColumnToMonomial[ColumnIntegers]=np.arange(card**len(expr_set))
    return EncodingColumnToMonomial


def EncodeA(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card):
    #original_product_cardinality=(card**np.rint(len(expr_set)/inflation_order)).astype(np.uint)
    original_product_cardinality=card**obs_count
    EncodingMonomialToRow=GenerateEncodingMonomialToRow(original_product_cardinality,inflation_order)
    EncodingColumnToMonomial=GenerateEncodingColumnToMonomial(card,num_vars,np.array(expr_set))
    result=EncodingMonomialToRow[EncodingColumnToMonomial][valid_column_orbits]
    #Once the encoding is done, the order of the columns can be tweaked at will!
    #result=np.sort(result,axis=0)
    result.sort(axis=0) #in-place sort
    #result=result[np.lexsort(result),:]
    return result
    #return EncodingMonomialToRow[EncodingColumnToMonomial][valid_column_orbits]


def SciPyArrayFromOnesPositions(OnesPositions):
    columncount=OnesPositions.shape[-1]
    columnspec=np.broadcast_to(np.arange(columncount), (len(OnesPositions), columncount)).ravel()
    return coo_matrix((np.ones(OnesPositions.size,np.uint), (OnesPositions.ravel(), columnspec)),(int(np.amax(OnesPositions)+1), columncount),dtype=np.uint)

def SciPyArrayFromOnesPositionsWithSort(OnesPositions):
    columncount=OnesPositions.shape[-1]
    columnspec=np.broadcast_to(np.lexsort(OnesPositions), (len(OnesPositions), columncount)).ravel()
    return coo_matrix((np.ones(OnesPositions.size,np.uint), (OnesPositions.ravel(), columnspec)),(int(np.amax(OnesPositions)+1), columncount),dtype=np.uint)

def SparseInflationMatrix(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card):
    return SciPyArrayFromOnesPositionsWithSort(EncodeA(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card))

def InflationMatrixFromGraph(g,inflation_order,card):
    obs_count,num_vars,expr_set,group_elem,det_assumptions,names = LearnInflationGraphParameters(g,inflation_order)
    print(names) #REMOVE THIS PRINTOUT after accepting fixed order of variables.
    valid_column_orbits=ValidColumnOrbits(card, num_vars, group_elem,det_assumptions)
    return SciPyArrayFromOnesPositions(EncodeA(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card))

def Generate_b_and_counts(Data, inflation_order):
    EncodingMonomialToRow=GenerateEncodingMonomialToRow(len(Data),inflation_order)
    s,idx,counts=np.unique(EncodingMonomialToRow,return_index=True,return_counts=True)
    preb=np.array(Data)
    b=preb
    for i in range(1,inflation_order):
        b=np.kron(preb,b)
    return b[idx],counts

def FindB(Data, inflation_order):
    return np.multiply(*Generate_b_and_counts(Data, inflation_order))

