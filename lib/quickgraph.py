#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning a little bit about the inflation graph from the original graph
"""

import numpy as np
#from igraph import * #Not needed, since we are using methods built into the object...!


def ToTopologicalOrdering(g):
    return g.permute_vertices(np.argsort(g.topological_sorting('out')).tolist())

def LearnParametersFromGraph(origgraph):
    g=ToTopologicalOrdering(origgraph)
    verts=g.vs
    verts["parents"]=g.get_adjlist('in');
    #verts["children"]=g.get_adjlist('out');
    verts["ancestors"]=[g.subcomponent(i,'in') for i in g.vs]
    verts["descendants"]=[g.subcomponent(i,'out') for i in g.vs]
    verts["indegree"]=g.indegree()
    #verts["outdegree"]=g.outdegree() #Not needed
    verts["grandparents"]=g.neighborhood(None, order=2, mode='in', mindist=2)
    #verts["parents_inclusive"]=g.neighborhood(None, order=1, mode='in', mindist=0) #Not needed
    has_grandparents=[idx for idx,v in enumerate(g.vs["grandparents"]) if len(v)>=1]
    verts["isroot"]=[0==i for i in g.vs["indegree"]]
    root_vertices=verts.select(isroot = True).indices
    nonroot_vertices=verts.select(isroot = False).indices
    verts["roots_of"]=[np.intersect1d(anc,root_vertices).tolist() for anc in g.vs["ancestors"]]
    def FindScreeningOffSet(root,observed):
        screeningset=np.intersect1d(root["descendants"],observed["parents"]).tolist()
        screeningset.append(observed.index)
        return screeningset
    determinism_checks=[(root,FindScreeningOffSet(verts[root],v)) for v in g.vs[has_grandparents] for root in np.setdiff1d(v["roots_of"],v["parents"])]
    return verts["name"],verts["parents"],verts["roots_of"],determinism_checks


def LearnSomeInflationGraphParameters(g,inflation_order):
    names,parents_of,roots_of,determinism_checks = LearnParametersFromGraph(g)
    #print(names)
    graph_structure=list(filter(None,parents_of))
    obs_count=len(graph_structure)
    latent_count=len(parents_of)-obs_count
    root_structure=roots_of[latent_count:]
    inflation_depths=np.array(list(map(len,root_structure)))
    inflationcopies=inflation_order**inflation_depths
    num_vars=inflationcopies.sum()
    return obs_count,num_vars,names[latent_count:]


