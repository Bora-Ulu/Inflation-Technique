import sys
import mosek
import numpy as np
#from scipy.sparse import csr_matrix


# Since the actual value of Infinity is ignores, we define it solely
# for symbolic purposes:
inf = 0.0

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def InfeasibilityCertificate(A,b):
    #A is presumed to by a SciPy sparse matrix, b is a numpy array.
    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            (numvar,numcon)=A.shape


            #rangecon=np.arange(numcon).astype(np.int_).tolist()
            #rangevar=np.arange(numvar).astype(np.int_).tolist()

            streamprinter('Inferred variable count: ' + str(numvar)+'\n')

            # Bounds for variables (We take the smallest value to be no lower than -1. Seems reasonable, right?)
            bkx = numvar * [mosek.boundkey.lo]
            blx = numvar * [-1.0]
            #blx = [0.] * numvar
            bux = numvar * [+inf]

            # Bounds for constraints
            bkc = numcon * [mosek.boundkey.lo]
            blc = numcon * [0.0]
            buc = numcon * [1.0]

            # A matrix parameters
            csr = A.asformat('csr', copy=False)
            ptrb = csr.indptr[:-1]
            ptre = csr.indptr[1:]

            task.inputdata(numcon,  # number of constraints
                           numvar,  # number of variables
                           b.tolist(),  # linear objective coefficients
                           0.0,  # objective fixed value
                           list(ptrb),
                           list(ptre),
                           list(csr.indices),
                           list(csr.data),
                           bkc,
                           blc,
                           buc,
                           bkx,
                           blx,
                           bux)
            """
            # Loading objective
            task.putclist(rangevar, b.tolist())
            task.putvarboundlistconst(rangevar, mosek.boundkey.fr, blx, bux)
            #task.putvarboundlist(rangevar, bkx, blx, bux)



            # Loading constraints
            task.putconboundlistconst(rangecon, mosek.boundkey.ra, blc, buc)
            #task.putconboundlist(rangecon, bkc, blc, buc)


            task.putarowlist(rangecon, ptrb, ptre, csr.indices, csr.data)
            """
            task.putobjsense(mosek.objsense.minimize)
            task.putintparam(mosek.iparam.bi_clean_optimizer, mosek.optimizertype.primal_simplex)
            task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.off)
            task.putintparam(mosek.iparam.presolve_lindep_use, mosek.onoffkey.off)
            task.putintparam(mosek.iparam.presolve_max_num_reductions, 0)
            task.putintparam(mosek.iparam.presolve_max_num_reductions, 0)
            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
            streamprinter('LP constructed, initiated optimizer.\n')
            #task.writetask('mosekLPinput.tar')
            task.writedata('mosek_prob.jtask')
            # Optimize the task
            task.optimize()
            task.writejsonsol('mosek_sol.json')

            # Print a summary containing information
            #about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            #Use basic if available. (It won't be.)
            if task.solutiondef(mosek.soltype.bas)==1:
                soltype = mosek.soltype.bas
            elif task.solutiondef(mosek.soltype.itr)>0:
                soltype = mosek.soltype.itr
            else:
                raise ValueError("No valid soltype for mosek detected.")
                soltype = mosek.soltype.itr


            prosta = task.getprosta(soltype)
            solsta = task.getsolsta(soltype)

            # Output a solution
            xx = numvar * [0.0]
            y = numcon * [0.0]
            skc = numcon * [mosek.stakey.unk]
            skx = numvar * [mosek.stakey.unk]
            skn = 0 * [mosek.stakey.unk]
            xc = numcon * [0.0]
            slc = numcon * [0.0]
            suc = numcon * [0.0]
            slx = numvar * [0.0]
            sux = numvar * [0.0]
            snx = numvar * [0.0]
            task.getsolution(soltype, skc, skx, skn, xc, xx, y, slc, suc, slx, sux, snx)
            gap = np.linalg.norm(np.subtract(np.array(suc),np.array(slc)), np.inf)
            Sol = {'x': xx, 'y': y, 'xc': xc, 'gap': gap}

            #task.dispose()
        #env.dispose()
    streamprinter('Problem status: '+str(prosta))
    streamprinter('Solution status: ' + str(solsta))

    return Sol

