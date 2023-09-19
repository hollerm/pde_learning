


import pde_learning as pdl
import pde_learning_supp as pdl_supp
import numpy as np



#Set parameters
pardict = {
'niter':10000,
'nsamples':1,
'nn_geometry':[1,2,4,2,1],
'unknowns':['u','w'],    
'a_pde': 500,
'a_phi' : 1e-1,
'a_w' : 1e-3,
'a_u' : 5e-4,
'a_u0': 5,
'n_tmeas': 6,
'nstd':0.03
}
outfolder = 'demo_result'

res = pdl.pde_learning(**pardict)
#Store result
res.save(fname='quadratic_u_w_discmeas',outpars=['niter','n_tmeas','nstd'],folder=outfolder)
print('#############################')
print('Result stored in ' + outfolder)

#Export result
pdl_supp.export_paper(res,display=False,mode='u_w',outfolder=outfolder + '/figures')                                
        
print('Figures printed to ' + outfolder + '/figures')
