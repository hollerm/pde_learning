import pde_learning as pdl
import pde_learning_supp as pdl_supp
import numpy as np


# Global flags
niter = 10000
load_result = False # Set true to load the result from file instead of computing it (for figure updates)


exptypes = ['u_w','u_w_phi','multisamples','nl_test','poly_approx','trig_approx','highres']

#Get results for quadratic f, single datum, different discrete observations, different noise levels
if 'u_w' in exptypes: 
    
    #Set test range   
    n_tmeas = [50,6,3]
    nstd = [0.01,0.03,0.05,0.1,0.2]

    
    #Set variable parameters
    a_u0 = [
    [10,100,4,5,2],
    [25,5,4,2,5],
    [100,18,18,18,2]
    ]

    #Set fixed parameters
    pardict = {
        'niter':niter,
        'nsamples':1,
        'nn_geometry':[1,2,4,2,1],
        'unknowns':['u','w'],    
        'a_pde': 500,
        'a_phi' : 1e-1,
        'a_w' : 1e-3,
        'a_u' : 5e-4
    }
    
    #Error containers
    nl_err = np.zeros((len(n_tmeas),len(nstd)))
    par_err = np.zeros(nl_err.shape)
    st_err = np.zeros(nl_err.shape)


    for i in range(len(n_tmeas)):
        
        pardict['n_tmeas'] = n_tmeas[i]
        
        for k in range(len(nstd)):
            
            pardict['nstd'] = nstd[k]
            pardict['a_u0'] = a_u0[i][k]

            if not load_result:
                                
                res = pdl.pde_learning(**pardict)
                #res.show()
                
                #Store result
                res.save(fname='quadratic_u_w_discmeas',outpars=['niter','n_tmeas','nstd'],folder='paper_results')
                
            else:
            
                fname = pdl_supp.output_name(fname='quadratic_u_w_discmeas',outpars=['niter','n_tmeas','nstd'],folder='paper_results',par=mp.parameter(pardict))
                res = mp.pload(fname)

            #Export result
            pdl_supp.export_paper(res,display=False,mode='u_w')                                

            # Nonlinearity error
            nl_err[i,k] = res.f_err[-1]/np.prod(res.fy0.shape).astype('float')
            # Parameter error
            par_err[i,k] = res.phi_err[-1]/np.prod(res.phi0.shape).astype('float')
            # State error
            st_err[i,k] = res.u_err[-1]/np.prod(res.u0.shape).astype('float')
                

    
    #Save and print errors
    np.save('paper_results/quadratic_u_w_discmeas_nl_err',nl_err)
    np.save('paper_results/quadratic_u_w_discmeas_par_err',par_err)
    np.save('paper_results/quadratic_u_w_discmeas_st_err',st_err)

    
    print('u_w')
    print(pdl_supp.np_to_latex(nl_err))
    print(pdl_supp.np_to_latex(par_err))    
    print(pdl_supp.np_to_latex(st_err))    
    
#Get results for quadratic f, single datum, different discrete observations, different noise levels
if 'u_w_phi' in exptypes: 
    
    #Set test range   
    n_tmeas = [50,10,6]
    nstd = [0.01,0.03,0.05,0.08,0.1]

    
    #Set variable parameters
    a_u0 = [
    [35,10,10,5,4], 
    [25,10,4,4,2], 
    [200,5,18,10,2] 
    ]

    #Set fixed parameters
    pardict = {
        'niter':niter,
        'nsamples':1,
        'nn_geometry':[1,2,4,2,1],
        'unknowns':['u','w','phi'],    
        'a_pde': 10,
        'a_phi' : 1e-1,
        'a_w' : 1e-4,
        'a_u' : 1e-3
    }
    
    
    #Error containers
    nl_err = np.zeros((len(n_tmeas),len(nstd)))
    par_err = np.zeros(nl_err.shape)
    st_err = np.zeros(nl_err.shape)


    for i in range(len(n_tmeas)):
        
        pardict['n_tmeas'] = n_tmeas[i]
        
        for k in range(len(nstd)):
        
            pardict['nstd'] = nstd[k]
            pardict['a_u0'] = a_u0[i][k]

            if not load_result:
                    
                res = pdl.pde_learning(**pardict)

                #Store corresponding results
                res.save(fname='quadratic_u_w_phi_discmeas',outpars=['niter','n_tmeas','nstd'],folder='paper_results')
                
            else:
            
                fname = pdl_supp.output_name(fname='quadratic_u_w_phi_discmeas',outpars=['niter','n_tmeas','nstd'],folder='paper_results',par=mp.parameter(pardict))
                res = mp.pload(fname)

            #Export result for paper
            pdl_supp.export_paper(res,display=False,mode='u_w_phi')                                

                
            # Nonlinearity error
            nl_err[i,k] = res.f_err[-1]/np.prod(res.fy0.shape).astype('float')
            # Parameter error
            par_err[i,k] = res.phi_err[-1]/np.prod(res.phi0.shape).astype('float')
            # State error
            st_err[i,k] = res.u_err[-1]/np.prod(res.u0.shape).astype('float')



    #Save and print errors
    np.save('paper_results/quadratic_u_w_phi_discmeas_nl_err',nl_err)
    np.save('paper_results/quadratic_u_w_phi_discmeas_par_err',par_err)
    np.save('paper_results/quadratic_u_w_phi_discmeas_st_err',st_err)


    print('u_w_phi')
    print(pdl_supp.np_to_latex(nl_err))
    print(pdl_supp.np_to_latex(par_err))    
    print(pdl_supp.np_to_latex(st_err))   
                                
            
    
    
    
    
#Get results for multiple measurements
if 'multisamples' in exptypes: 

    
    #Set test range   
    nsamples = [1,3,5,20]

    #Set variable parameters
    a_u0 = [5,5,5,5]
    a_w = [1e-2,1e-4,1e-4,1e-5]

    #Set fixed parameters
    pardict = {
        'niter':niter,
        'n_tmeas':3,
        'explicit_samples':False,
        'nn_geometry':[1,2,4,2,1],
        'nstd': 0.08,
        'unknowns':['u','w','phi'],    
        'a_pde': 10,
        'a_phi' : 1e-1,
        'a_u' : 1e-3
    }
    
    
    #Error containers
    nl_err = np.zeros(len(nsamples))
    par_err = np.zeros(nl_err.shape)
    st_err = np.zeros(nl_err.shape)


    for i in range(len(nsamples)):
        
        pardict['nsamples'] = nsamples[i]
        pardict['a_u0'] = a_u0[i]
        pardict['a_w'] = a_w[i]

        if not load_result:
            res = pdl.pde_learning(**pardict)

            res.save(fname='quadratic_u_w_phi_multsamples',outpars=['niter','nsamples'],folder='paper_results')

        else: 
        
            fname = pdl_supp.output_name(fname='quadratic_u_w_phi_multsamples',outpars=['niter','nsamples'],folder='paper_results',par=mp.parameter(pardict))
            res = mp.pload(fname)
        
        pdl_supp.export_paper(res,display=False,mode='u_w_phi_multsamples')
        
        # Nonlinearity error
        nl_err[i] = res.f_err[-1]/np.prod(res.fy0.shape).astype('float')
        # Parameter error
        par_err[i] = res.phi_err[-1]/np.prod(res.phi0.shape).astype('float')
        # State error
        st_err[i] = res.u_err[-1]/np.prod(res.u0.shape).astype('float')

            


    #Save and print errors
    np.save('paper_results/quadratic_u_w_phi_multsamples_nl_err',nl_err)
    np.save('paper_results/quadratic_u_w_phi_multsamples_par_err',par_err)
    np.save('paper_results/quadratic_u_w_phi_multsamples_st_err',st_err)

    
    #Save and print errors
    #print(pdl_supp.np_to_latex(nl_err))
    #print(pdl_supp.np_to_latex(par_err))
    #print(pdl_supp.np_to_latex(st_err))        
    



#Get results for different nonlinearities
if 'nl_test' in exptypes: 


    #type of nonlinearity
    nonlinearity = ['square','poly','linear','cos']    
    
    #Set test range   

    #Set variable parameters
    a_u0 = [1,5,1,5]
    a_w = [1e-4,1e-6,1e-6,1e-6]

    #Set fixed parameters
    pardict = {
        'niter':niter,
        'explicit_samples':False,
        'nsamples':1,
        'n_tmeas':10,
        'nn_geometry':[1,2,4,2,1],
        'nstd': 0.03,
        'unknowns':['u','w'],
        'a_pde': 10,
    #    'a_u0' : 10.0,
        'a_phi' : 1e-1,
#        'a_w' : 1e-6,
        'a_u' : 1e-3,
        'u_datainit':True,
        'u_update_delay' : 0.7,
    }
    
    
    #Error containers
    nl_err = np.zeros(len(nonlinearity))
    par_err = np.zeros(nl_err.shape)
    st_err = np.zeros(nl_err.shape)


    for i in range(len(nonlinearity)):
        
        pardict['nonlinearity'] = nonlinearity[i]
        pardict['a_u0'] = a_u0[i]
        pardict['a_w'] = a_w[i]

        if not load_result:
            res = pdl.pde_learning(**pardict)
     
            res.save(fname='nonlinearity_test',outpars=['niter','nonlinearity'],folder='paper_results')

        else: 
        
            fname = pdl_supp.output_name(fname='nonlinearity_test',outpars=['niter','nonlinearity'],folder='paper_results',par=mp.parameter(pardict))
            res = mp.pload(fname)

        pdl_supp.export_paper(res,display=False,mode='nonlinearity_test')
        
        # Nonlinearity error
        nl_err[i] = res.f_err[-1]/np.prod(res.fy0.shape).astype('float')
        # Parameter error
        par_err[i] = res.phi_err[-1]/np.prod(res.phi0.shape).astype('float')
        # State error
        st_err[i] = res.u_err[-1]/np.prod(res.u0.shape).astype('float')



    #Save and print errors
    np.save('paper_results/nonlinearity_test_nl_err',nl_err)
    np.save('paper_results/nonlinearity_test_par_err',par_err)
    np.save('paper_results/nonlinearity_test_st_err',st_err)

    
    #Save and print errors

    #print(pdl_supp.np_to_latex(nl_err))
    #print(pdl_supp.np_to_latex(par_err))
    #print(pdl_supp.np_to_latex(st_err))            


#Get results for different nonlinearities with poly approx
if 'poly_approx' in exptypes: 


    #type of nonlinearity
    nonlinearity = ['square','poly','linear','cos']    
    
    #Set test range   

    #Set variable parameters
    a_u0 = [5,100,1,10]
    a_w = [0.001,1e-06,0.001,1e-05]

    #Set fixed parameters
    pardict = {
        'niter':niter,
        'explicit_samples':False,
        'nsamples':1,
        'n_tmeas':10,
        'nn_geometry':[1,2,4,2,1],
        'nstd': 0.03,
        'unknowns':['u','w'],
        'a_pde': 10,
    #    'a_u0' : 10.0,
        'a_phi' : 1e-1,
#        'a_w' : 1e-6,
        'a_u' : 1e-3,
        'u_datainit':True,
        'u_update_delay' : 0.7,
        'approx_type':'poly',
        'poly_approx_degree':29
    }
    
    
    #Error containers
    nl_err = np.zeros(len(nonlinearity))
    par_err = np.zeros(nl_err.shape)
    st_err = np.zeros(nl_err.shape)


    for i in range(len(nonlinearity)):
        
        pardict['nonlinearity'] = nonlinearity[i]
        pardict['a_u0'] = a_u0[i]
        pardict['a_w'] = a_w[i]

        if not load_result:
            res = pdl.pde_learning(**pardict)
     
            res.save(fname='poly_nonlinearity_test',outpars=['niter','nonlinearity'],folder='paper_results')

        else: 
        
            fname = pdl_supp.output_name(fname='poly_nonlinearity_test',outpars=['niter','nonlinearity'],folder='paper_results',par=mp.parameter(pardict))
            res = mp.pload(fname)

        pdl_supp.export_paper(res,display=False,mode='nonlinearity_test')
        
        # Nonlinearity error
        nl_err[i] = res.f_err[-1]/np.prod(res.fy0.shape).astype('float')
        # Parameter error
        par_err[i] = res.phi_err[-1]/np.prod(res.phi0.shape).astype('float')
        # State error
        st_err[i] = res.u_err[-1]/np.prod(res.u0.shape).astype('float')

            

    #Save and print errors
    np.save('paper_results/poly_nonlinearity_test_nl_err',nl_err)
    np.save('paper_results/poly_nonlinearity_test_par_err',par_err)
    np.save('paper_results/poly_nonlinearity_test_st_err',st_err)
    
    #Save and print errors

    #print(pdl_supp.np_to_latex(nl_err))
    #print(pdl_supp.np_to_latex(par_err))
    #print(pdl_supp.np_to_latex(st_err))            




#Get results for different nonlinearities with trig approx
if 'trig_approx' in exptypes: 


    #type of nonlinearity
    nonlinearity = ['square','poly','linear','cos']    
    
    #Set test range   

    #Set variable parameters
    a_u0 = [5,10,1,10]
    a_w = [0.001,1e-06,0.001,1e-05]

    #Set fixed parameters
    pardict = {
        'niter':niter,
        'explicit_samples':False,
        'nsamples':1,
        'n_tmeas':10,
        'nn_geometry':[1,2,4,2,1],
        'nstd': 0.03,
        'unknowns':['u','w'],
        'a_pde': 10,
    #    'a_u0' : 10.0,
        'a_phi' : 1e-1,
#        'a_w' : 1e-6,
        'a_u' : 1e-3,
        'u_datainit':True,
        'u_update_delay' : 0.7,
        'approx_type':'trig',
        'poly_approx_degree':15
    }
    
    
    #Error containers
    nl_err = np.zeros(len(nonlinearity))
    par_err = np.zeros(nl_err.shape)
    st_err = np.zeros(nl_err.shape)


    for i in range(len(nonlinearity)):
        
        pardict['nonlinearity'] = nonlinearity[i]
        pardict['a_u0'] = a_u0[i]
        pardict['a_w'] = a_w[i]

        if not load_result:
            res = pdl.pde_learning(**pardict)
     
            res.save(fname='trig_nonlinearity_test',outpars=['niter','nonlinearity'],folder='paper_results')

        else: 
        
            fname = pdl_supp.output_name(fname='trig_nonlinearity_test',outpars=['niter','nonlinearity'],folder='paper_results',par=mp.parameter(pardict))
            res = mp.pload(fname)

        pdl_supp.export_paper(res,display=False,mode='nonlinearity_test')
        
        # Nonlinearity error
        nl_err[i] = res.f_err[-1]/np.prod(res.fy0.shape).astype('float')
        # Parameter error
        par_err[i] = res.phi_err[-1]/np.prod(res.phi0.shape).astype('float')
        # State error
        st_err[i] = res.u_err[-1]/np.prod(res.u0.shape).astype('float')



    #Save and print errors
    np.save('paper_results/trig_nonlinearity_test_nl_err',nl_err)
    np.save('paper_results/trig_nonlinearity_test_par_err',par_err)
    np.save('paper_results/trig_nonlinearity_test_st_err',st_err)
    
    #Save and print errors

    #print(pdl_supp.np_to_latex(nl_err))
    #print(pdl_supp.np_to_latex(par_err))
    #print(pdl_supp.np_to_latex(st_err))           





#Get results for quadratic f, single datum, different discrete observations, different resolution levels
if 'highres' in exptypes: 

    #Set test range   
    nx = [501,5001]
    
    #Set variable parameters
    nt = [500,5000]
    
    a_u0 = [400,400]    
    a_u = [5e-2,5e-2]
    

    #Set fixed parameters
    pardict = {
        'niter':niter,
        'nsamples':1,
        'nn_geometry':[1,2,4,2,1],
        'unknowns':['u','w'],    
        'a_pde': 500,
        'a_phi' : 1e-1,
        'a_w' : 1e-3,
        'n_tmeas':6,
        'nstd':0.03,
        'u_datainit':True
    }


    
    #Error containers
    nl_err = np.zeros((len(nx)))
    par_err = np.zeros(nl_err.shape)
    st_err = np.zeros(nl_err.shape)


    for i in range(len(nx)):
        
        pardict['nx'] = nx[i]
        pardict['nt'] = nt[i]
        pardict['a_u0'] = a_u0[i]
        pardict['a_u'] = a_u[i]
        
        if not load_result:
            res = pdl.pde_learning(**pardict)
            #res.show()
            res.save(fname='quadratic_u_w_discmeas_nxtest',outpars=['niter','nx','nt','nstd'],folder='paper_results')

        else:
        
            fname = pdl_supp.output_name(fname='quadratic_u_w_discmeas_nxtest',outpars=['niter','nx','nt','nstd'],folder='paper_results',par=mp.parameter(pardict))
            res = mp.pload(fname)
                    

        pdl_supp.export_paper(res,display=False,mode='u_w_highres')
    
        # Nonlinearity error
        nl_err[i] = res.f_err[-1]/np.prod(res.fy0.shape).astype('float')
        # Parameter error
        par_err[i] = res.phi_err[-1]/np.prod(res.phi0.shape).astype('float')
        # State error
        st_err[i] = res.u_err[-1]/np.prod(res.u0.shape).astype('float')

            



    #Save and print errors
    np.save('paper_results/quadratic_u_w_discmeas_nxtest_nl_err',nl_err)
    np.save('paper_results/quadratic_u_w_discmeas_nxtest_par_err',par_err)
    np.save('paper_results/quadratic_u_w_discmeas_nxtest_st_err',st_err)
    
    #Save and print errors
    print('highres')
    print(pdl_supp.np_to_latex(nl_err))
    print(pdl_supp.np_to_latex(par_err))
    print(pdl_supp.np_to_latex(st_err))
    

    
