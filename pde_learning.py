

import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

import numpy as np
  
import time

from pde_learning_supp import *

from scipy.interpolate import interp2d
        
    
def Dt(u,dtu):

    dtu[:,:,1:-1] = u[:,:,:-2] - u[:,:,2:]
    dtu[:,:,0] = - u[:,:,1]
    dtu[:,:,-1] = u[:,:,-3] - 4*u[:,:,-2] + 3*u[:,:,-1]

    return dtu



def lpl(u,lpl_u):

    lpl_u[:,1:-1,:] = u[:,:-2,:] - 2*u[:,1:-1,:] + u[:,2:,:]
    lpl_u[:,0,:] =  - 2*u[:,0,:] + u[:,1,:]
    lpl_u[:,-1,:] = u[:,-2,:] -2*u[:,-1,:]

    return lpl_u


def Dt_dev(u,dtu):

    dtu[:,:,1:-1,:] = u[:,:,:-2] - u[:,:,2:,:]
    dtu[:,:,0,:] = - u[:,:,1,:]
    dtu[:,:,-1,:] = u[:,:,-3,:] - 4*u[:,:,-2,:] + 3*u[:,:,-1,:]

    return dtu



def lpl_dev(u,lpl_u):

    lpl_u[:,1:-1,:,:] = u[:,:-2,:,:] - 2*u[:,1:-1,:,:] + u[:,2:,:,:]
    lpl_u[:,0,:,:] =  - 2*u[:,0,:,:] + u[:,1,:,:]
    lpl_u[:,-1,:,:] = u[:,-2,:,:] -2*u[:,-1,:,:]

    return lpl_u





def pde_learning(**par_in):


    #Initialize parameters and data input
    par = parameter({})

    #Set seed for random init
    par.seed = 222

    par.nx = 51
    par.nt = 50
    
    #Domain size
    par.omin = 0
    par.omax = 1
    
    #Lenght of time interval
    par.tmax = 0.1
    
    par.niter = 1000
    
    
    par.n_tmeas = 10 #Number of time measurements
    par.nstd = 0.05 #Standard deviation of noise
    
    par.nsamples = 5 #Number of data samples
    
    par.explicit_samples = False #If true, we use a fixed number of explicitly defined sampels
    
    
    #Regularization parameters
    par.a_pde = 10
    par.a_u0 = 50.0
    par.a_phi = 0.1
    par.a_w = 1e-4
    par.a_u = 0.001
    
    
    
    par.track_every = 100 #Defines in which iterations to track performance measures
    
    
    par.unknowns = ['u','phi','w'] #Options: 'u','phi','w'
    
    par.nn_geometry = [1,2,8,2,1]
    
    #Set true to initialize the state with the noisy data, false means random init
    par.u_datainit = False

    #Factor to delay the update of the state, 0 means no delay
    par.u_update_delay = 0


    #Explicit input of nonlinearity via name. Set False to use default f(x) = x^2 - 1
    par.nonlinearity = False

    par.approx_type = 'nn' #Type of approximation, choices are 'nn', 'poly', 'trig'
    
    par.poly_approx_degree = 3 #Degree of polynomial-1 in case of poly or trig approximation. needs to be at least 2

    
    # Set code version
    #Update from verson NONE to version 1.0: added square in error measures and 
    #  cshift to remove shit ambiguity
    #Update to version 1.1: Defined seed as parameter, used it also for numpy
    #Update to version 1.2: Allowed explicit input of nonlinearity
    #Update to version 1.3: Allowed for different approximation types and update dealy with init
    par.version = 1.3


    #Set parameters according to par_in
    par_parse(par_in,[par])
    
    torch.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    #########################################
    # Preparations
    #########################################
        
    #Discrete domains
    x = np.linspace(par.omin,par.omax,par.nx)[:,np.newaxis]
    t = np.linspace(0,par.tmax,par.nt)[np.newaxis,:]
    
    
    
    
    #Non-linearity
    if not par.nonlinearity or par.nonlinearity == 'square':
        def f(y_in):
            return y_in*y_in-1.0
    elif par.nonlinearity == 'cos':
        def f(y_in):
            return np.cos(3*np.pi*y_in)
    elif par.nonlinearity == 'linear':
        def f(y_in):
            return 2.0 - 1.0*y_in
    elif par.nonlinearity == 'poly':
        def f(y_in):
            #return ((y_in-0.1)**2)*((y_in-0.3)**2) - 1.0
            return (y_in - 0.1)*(y_in - 0.5)*( 141.6*y_in - 30.0 ) 
    else:
        raise ValueError('Unknown nonlinearity')


    #Define data
    #First check for explicit samples:
    if par.explicit_samples:
        par.nsamples = 3
    
    u0 = np.zeros((par.nsamples,par.nx,par.nt))
    phi0 = np.zeros((par.nsamples,par.nx,1))
    c0 = np.zeros(phi0.shape)
    
    #Buffer for dt and laplace
    dt_buf = np.zeros(u0.shape)
    lpl_buf = np.zeros(u0.shape)
    
    if par.explicit_samples:
        tmax = 0.1; C = 5
        
        u0[0,...] = np.sin(np.pi*x)*t*5
        u0[1,...] = (( -(x-C*tmax)**4/(C*tmax)**3	+ C*tmax)/tmax)*t
        u0[2,...] = ((-x**2+x)*40*C	)*np.square(t)
        
        phi0[0,...] = np.sin(2.0*np.pi*x)
        phi0[1,...] = np.sin(np.pi*x)
        phi0[2,...] = 0
    
    else:
        for sample in range(par.nsamples):
            
            freq = float(sample+1)/float(par.nsamples)        
            #Setup data
            u0[sample,...] = np.sin(np.pi*x*freq)*t*5
            
            phi0[sample,...] = np.sin(2.0*np.pi*x*freq)  #np.ones((1,1)) 

            c0[sample,...] = 0#-np.cos(2*np.pi*x*freq) + 1
        

    fu0 = f(u0)
    
       
    #PDE: 0 = Dt(u) - lpl(u) + c*u + f(u) - phi - source
    source = Dt(u0,dt_buf) - lpl(u0,lpl_buf) + c0*u0 + fu0 - phi0
        
    #Measurements        
    mtimes = np.round(np.linspace(0,par.nt-1,par.n_tmeas)).astype('int')

    ###############################
    #Original way to generate data with noise

    mu0 = u0[...,mtimes] + np.random.normal(0,par.nstd,u0[...,mtimes].shape)
    
    #Generate initialization
    
    if par.u_datainit:
        u_init = np.zeros(u0.shape)
        
        x_inter = np.arange(0,par.nx,1)
        t_inter = np.arange(0,par.nt,1)
        
        for nsample in range(u0.shape[0]):

            tmpsave = mu0[nsample,...]
            f_inter = interp2d(mtimes,x_inter,tmpsave)

            u_init[nsample,...] = f_inter(t_inter,x_inter)
        
    
    #Plot of nonlinearity
    y = np.linspace(np.amin(u0),np.amax(u0),par.nx)
    fy0 = f(y)
        
    
    
    #Select approximation function
    if par.approx_type == 'nn':
    
        # Set network architecture    
        architecture = []
        for elem in range(len(par.nn_geometry)-1):
            architecture += [nn.Linear(par.nn_geometry[elem],par.nn_geometry[elem+1])]
            if elem < len(par.nn_geometry)-2:
                architecture += [nn.Tanh()]

        #Build neural network
        class NeuralNetwork_tanh(nn.Module):
            def __init__(self):
                super(NeuralNetwork_tanh, self).__init__()
                self.flatten = nn.Flatten()
                self.linear_relu_stack = nn.Sequential(
                    *architecture
                )

            def forward(self, x):
                return self.linear_relu_stack(x)

        model = NeuralNetwork_tanh().to(device)

        modelpars = list(model.parameters())
    
        print('Number of model parameters: ')
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    elif par.approx_type == 'poly':
    
        if par.poly_approx_degree <2:
            raise ValueError('Degree of approximatoin polynomial need to be at least 2')
            
        polycofs = torch.rand(par.nsamples,1,1,1,par.poly_approx_degree,device=device,requires_grad = True)
        
        def model(x):

            out = polycofs[...,0] + polycofs[...,1]*x
            for i in range(2,par.poly_approx_degree):
                out += polycofs[...,i]*torch.pow(x,i)
                
            return out    
            
        modelpars = [polycofs]

    elif par.approx_type == 'trig':
    
        if par.poly_approx_degree <2:
            raise ValueError('Degree of approximatoin polynomial need to be at least 2')
            
        polycofs = torch.rand(par.nsamples,1,1,1,par.poly_approx_degree,2,device=device,requires_grad = True)
        
        def model(x):

            out = polycofs[...,0,0] + polycofs[...,0,0]*torch.cos(np.pi*x) + polycofs[...,0,1]*torch.sin(np.pi*x)
            for i in range(2,par.poly_approx_degree):
                out += polycofs[...,i,0]*torch.cos(np.pi*i*x) + polycofs[...,i,1]*torch.sin(np.pi*i*x)
                
            return out    
            
        modelpars = [polycofs]


    else:
        raise ValueError('Uknown type of function approximation')
    
    
    
    ##Define variables
    
    #List of variables for optimizer
    opt_vars = []
    
    if 'u' in par.unknowns:
        u = torch.rand(par.nsamples,par.nx,par.nt,1,device=device,requires_grad = True)
        opt_vars += [u]
        if par.u_datainit:
            u.data[...,0] = torch.from_numpy(u_init).float().detach().to(device)
            u.requires_grad = True
            print('Initializing u with interpolated, noisy data')
    else:
        u = torch.from_numpy(np.expand_dims(u0,(3))).float().to(device)
        
    
    if 'phi' in par.unknowns:
        phi = torch.rand(par.nsamples,par.nx,1,1,device=device,requires_grad = True)
        opt_vars += [phi]
    else:
        phi = torch.from_numpy(np.expand_dims(phi0,(3))).float().to(device)
        
    if 'w' in par.unknowns:
        opt_vars += modelpars
    
    
    

    if par.u_update_delay>0:
        u.requires_grad = False

    #Stor init values
    u_init = u.cpu().detach().numpy()[...,0]
    phi_init = phi.cpu().detach().numpy()[...,0]
    

    fy_init = model(torch.from_numpy(np.expand_dims(y,(0,2,3))).float().to(device)).cpu().detach().numpy()[0,:,0,0]
    

    
    #Data for loss 
    mu0_dev = torch.from_numpy(np.expand_dims(mu0,(3))).float().to(device)
    c0_dev = torch.from_numpy(np.expand_dims(c0,(3))).float().to(device)
    source_dev = torch.from_numpy(np.expand_dims(source,(3))).float().to(device)
    

    def criterion(u,phi,parameters): #phi
    
        #Network output
        fu = model(u)
    
        #Error measure
        mse = nn.MSELoss()


        #Define operators with buffers
        Dtbuf = torch.empty(u.shape,device=device,requires_grad = False)
        lplbuf = torch.empty(u.shape,device=device,requires_grad = False)

        #print(lplbuf.requires_grad)
        loss = par.a_pde*mse( Dt_dev(u,Dtbuf) - lpl_dev(u,lplbuf) + c0_dev*u + fu - phi , source_dev)
        #print(lplbuf.requires_grad)
        
        if 'u' in par.unknowns:
            loss += par.a_u0*mse( u[...,mtimes,:],mu0_dev)
            
            
            loss += par.a_u*( u[:,:,0,0].square().sum() + lplbuf[:,:,0,0].detach().square().sum() ) #||u(0)||_V
            #loss += par.a_u*( u[:,:,0,0].square().sum()  ) #||u(0)||_V
            
            #Save lpl(dt u) to lplbuf
            lplbuf = lpl_dev(Dtbuf,lplbuf)
            
            #Due to the friedrichs inequality (see Trölzsch, Lemma 2.2.3), the following norm is equivalent 
            # to th h2 norm
            loss += par.a_u*( Dtbuf.square().sum() + lplbuf.square().sum() )
        
        if 'phi' in par.unknowns:
            loss += par.a_phi*mse(phi,torch.zeros(phi.shape,device=device,requires_grad=False))

        if 'w' in par.unknowns:
            for model_par in parameters:
                loss += float(par.nsamples)*par.a_w*mse(model_par,torch.zeros(model_par.shape,device=device,requires_grad=False))
        
        
        return loss
 
    optimizer = optim.Adam(opt_vars, lr=0.01)
    
    ntrack = int(np.floor(par.niter/par.track_every))+1

    ob_val = np.zeros(ntrack)
    f_err = np.zeros(ntrack)
    u_err = np.zeros(ntrack)
    phi_err = np.zeros(ntrack)
    pde_err = np.zeros(ntrack)
    f_m_phi_err = np.zeros(ntrack)

    # Get timenig
    time_cur = time.time()
    


    for epoch in range(par.niter):  # loop over the dataset multiple times

        running_loss = 0.0
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = criterion(u,phi,modelpars)#model.parameters())
        loss.backward()
        optimizer.step()
        
        if par.u_update_delay>0 and epoch > par.niter*par.u_update_delay and not u.requires_grad:
            u.requires_grad = True

        if (epoch == 0) or (np.remainder(epoch+1,par.track_every)==0):
            # print statistics
            tr_dx = int((epoch+1)/par.track_every)
            
            ob_val[tr_dx] = loss.item()
            
            #Model output on y
            mody0 = model(torch.from_numpy(np.expand_dims(y,(0,2,3))).float().to(device)).cpu().detach().numpy()[0,:,0,0]
            # Copy of parameter to cpu
            phiout_tmp = phi.cpu().detach().numpy()[...,0]
            
            #This gives cshift as solution to min_c (fy0-M(y0) - c)^2 + (phi0 - phi - c)^2
            cshift  = ( (fy0 - mody0).sum() + (phi0 - phiout_tmp).sum() ) / (np.prod(fy0.shape) + np.prod(phi0.shape))
            
            
            f_err[tr_dx] = np.square(fy0 - (mody0 + cshift) ).sum()

            u_err[tr_dx] = np.square(u0 - u.cpu().detach().numpy()[...,0]).sum()
            
            phi_err[tr_dx] = np.square(phi0 - (phiout_tmp + cshift) ).sum()
            
            f_m_phi_err[tr_dx] = np.square( fu0 - phi0 - (  model(u.detach()).cpu().detach().numpy()[...,0] - phi.cpu().detach().numpy()[...,0]) ).sum()

            print('Epoch: ' + str(epoch+1) + ', Loss: ' + str(ob_val[tr_dx]))


            
            

    print('Finished Training in ' + str(np.round(np.abs(time_cur - time.time()),decimals=2)) + ' seconds')    
    
    
    u_est = u.cpu().detach().numpy()[...,0]
    fu_est = model(u).cpu().detach().numpy()[...,0]
    phi_est = phi.cpu().detach().numpy()[...,0]

    
    #########################################
    # Process output
    #########################################    
    
    #Initialize output class
    res = output(par)
    
    
    #res.model = model
    res.u = u
    
    res.x = x
    res.t = t
    res.fu0 = fu0
    res.u0 = u0
    res.mu0 = mu0
    res.phi0 = phi0
    res.c0 = c0
    res.source = source
    
    res.u_init = u_init
    res.phi_init = phi_init
    res.fy_init = fy_init
    
    
    res.u_est = u_est
    res.fu_est = fu_est
    res.phi_est = phi_est

    res.y = y
    
    res.fy0 = fy0
    res.fy = model(torch.from_numpy(np.expand_dims(res.y,(0,2,3))).float().to(device)).cpu().detach().numpy()[0,:,0,0]

    res.mtimes = mtimes
    
    res.ob_val = ob_val
    res.f_err = f_err
    res.u_err = u_err
    res.phi_err = phi_err
    res.f_m_phi_err = f_m_phi_err
    res.par = par
    
    
    return res
    
