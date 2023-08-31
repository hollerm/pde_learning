
import matplotlib.pyplot as plt
import sys #For error handling
import copyreg
import pickle
import types
import os
import scipy.stats
from itertools import product

import numpy as np




    
# Object to store parameters
class parameter(object):
    
    def __init__(self,adict):
        self.__dict__.update(adict)
    
    #Define print function
    def __str__(self):
        #self.__dict__: Class members as dict
        #.__str__() is theprint function of a class
        return self.__dict__.__str__()


#Function to parse the arguments from par_in        
#Take a par_in dict and a list of parameter classes as input
#Sets the class members all elements of parlist according to par_in
#Raises an error when trying to set a non-existing parameter
def par_parse(par_in,parlist):

    for key,val in par_in.items():
        foundkey = False
        for par in parlist:
            if key in par.__dict__:
                par.__dict__[key] = val
                foundkey = True
        if not foundkey:
            raise NameError('Unknown parameter: ' + key)


#Data storage
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
        
    return func.__get__(obj, cls)


#Currently not used, only relevant as noted in psave
def pickle_get_current_class(obj):
    name = obj.__class__.__name__
    module_name = getattr(obj, '__module__', None)
    obj2 = sys.modules[module_name]
    for subpath in name.split('.'): obj2 = getattr(obj2, subpath)
    return obj2

def psave(name,data):
    
    #This might potentially fix erros with pickle and autoreload...try it next time the error ocurs
    data.__class__ = pickle_get_current_class(data)
    
    copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    
    if not name[-4:] == '.pkl':
        name = name + '.pkl'
        
    output = open(name,'wb')
    # Pickle the list using the highest protocol available.
    pickle.dump(data, output, -1)
    output.close()
    
def pload(name):

    copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    
    if not name[-4:] == '.pkl':
        name = name + '.pkl'
        
    
    try:
        pkl_file = open(name, 'rb')
        data = pickle.load(pkl_file)
    except:
        pkl_file.close()


        #print('Standard loading failed, resorting to python2 compatibility...')
        pkl_file = open(name, 'rb')
        data = pickle.load(pkl_file,encoding='latin1')

    pkl_file.close()
    return data
#Convert 2D numpy array entries to latex table
def np_to_latex(x):

    latex = ''

    if x.ndim == 2:    
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                latex += ' & ' + format(x[i,j],'.2e') 
            latex +='\n'
    elif x.ndim == 1:
       for i in range(x.shape[0]):
            latex += ' & ' + format(x[i],'.2e') 

    else:
        raise ValueError('Unsupported number of dimensions for latex plotting')
        
    return latex

#Convert 3D numpy array entries to latex table showing median and mean absolute deviation in w.r.t last dimension
def np_avg_to_latex(x):

    latex = ''
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
        
            med = np.median(x[i,j,:]) #median
            mad = scipy.stats.median_abs_deviation(x[i,j,:]) #median absolute deviation
            

            med_nr,med_exp = format(med,'.2e').split('e')
            #print(med_nr)
            #print(med_exp)
            #print(format(mad,'.2e'))
            
            latex += ' & (' + str(med_nr) + '\u00B1' + ('{:0=4.2f})e'+med_exp).format(mad / float('1e'+med_exp)) 
        latex +='\n'
    
    return latex



#Helper function to generate output name
def output_name(fname='',folder='results',outpars=[],par=parameter({})):


    #Try to get filename from par
    if not fname:
        if hasattr(par,'imname'):
            fname = par.imname[par.imname.rfind('/')+1:]
        else:
            raise NameError('No filename given')


    #Generate folder if necessary
    if folder:
        if not os.path.exists(folder):
            os.makedirs(folder)

            
    #Remove ending if necessary
    pos = fname.find('.')
    if pos>-1:
        fname = fname[:pos]
    
    #Concatenate folder and filename
    outname = fname
    if folder:
        outname = folder + '/' +  outname
        #Remove double //
        outname = outname.replace('//','/')
    
    #Check for keyword DOT
    if outname.find('DOT')>-1:
        raise NameError('Keyword "DOT" not allowd')
        

    #Add outpars to filename
    for parname in outpars:
        if hasattr(par,parname):
            val = par.__dict__[parname]
            #exec('val = self.par.'+par)
            outname += '__' + parname + '_' + num2str(val)
        else:
            raise NameError('Non-existent parameter: ' + parname)        

    
    return outname        





# Function to export the figure for the paper

def export_paper(res,display=True,mode='u_w',outfolder='paper_results/figures'):



    fig = plt.figure()

    if (mode == 'u_w') or (mode == 'u_w_highres'):
           
        
        
        urg_ext = (np.min(res.u0)-0.05,np.max(res.u0)*1.1)
        urg = (np.min(res.u0),np.max(res.u0))
        ax = fig.add_subplot(131, projection='3d')
        if mode == 'u_w':
            plt.title('Data: n_T = ' + str(res.par.n_tmeas) + ', \u03C3 = ' + str(res.par.nstd))
            outpars = ['nstd','a_u0','n_tmeas']
            rstride=1
            cstride=1
        else:
            plt.title('Data: n_T = ' + str(res.par.n_tmeas) + ', nx = ' + str(res.par.nx) + ', nt = ' + str(res.par.nt))
            outpars = ['nstd','a_u0','nx']
            if res.par.nx>1000:
                rstride=100
            else:
                rstride=10
            if res.par.nt>1000:
                cstride=100
            else:
                cstride=10


        plt.xlabel("space")
        plt.ylabel(" time")
        surf=ax.plot_surface(res.x[::rstride,:], res.t[:,res.mtimes], res.mu0[0,::rstride,:], cmap='viridis', edgecolor='none',linewidth=0,vmin=urg[0],vmax=urg[1],rstride=1,cstride=1)
        ax.set_zlim(urg_ext)
    #    plt.clim(urg)
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)


        ax = fig.add_subplot(132, projection='3d')
        plt.title('Approximate state')
        plt.xlabel("space")
        plt.ylabel(" time")
        surf=ax.plot_surface(res.x[::rstride,:], res.t[:,::cstride], res.u_est[0,::rstride,::cstride], cmap='viridis', edgecolor='none',linewidth=0,vmin=urg[0],vmax=urg[1],rstride=1,cstride=1)
        ax.set_zlim(urg_ext)
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)
        

        nl_stride = 1
        if res.par.nx>100:
            nl_stride = 10
        if res.par.nx>1000:
            nl_stride = 100

        plt.subplot(133)
        plt.title('Nonlinearity')
        plt.plot(res.y[::nl_stride],res.fy0[::nl_stride],'.',label='exact')
        plt.plot(res.y[::nl_stride],res.fy[::nl_stride],label='approx.')
        #plt.plot(res.y,res.fy_init,'-.',label='init')
        plt.legend()    
        
        plt.gcf().set_size_inches(16, 4)

    if mode == 'u_w_phi':
           
        outpars = ['nstd','a_u0','n_tmeas']
        
        urg_ext = (np.min(res.u0)-0.05,np.max(res.u0)*1.1)
        urg = (np.min(res.u0),np.max(res.u0))
        ax = fig.add_subplot(141, projection='3d')
        plt.title('Data: n_T = ' + str(res.par.n_tmeas) + ', \u03B4 = ' + str(res.par.nstd)) # \u03C3
        plt.xlabel("space")
        plt.ylabel(" time")
        surf=ax.plot_surface(res.x, res.t[:,res.mtimes], res.mu0[0,...], rstride=1, cstride=1, cmap='viridis', edgecolor='none',linewidth=0,vmin=urg[0],vmax=urg[1])
        ax.set_zlim(urg_ext)
    #    plt.clim(urg)
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)



        ax = fig.add_subplot(142, projection='3d')
        plt.title('Approximate state')
        plt.xlabel("space")
        plt.ylabel(" time")
        surf=ax.plot_surface(res.x, res.t, res.u_est[0,...], rstride=1, cstride=1, cmap='viridis', edgecolor='none',linewidth=0,vmin=urg[0],vmax=urg[1])
        ax.set_zlim(urg_ext)
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)
        
        
        #This gives cshift as solution to min_c (fy0-M(y0) - c)^2 + (phi0 - phi - c)^2
        cshift  = ( (res.fy0 - res.fy).sum() + (res.phi0 - res.phi_est).sum() ) / (np.prod(res.fy0.shape) + np.prod(res.phi0.shape))
            

        plt.subplot(143)
        plt.title('Nonlinearity')
        plt.plot(res.y,res.fy0,'.',label='exact')
        plt.plot(res.y,res.fy+cshift,label='approx.')
        #plt.plot(res.y,res.fy_init,'-.',label='init')
        plt.legend() 



        plt.figure(fig.number)
        plt.subplot(144)
        plt.title('Parameter')
        plt.plot(res.x,res.phi0[0,:,0],".",label='exact')
        plt.plot(res.x,res.phi_est[0,:,0]+cshift,label='approx.')
        plt.plot(res.x,res.phi_init[0,:,0],"-.",label='initial')
        plt.legend()

        plt.gcf().set_size_inches(20, 4)


    if mode == 'u_w_phi_multsamples':
           
        outpars = ['nstd','nsamples']
        

        
        
        #This gives cshift as solution to min_c (fy0-M(y0) - c)^2 + (phi0 - phi - c)^2
        cshift  = ( (res.fy0 - res.fy).sum() + (res.phi0 - res.phi_est).sum() ) / (np.prod(res.fy0.shape) + np.prod(res.phi0.shape))
            

        plt.subplot(131)
        plt.title('Nonlinearity (Error = ' + format(res.f_err[-1],'.2e') + ')')
        plt.plot(res.y,res.fy0,'.',label='exc')
        plt.plot(res.y,res.fy+cshift,label='approx')
        plt.legend() 



        plt.figure(fig.number)
        plt.subplot(132)
        if res.phi0.shape[0]==1:
            plt.title('Exact parameters (' + str(res.phi0.shape[0]) + ' sample)')
        else:
            plt.title('Exact parameters (' + str(res.phi0.shape[0]) + ' samples)')
        for i in range(res.phi0.shape[0]):
            plt.plot(res.x,res.phi0[i,:,0])


        plt.subplot(133)
        plt.title('Approximate parameters')
        for i in range(res.phi0.shape[0]):
            plt.plot(res.x,res.phi_est[i,:,0]+cshift)



        plt.gcf().set_size_inches(16, 4)

    if mode == 'nonlinearity_test':


        outpars = ['nonlinearity','a_u0','approx_type']
        

        #This gives cshift as solution to min_c (fy0-M(y0) - c)^2 + (phi0 - phi - c)^2
        cshift  = ( (res.fy0 - res.fy).sum() + (res.phi0 - res.phi_est).sum() ) / (np.prod(res.fy0.shape) + np.prod(res.phi0.shape))
            

        plt.subplot(111)
        plt.title('Nonlinearity (Error = ' + format(res.f_err[-1],'.2e') + ')')
        plt.plot(res.y,res.fy0,'.',label='exc')
        plt.plot(res.y,res.fy+cshift,label='approx')
        #plt.plot(res.y,res.fy_init,'-.',label='init')
        plt.legend() 


        plt.gcf().set_size_inches(4, 4)


    
    if display:
        fig.show()
    
    
    #Save figure
    outname = output_name(fname=mode,folder=outfolder,outpars=outpars,par=res.par)
    plt.savefig(outname + '.pdf',format='pdf')
    
    
    
    return fig
        
        


# Object to store output
class output(object):
    
    def __init__(self,par=parameter({})):
        
        self.par = par
    

    def save(self,fname='',outpars=[],folder=''):
    
        #Get name
        self.outname = output_name(outpars=outpars,fname=fname,folder=folder,par=self.par)
        #Save data
        psave(self.outname,self) 
        #Save figure
        self.show(display=False)
        plt.savefig(self.outname + '.pdf',format='pdf')
        plt.close('all')
        
        if self.par.nsamples>1:
            #Save data figure
            self.show_data(display=False)
            plt.savefig(self.outname + '_data.pdf',format='pdf')
            plt.close('all')
            

    def show(self,display=True):

        #y = np.linspace(np.amin(self.u0),np.amax(self.u0),nx).selfhape(-1,1)

        #Set stride
        rstride = 1
        cstride = 1
        if len(self.x)>100:
            rstride = 10
        if len(self.x)>1000:
            rstride = 100
        if len(self.t[0,:])>100:
            cstride = 10
        if len(self.t[0,:])>1000:
            cstride = 100
            
        
        fig = plt.figure()
       
        plt.figure(fig.number)
        plt.subplot(331)
        plt.title('Parameter')
        plt.plot(self.phi0[0,::rstride,0],".",label='exct')
        plt.plot(self.phi_est[0,::rstride,0],label='approx')
        plt.plot(self.phi_init[0,::rstride,0],"-.",label='init')
        plt.legend()

        
        urg_ext = (np.min(self.u0)-0.05,np.max(self.u0)*1.1)
        urg = (np.min(self.u0),np.max(self.u0))
        ax = fig.add_subplot(334, projection='3d')
        plt.title('Data')
        plt.xlabel("x")
        plt.ylabel("t")
        surf=ax.plot_surface(self.x, self.t[:,self.mtimes], self.mu0[0,...], rstride=rstride, cstride=cstride, cmap='viridis', edgecolor='none',linewidth=0,vmin=urg[0],vmax=urg[1])
        ax.set_zlim(urg_ext)
        ax.set_xlim((self.x[0],self.x[-1]))
    #    plt.clim(urg)
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)



        ax = fig.add_subplot(335, projection='3d')
        plt.title('Approximate state')
        plt.xlabel("x")
        plt.ylabel("t")
        surf=ax.plot_surface(self.x, self.t, self.u_est[0,...], rstride=rstride, cstride=cstride, cmap='viridis', edgecolor='none',linewidth=0,vmin=urg[0],vmax=urg[1])
        ax.set_zlim(urg_ext)
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)
        
        ax = fig.add_subplot(336, projection='3d')
        plt.title('State error (plot)')
        plt.xlabel("x")
        plt.ylabel("t")
        surf=ax.plot_surface(self.x, self.t, np.abs(self.u0[0,...] - self.u_est[0,...]), rstride=rstride, cstride=cstride, cmap='viridis', edgecolor='none',linewidth=0,vmin=urg[0],vmax=urg[1])
        ax.set_zlim(urg_ext)
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)


        plt.subplot(337)
        plt.title('Nonlinearity')
        plt.plot(self.y[::cstride],self.fy0[::cstride],'.',label='exc')
        plt.plot(self.y[::cstride],self.fy[::cstride],label='approx')
        #plt.plot(self.y,self.fy_init,'-.',label='init')
        plt.legend()    


        plt.subplot(332)
        plt.title('Parameter error')
        plt.plot(self.phi_err)
        plt.yscale('log')

        plt.subplot(333)
        plt.title('State error (curve)')
        plt.plot(self.u_err)
        plt.yscale('log')

        plt.subplot(338)
        plt.title('Nonlinearity error')
        plt.plot(self.f_err)
        plt.yscale('log')


        plt.subplot(339)
        plt.title('Objective value')
        plt.plot(self.ob_val)
        plt.yscale('log')


        plt.gcf().set_size_inches(10, 10)
        if display:
            fig.show()
        
        return fig


    def show_data(self,display=True,ndata=5):

        #y = np.linspace(np.amin(self.u0),np.amax(self.u0),nx).selfhape(-1,1)

        fig = plt.figure()
        
        ndata = min(ndata,self.par.nsamples)
        
        step = int(np.ceil(self.par.nsamples/ndata))
        
        for sample in range(0,ndata):
            
            #Parameter

            
            plt.figure(fig.number)
            plt.subplot(3,ndata,sample+1)
            plt.title('Parameter (' + str(sample*step) + ')')
            plt.plot(self.phi0[sample*step,:,0],".",label='exct')
            plt.plot(self.phi_est[sample*step,:,0],label='approx')
            plt.plot(self.phi_init[sample*step,:,0],"-.",label='init')
            plt.legend()

            
            
            # Data
            urg_ext = (np.min(self.u0)-0.05,np.max(self.u0)*1.1)
            urg = (np.min(self.u0),np.max(self.u0))
            
            ax = fig.add_subplot(3,ndata,ndata + sample+1, projection='3d')
            plt.title('Data (' + str(sample*step) + ')')
            plt.xlabel("x")
            plt.ylabel("t")
            surf=ax.plot_surface(self.x, self.t[:,self.mtimes], self.mu0[sample*step,...], rstride=1, cstride=1, cmap='viridis', edgecolor='none',linewidth=0,vmin=urg[0],vmax=urg[1])
            ax.set_zlim(urg_ext)
        #    plt.clim(urg)
            fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)


            #State
            ax = fig.add_subplot(3,ndata,2*ndata + sample+1, projection='3d')
            plt.title('Approximate state (' + str(sample*step) + ')')
            plt.xlabel("x")
            plt.ylabel("t")
            surf=ax.plot_surface(self.x, self.t, self.u_est[sample*step,...], rstride=1, cstride=1, cmap='viridis', edgecolor='none',linewidth=0,vmin=urg[0],vmax=urg[1])
            ax.set_zlim(urg_ext)
            fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)
            

        plt.gcf().set_size_inches(10, 3.3*ndata)
        
        if display:
            fig.show()
        
        return fig

        
#Class for parameter testing
class partest(object):

    def __init__(self,method,fname,fixpars={},testpars={},namepars=[],folder=''):
    
        
        self.method = method
        self.fixpars = fixpars
        self.testpars = testpars
        self.namepars = namepars
        self.folder = folder
        self.fname = fname
        
        #Dictionaries for error storage
        self.pardict     = {key:[] for key in self.testpars.keys()}
        
        self.f_err       = {key:[] for key in self.testpars.keys()}
        self.u_err       = {key:[] for key in self.testpars.keys()}
        self.phi_err     = {key:[] for key in self.testpars.keys()}
        self.f_m_phi_err = {key:[] for key in self.testpars.keys()}


    def run_test(self):
    
        #Check for conflicts
        for key in self.testpars.keys():
            if key in self.fixpars:
                raise NameError('Double assignement of ' + key)


                
        #Get keys
        testkeys = self.testpars.keys()
        #Iterate over all possible combinations
        for valtuple in list(product(*self.testpars.values())):
            
            
            #Set test values
            for key,val in zip(testkeys,valtuple):
                self.fixpars[key] = val
                
                
            #Print parameter setup
            print('Testing: ')
            print(self.fixpars)
            #Get result
            res = self.method(**self.fixpars)
            #Set parameters for output name from namepars and testkeys
            outpars = list(dict.fromkeys(self.namepars + list(testkeys))) #This removes duplicates
            #Save
            res.save(fname=self.fname,outpars=outpars,folder=self.folder)
            
            #Set test values
            for key,val in zip(testkeys,valtuple):
                self.pardict[key] += [val]
                
                self.u_err[key] += [res.u_err[-1]]
                self.f_err[key] += [res.f_err[-1]]
                self.phi_err[key] += [res.phi_err[-1]]
                self.f_m_phi_err[key] += [res.f_m_phi_err[-1]]
            
        #Store test instance with overall result
        overall_outname = output_name(fname=self.fname,outpars=self.namepars,folder=self.folder,par=res.par)
        self.method = self.method.__name__ # Store only name of method 
        psave(overall_outname + '_overall',self)


#Read value of parameter from file        
def read_parval(fname,parname):

    #Set position of value    
    star = fname.find('_'+parname+'_')+len('_'+parname+'_')
    #Set end position of value
    end = fname[star:].find('__')
    if end == -1:
        end = fname[star:].find('.')
    end += star 
    
    return str2num(fname[star:end])
    

#Return all file names with .pkl extension matching a parameter combination
def get_file_list(basename,pars = {},folder = '.'):


    flist = os.listdir(folder)


    
       
    #Remove non-matching filenames
    for fname in flist[:]:
        if (basename not in fname) or ('.pkl' not in fname): #Basename
            flist.remove(fname)
        else:
            for par in pars.keys():
                #Check parameter name
                if '_' + par + '_' not in fname:
                    flist.remove(fname)
                    break
                else:
                    #Check parameter values
                    valcount = len(pars[par])
                    if valcount>0:
                        for val in pars[par]:
                            if '_' + par + '_' + num2str(val) not in fname: #Parameter value pairs
                                valcount -= 1
                        if valcount == 0: #If no parameter is present
                            flist.remove(fname)
                            break


    return flist
                

#Return a list of file names with .pkl extension matching a parameter combination together with the parameters
def get_file_par_list(basename,pars = {},folder = '.'):

    #Get list of files matching pattern
    flist = get_file_list(basename,pars = pars,folder = folder)
    
    parnames = list(pars)
    parvals = []
    for fname in flist:
        parval = []
        for parname in parnames:
            parval.append(read_parval(fname,parname))
        parvals.append(parval[:])
    
    return flist,parnames,parvals

  
#Show image-like data
def imshow(x,stack=1,fig=0,title=0,colorbar=1,cmap='gray',vrange=[],x_labels=[],y_labels=[],x_label='',y_label=''):

    extent = (-0.5,x.shape[0]-0.5,-0.5,x.shape[1]-0.5)

    try:

        if x.ndim>2 and stack:
            x = imshowstack(x)

        if not fig:
            fig = plt.figure()
            
        plt.figure(fig.number)
        
        if not vrange:
            plt.imshow(x,cmap=cmap,interpolation='none',extent=extent)
        else:
            plt.imshow(x,cmap=cmap,vmin=vrange[0],vmax=vrange[1],interpolation='none')
        if colorbar:
            plt.colorbar()
        if title:
            plt.title(title)
            
        if not y_labels==[]:
            ax = fig.axes[0]
            ax.set_yticks(np.arange(0,y_labels.shape[0]))
            labels = [item.get_text() for item in ax.get_yticklabels()]
            for i in range(len(y_labels)):
                labels[i] = y_labels[i]
            ax.set_yticklabels(labels)            
        if not x_labels==[]:
            ax = fig.axes[0]
            ax.set_xticks(np.arange(0,x_labels.shape[0]))
            labels = [item.get_text() for item in ax.get_xticklabels()]
            for i in range(len(x_labels)):
                labels[i] = x_labels[i]
            ax.set_xticklabels(labels)            
            ax.set_xlabel('xx')
        if x_label:
            ax = fig.axes[0]
            ax.set_xlabel(x_label)
        if y_label:
            ax = fig.axes[0]
            ax.set_ylabel(y_label)


        fig.show()
        
    except Exception as e:
        print('Display error. I assume that no display is available and continue...')
        print('Error message: ')
        print(e)
        fig = 0
    
    return fig


#Function to evaluate results for given parameter range(s)
#Example usage:
#pdl_supp.eval_results('a_pde_5_',pars={'a_w':[],'a_u':[]},folder='script_results/model_parameter_quadratic_u_w')
def eval_results(basename,pars={},folder='.',show_all=True,export_paper=False):

    #Get sortet list of filenames, parnames and values
    flist,parnames,parvals = get_file_par_list(basename,pars=pars,folder=folder)
        
    
    if not flist:
        raise ValueError('No files matching search pattern found')
            
    #Determine parmeter values
    p1 = {}
    p2 = {}
    for pos,parval in enumerate(parvals):
        if parval[0] not in p1:
            p1[parval[0]] = []
        if parval[1] not in p2:
            p2[parval[1]] = []
            
        p1[parval[0]].append(pos)
        p2[parval[1]].append(pos)
        

        
    #p1list = p1.keys()
    #p2list = p2.keys()
    p1list = list(p1)
    p2list = list(p2)
    p1list.sort()
    p2list.sort()
    #p1list = sorted(p1list)
    #p2list = sorted(p2list)
    

    phi_err = np.zeros( ( len(p1list),len(p2list) ) )
    f_err = np.zeros( ( len(p1list),len(p2list) ) )
    f_m_phi_err = np.zeros( ( len(p1list),len(p2list) ) )
    
    phi_err_min = 1e100
    f_err_min = 1e100
    f_m_phi_err_min = 1e100
    
    
    x_labels = np.zeros( ( len(p2list)) )
    y_labels = np.zeros( ( len(p1list)) )
    
    for k1 in p1.keys():
        for pos in p1[k1]:
            for k2 in p2.keys():
                if pos in p2[k2]:
                    
                    fullname = folder + '/' + flist[pos]
                    fullname = fullname.replace('//','/')    
                    res = pload(fullname)
                    
                    if hasattr(res, 'ob_val'):
                        if res.ob_val[0]<0.5*res.ob_val[-1]:
                            print('Warning: bad concergence: obval[0]: '+str(res.ob_val[0]) + ' ob_val[-1]' + str(res.ob_val[-1]))
                            print(parnames)
                            print(parvals[pos])
                        
                    phi_err[p1list.index(k1),p2list.index(k2)] = res.phi_err[-1]
                    f_err[p1list.index(k1),p2list.index(k2)] = res.f_err[-1]
                    f_m_phi_err[p1list.index(k1),p2list.index(k2)] = res.f_m_phi_err[-1]
                    
                    #x_labels[p1list.index(k1)] = k1
                    #y_labels[p2list.index(k2)] = k2
                    x_labels[p2list.index(k2)] = k2
                    y_labels[p1list.index(k1)] = k1

                    #Store optimal value
                    if res.phi_err[-1] < phi_err_min:
                        phi_err_min = res.phi_err[-1]
                        phi_err_optfile = flist[pos]

                    if res.f_err[-1] < f_err_min:
                        f_err_min = res.f_err[-1]
                        f_err_optfile = flist[pos]

                    if res.f_m_phi_err[-1] < f_m_phi_err_min:
                        f_m_phi_err_min = res.f_m_phi_err[-1]
                        f_m_phi_err_optfile = flist[pos]

    

    #Set non-existent error to be maximal
    

    phi_err[phi_err==0] = 2.0*phi_err.max() + 1.0
    f_err[f_err==0] = 2.0*f_err.max() + 1.0
    f_m_phi_err[f_m_phi_err==0] = 2.0*f_m_phi_err.max() + 1.0
    
    #Flip labels to make them compatible with imshow
    y_labels = np.flip(y_labels)
    
    if show_all:
    
        imshow(np.log(phi_err),cmap='hot',title='log(phi_err)',x_labels=x_labels,y_labels=y_labels,x_label=parnames[1],y_label=parnames[0])
        imshow(np.log(f_err),cmap='hot',title='log(f_err)',x_labels=x_labels,y_labels=y_labels,x_label=parnames[1],y_label=parnames[0])
        imshow(np.log(f_m_phi_err),cmap='hot',title='log(f_m_phi_err)',x_labels=x_labels,y_labels=y_labels,x_label=parnames[1],y_label=parnames[0])
    
    print('Optimal log phi_err is ' + str(np.log(phi_err_min)) + ' achieved with ')
    print(phi_err_optfile)
    fullname = folder + '/' + phi_err_optfile
    fullname = fullname.replace('//','/')    
    res = pload(fullname)
    if show_all:    
        fig = res.show()
        fig.suptitle(phi_err_optfile)

    print('Optimal log f_err is ' + str(np.log(f_err_min)) + ' achieved with ')
    print(f_err_optfile)
    fullname = folder + '/' + f_err_optfile
    fullname = fullname.replace('//','/')    
    res = pload(fullname)    
    if show_all:    
        fig = res.show()
        fig.suptitle(f_err_optfile)

    print('Optimal log f_m_phi_err is ' + str(np.log(f_m_phi_err_min)) + ' achieved with ')
    print(f_m_phi_err_optfile)
    fullname = folder + '/' + f_m_phi_err_optfile
    fullname = fullname.replace('//','/')    
    res = pload(fullname)    
    if show_all:    
        fig = res.show()
        fig.suptitle(f_m_phi_err_optfile)
    
    
    if export_paper:
        print('###########################')
        print('Printing result of optimal parameter w.r.t. f_err for paper:')
        
        fullname = folder + '/' + f_err_optfile
        fullname = fullname.replace('//','/')    
        res = pload(fullname)    
        
        
        print('Nonlinearity error:')
        print(res.f_err[-1])
        
        print('Parameter error:')
        print(res.phi_err[-1])

        print('State error:')
        print(res.u_err[-1])
        
    
    
    
    return phi_err,f_err,f_m_phi_err


    

#Close all figures
def closefig():
    plt.close('all')


#Convert number to string and reverse
def num2str(x):
    return str(x).replace('.','DOT')


def str2num(s):
    return float(s.replace('DOT','.'))


