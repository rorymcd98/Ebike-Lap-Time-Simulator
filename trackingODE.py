import openmdao.api as om
import numpy as np

#Equations 18, 19
class tracking(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants

        #states
        self.add_input('V', val=np.zeros(nn), desc='forward velocity', units='m/s')
        self.add_input('Beta', val=np.zeros(nn), desc='drift angle', units='rad')
        self.add_input('alpha', val=np.zeros(nn), desc='relative heading', units='rad')
        self.add_input('n', val=np.zeros(nn), desc='road lateral position', units='m')
        self.add_input('k', val=np.zeros(nn), desc='road curvature x-y', units='1/m')
        self.add_input('Omega_z', val=np.zeros(nn), desc='yaw rate', units='rad/s')

        #outputs
        self.add_output('sdot', val=np.zeros(nn), desc='road longitudinal velocity', units='m/s')
        self.add_output('ndot', val=np.zeros(nn), desc='road lateral velocity', units='m/s')
        self.add_output('alphadot', val=np.zeros(nn), desc='relative heading rate', units='rad/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='sdot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='Beta', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='n', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='k', rows=arange, cols=arange)
        self.declare_partials(of='ndot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='ndot', wrt='Beta', rows=arange, cols=arange)
        self.declare_partials(of='ndot', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='Beta', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='n', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='k', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='Omega_z', rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        V = inputs['V']
        Beta = inputs['Beta']
        alpha = inputs['alpha']
        n = inputs['n']
        k = inputs['k']
        Omega_z = inputs['Omega_z']

        outputs['sdot'] = V*(Beta*np.sin(alpha)+np.cos(alpha))/(1-n*k)
        outputs['ndot'] = V*np.sin(alpha)-V*Beta*np.cos(alpha)
        outputs['alphadot'] = Omega_z - k*(V*np.cos(alpha)+V*Beta*np.sin(alpha))/(1-n*k)

    def compute_partials(self, inputs, jacobian):
        V = inputs['V']
        Beta = inputs['Beta']
        alpha = inputs['alpha']
        n = inputs['n']
        k = inputs['k']
        Omega_z = inputs['Omega_z']

        jacobian['sdot','V']=-(np.cos(alpha) + Beta*np.sin(alpha))/(k*n - 1)
        jacobian['sdot','Beta']=-(V*np.sin(alpha))/(k*n - 1)
        jacobian['sdot','alpha']=(V*(np.sin(alpha) - Beta*np.cos(alpha)))/(k*n - 1)
        jacobian['sdot','n']=(V*k*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2
        jacobian['sdot','k']=(V*n*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2
        jacobian['ndot','V']=np.sin(alpha) - Beta*np.cos(alpha)
        jacobian['ndot','Beta']=-V*np.cos(alpha)
        jacobian['ndot','alpha']=V*np.cos(alpha) + Beta*V*np.sin(alpha)
        jacobian['alphadot','V']=(k*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)
        jacobian['alphadot','Beta']=(V*k*np.sin(alpha))/(k*n - 1)
        jacobian['alphadot','alpha']=-(k*(V*np.sin(alpha) - Beta*V*np.cos(alpha)))/(k*n - 1)
        jacobian['alphadot','n']=-(k**2*(V*np.cos(alpha) + Beta*V*np.sin(alpha)))/(k*n - 1)**2
        jacobian['alphadot','k']=(V*np.cos(alpha) + Beta*V*np.sin(alpha))/(k*n - 1) - (k*n*(V*np.cos(alpha) + Beta*V*np.sin(alpha)))/(k*n - 1)**2
        jacobian['alphadot','Omega_z']=1