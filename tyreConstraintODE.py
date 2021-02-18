import openmdao.api as om
import numpy as np

#Equations 18, 19
class tyreConstraint(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('mu_r', val=0.015, desc='tyre rolling resistance', units=None)
        self.add_input('mu_x', val=0.8, desc='tyre longitudinal adherence', units=None)
        self.add_input('mu_y', val=0.6, desc='tyre lateral adherence', units=None)
        self.add_input('Ks', val=2e5, desc='tyre radial stiffness', units=None)
        self.add_input('Kc', val=1e3, desc='tyre damping stiffness', units=None)

        #states
        self.add_input('N', val=np.zeros(nn), desc='tyre load', units='N')
        self.add_input('Fx', val=np.zeros(nn), desc='tyre longitudinal force', units='N')
        self.add_input('Fy', val=np.zeros(nn), desc='tyre lateral force', units='N')

        #outputs
        self.add_output('TC', val=np.zeros(nn), desc='tyre constraint', units=None)

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='TC', wrt='N', rows=arange, cols=arange)
        self.declare_partials(of='TC', wrt='Fx', rows=arange, cols=arange)
        self.declare_partials(of='TC', wrt='Fy', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        mu_r = inputs['mu_r']
        mu_x = inputs['mu_x']
        mu_y = inputs['mu_y']
        Ks = inputs['Ks']
        Kc = inputs['Kc']
        N = inputs['N']
        Fx = inputs['Fx']
        Fy = inputs['Fy']

        outputs['TC'] = (Fx/(N*mu_x))**2 + (Fy*(N*mu_y))**2

    def compute_partials(self, inputs, jacobian):
        mu_r = inputs['mu_r']
        mu_x = inputs['mu_x']
        mu_y = inputs['mu_y']
        Ks = inputs['Ks']
        Kc = inputs['Kc']
        N = inputs['N']
        Fx = inputs['Fx']
        Fy = inputs['Fy']

        jacobian['TC','N']=2*Fy**2*N*mu_y**2 - (2*Fx**2)/(N**3*mu_x**2)
        jacobian['TC','Fx']=(2*Fx)/(N**2*mu_x**2)
        jacobian['TC','Fy']=2*Fy*N**2*mu_y**2