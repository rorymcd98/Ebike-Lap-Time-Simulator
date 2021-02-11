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

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=True)

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