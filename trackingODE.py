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

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=True)

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