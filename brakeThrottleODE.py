import openmdao.api as om
import numpy as np

#Equations 18, 19
class brakeThrottle(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #states
        self.add_input('tau_t', val=np.zeros(nn), desc='motor torque', units='N*m')
        self.add_input('tau_b', val=np.zeros(nn), desc='braking torque', units='N*m')

        #output
        self.add_output('tau_w', val=np.zeros(nn), desc='driving torque', units='N*m')

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=True)
        # Setup partials
        # arange = np.arange(self.options['num_nodes'], dtype=int)

        # #partials
        # self.declare_partials(of='tau_w', wrt='tau_t', rows=arange, cols=arange)
        # self.declare_partials(of='tau_w', wrt='tau_b', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        tau_t = inputs['tau_t']
        tau_b = inputs['tau_b']
        tau_w = tau_t + tau_b

    # def compute_partials(self, inputs, jacobian):
    #     tau_t = inputs['tau_t']
    #     tau_b = inputs['tau_b']

    #     jacobian['tau_w', 'tau_t'] = 1
    #     jacobian['tau_w', 'tau_b'] = 1