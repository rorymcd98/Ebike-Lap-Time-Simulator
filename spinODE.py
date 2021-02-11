import numpy as np
import openmdao.api as om

#Equations 13a,b,c, 14, 15, 16, 18(3)
class spin(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('rt', val=0.1, desc='wheel toroid radius', units=None)
        self.add_input('r', val=0.3, desc='wheel radius', units='m')
        self.add_input('Iwy', val=0.7, desc='wheelspin moment of inertia', units='kg*m**2')

        #states
        self.add_input('Phi', val=np.zeros(nn), desc='roll', units='rad')
        self.add_input('Phidot', val=np.zeros(nn), desc='roll rate', units='rad/s')
        self.add_input('Beta', val=np.zeros(nn), desc='drift angle', units='rad')
        self.add_input('Betadot', val=np.zeros(nn), desc='drift rate', units='rad/s')
        self.add_input('alpha', val=np.zeros(nn), desc='relative heading', units='rad')
        self.add_input('alphadot', val=np.zeros(nn), desc='relative heading rate', units='rad/s')
        self.add_input('nu', val=np.zeros(nn), desc='road curvature x-z', units='1/m')
        self.add_input('k', val=np.zeros(nn), desc='road curvature x-y', units='1/m')
        self.add_input('tau', val=np.zeros(nn), desc='road curvature torsional', units='1/m')
        self.add_input('V', val=np.zeros(nn), desc='forward velocity', units='m/s')
        self.add_input('Vdot', val=np.zeros(nn), desc='forward acceleration', units='m/s**2')
        self.add_input('n', val=np.zeros(nn), desc='road lateral position', units='m')
        self.add_input('z', val=np.zeros(nn), desc='vertical displacement', units='m')
        self.add_input('sdot', val=np.zeros(nn), desc='road longitudinal velocity', units='m/s')
        self.add_input('Omega_z', val=np.zeros(nn), desc='yaw rate', units='rad/s')
        self.add_input('Omegadot_z', val=np.zeros(nn), desc='yaw rate2', units='rad/s**2')
        self.add_input('Fx', val=np.zeros(nn), desc='longitudinal tyre force', units='N')
        self.add_input('mu_r', val=0.015, desc='tyre rolling resistance', units=None)
        self.add_input('omega_w', val=np.zeros(nn), desc='wheel spin', units='rad/s')
        self.add_input('tau_w', val=np.zeros(nn), desc='driving torque', units='N*m')
        self.add_input('N', val=np.zeros(nn), desc='tyre load', units='N')
        self.add_input('Fy', val=np.zeros(nn), desc='lateral tyre load', units='N')

        #outputs
        self.add_output('omegadot_w', val=np.zeros(nn), desc='', units=None)

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=True)

    def compute(self, inputs, outputs):
        rt = inputs['rt']
        r = inputs['r']
        Iwy = inputs['Iwy']

        Phi = inputs['Phi']
        Phidot = inputs['Phidot']
        Beta = inputs['Beta']
        Betadot = inputs['Betadot']
        alpha = inputs['alpha']
        alphadot = inputs['alphadot']
        nu = inputs['nu']
        tau = inputs['tau']
        V = inputs['V']
        n = inputs['n']
        k = inputs['k']
        z = inputs['z']
        sdot = inputs['sdot']
        Omega_z = inputs['Omega_z']
        Omegadot_z = inputs['Omegadot_z']
        Fx = inputs['Fx']
        tau_w = inputs['tau_w']
        mu_r = inputs['mu_r']
        Vdot = inputs['Vdot']
        N = inputs['N']
        Fy = inputs['Fy']

        Omega_y = sdot*(nu*np.cos(alpha)-tau*np.sin(alpha))

        sddot = - (V*(np.sin(alpha)*Betadot - np.sin(alpha)*alphadot + np.cos(alpha)*Beta*alphadot))/(k*n - 1) - Vdot*(np.cos(alpha) + np.sin(alpha)*Beta)/(k*n - 1)
        Omegadot_y = (V*(tau*np.cos(alpha)*alphadot + nu*np.sin(alpha)*alphadot)*(np.cos(alpha) + np.sin(alpha)*Beta))/(k*n - 1) - (V*(nu*np.cos(alpha) - tau*np.sin(alpha))*(np.sin(alpha)*Betadot - np.sin(alpha)*alphadot + np.cos(alpha)*Beta*alphadot))/(k*n - 1) - ((nu*np.cos(alpha) - tau*np.sin(alpha))*Vdot*(np.cos(alpha) + np.sin(alpha)*Beta))/(k*n - 1)        

        tau_r = (N*np.cos(Phi)+Fy*np.sin(Phi))*mu_r

        outputs['omegadot_w'] = (tau_w - tau_r - Fx*(r-rt*(1-np.cos(Phi))-z*np.cos(Phi)) \
                            + Iwy*(Omegadot_y*np.cos(Phi)+np.sin(Phi)*Omegadot_z+(Omega_z*np.cos(Phi)-Omega_y*np.sin(Phi))*Phidot))/Iwy