import numpy as np
import openmdao.api as om

class zEquation(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('M', val=300.0, desc='total mass', units='kg')
        self.add_input('g', val=9.81, desc='gravity', units='m/s**2')
        self.add_input('rho', val=1.2, desc='air density', units='kg/m**3')
        self.add_input('Cl', val=0.03, desc='lift coefficient', units=None)
        self.add_input('rt', val=0.1, desc='wheel toroid radius', units=None)
        self.add_input('h', val=0.6, desc='CoG height', units=None)

        #states
        self.add_input('Phi', val=np.zeros(nn), desc='roll', units='rad')
        self.add_input('Phidot', val=np.zeros(nn), desc='roll rate', units='rad/s')
        self.add_input('Phiddot', val=np.zeros(nn), desc='roll rate2', units='rad/s**2')
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
        self.add_input('N', val=np.zeros(nn), desc='tyre load', units='N')

        #outputs
        self.add_output('zddot', val=np.zeros(nn), desc='vertical acceleration', units='m/s**2')

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=True)

    def apply_nonlinear(self, inputs, outputs, residuals):
        M = inputs['M']
        g = inputs['g']
        rho = inputs['rho']
        Cl = inputs['Cl']
        rt = inputs['rt']
        h = inputs['h']


        Phi = inputs['Phi']
        Phidot = inputs['Phidot']
        Phiddot = inputs['Phiddot']
        Beta = inputs['Beta']
        Betadot = inputs['Betadot']
        alpha = inputs['alpha']
        alphadot = inputs['alphadot']
        nu = inputs['nu']
        tau = inputs['tau']
        V = inputs['V']
        Vdot = inputs['Vdot']
        n = inputs['n']
        k = inputs['k']
        z = inputs['z']
        sdot = inputs['sdot']
        Omega_z = inputs['Omega_z']
        N = inputs['N']


        zddot = outputs['zddot']

        sddot = - (V*(np.sin(alpha)*Betadot - np.sin(alpha)*alphadot + np.cos(alpha)*Beta*alphadot))/(k*n - 1) - Vdot*(np.cos(alpha) + np.sin(alpha)*Beta)/(k*n - 1)
        

        Omega_x = sdot*(nu*np.sin(alpha)+tau*np.cos(alpha))
        Omega_y = sdot*(nu*np.cos(alpha)-tau*np.sin(alpha))

        Omegadot_x = - ((tau*np.cos(alpha) + nu*np.sin(alpha))*Vdot*(np.cos(alpha) + np.sin(alpha)*Beta))/(k*n - 1) - (V*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.sin(alpha)*Betadot - np.sin(alpha)*alphadot + np.cos(alpha)*Beta*alphadot))/(k*n - 1) - (V*(nu*np.cos(alpha)*alphadot - tau*np.sin(alpha)*alphadot)*(np.cos(alpha) + np.sin(alpha)*Beta))/(k*n - 1)

        V_x = V
        V_y = -V*Beta

        Vdot_z = sddot*tau*n

        L=0.5*rho*Cl*V**2

        residuals['zddot'] = M*zddot+M*Phiddot*np.sin(Phi)*(h-rt)+M*np.cos(Phi)*(h-rt)*Phidot**2+2*M*np.cos(Phi)*Omega_x*(h-rt)*Phidot \
                        + M*np.sin(Phi)*(h-rt)*Omegadot_x + M*Vdot_z + M*np.cos(Phi)*(h-rt)*(Omega_x**2+Omega_y**2)+M*Omega_y*Omega_z*(h-rt)*np.sin(Phi) \
                        - M*(-rt+z)*Omega_x**2 + M*Omega_x*V_y - M*(-rt+z)*Omega_y**2 - M*Omega_y*V_x - M*g + N + L*np.cos(Phi)
