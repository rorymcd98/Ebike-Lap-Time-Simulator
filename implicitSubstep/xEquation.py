import numpy as np
import openmdao.api as om

class xEquation(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('M', val=300.0, desc='total mass', units='kg')
        self.add_input('rt', val=0.1, desc='wheel toroid radius', units=None)
        self.add_input('rho', val=1.2, desc='air density', units='kg/m**3')
        self.add_input('Cd', val=0.41, desc='drag coefficient', units=None)
        self.add_input('h', val=0.6, desc='CoG height', units=None)
        self.add_input('g', val=9.81, desc='gravity', units='m/s**2')
        

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
        self.add_input('sig', val=np.zeros(nn), desc='road slope', units='rad')
        self.add_input('V', val=np.zeros(nn), desc='forward velocity', units='m/s')
        self.add_input('n', val=np.zeros(nn), desc='road lateral position', units='m')
        self.add_input('z', val=np.zeros(nn), desc='vertical displacement', units='m')
        self.add_input('zdot', val=np.zeros(nn), desc='vertical velocity', units='m/s')
        self.add_input('sdot', val=np.zeros(nn), desc='road longitudinal velocity', units='m/s')
        self.add_input('Omega_z', val=np.zeros(nn), desc='yaw rate', units='rad/s')
        self.add_input('Omegadot_z', val=np.zeros(nn), desc='yaw rate2', units='rad/s**2')
        self.add_input('Fx', val=np.zeros(nn), desc='longitudinal tyre force', units='N')
        
        #outputs
        self.add_output('Vdot', val=np.ones(nn), desc='forward acceleration', units='m/s**2', lower = -4, upper = 4)

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=True)

    def apply_nonlinear(self, inputs, outputs, residuals):
        M = inputs['M']
        rt = inputs['rt']
        h = inputs['h']
        rho = inputs['rho']
        Cd = inputs['Cd']
        g = inputs['g']

        Phi = inputs['Phi']
        Phidot = inputs['Phidot']
        Beta = inputs['Beta']
        Betadot = inputs['Betadot']
        alpha = inputs['alpha']
        alphadot = inputs['alphadot']
        nu = inputs['nu']
        tau = inputs['tau']
        sig = inputs['sig']
        V = inputs['V']
        n = inputs['n']
        k = inputs['k']
        z = inputs['z']
        zdot = inputs['zdot']
        sdot = inputs['sdot']
        Omega_z = inputs['Omega_z']
        Omegadot_z = inputs['Omegadot_z']
        Fx = inputs['Fx']
        
        Vdot = outputs['Vdot']

        Omega_x = sdot*(nu*np.sin(alpha)+tau*np.cos(alpha))
        Omega_y = sdot*(nu*np.cos(alpha)-tau*np.sin(alpha))

        Omegadot_y = (V*(tau*np.cos(alpha)*alphadot + nu*np.sin(alpha)*alphadot)*(np.cos(alpha) + np.sin(alpha)*Beta))/(k*n - 1) - (V*(nu*np.cos(alpha) - tau*np.sin(alpha))*(np.sin(alpha)*Betadot - np.sin(alpha)*alphadot + np.cos(alpha)*Beta*alphadot))/(k*n - 1) - ((nu*np.cos(alpha) - tau*np.sin(alpha))*Vdot*(np.cos(alpha) + np.sin(alpha)*Beta))/(k*n - 1)

        V_x = V
        V_y = -V*Beta
        V_z = sdot*tau*n

        Vdot_x = Vdot

        D=0.5*rho*Cd*V**2

        residuals['Vdot'] = M*Vdot_x + (np.cos(Phi)*(-h+rt)-rt+z)*M*Omegadot_y-2*M*(h-rt)*(Omega_z*np.cos(Phi)-Omega_y*np.sin(Phi))*Phidot \
                    -M*np.sin(Phi)*(h-rt)*Omegadot_z + 2*M*Omega_y*zdot - M*Omega_x*Omega_z*(h-rt)*np.cos(Phi) + M*Omega_x*Omega_y*(h-rt)*np.sin(Phi) \
                    + M*((-rt+z)*Omega_x+V_y)*Omega_z + M*Omega_y*V_z - Fx + D + M*g*sig*np.cos(alpha)