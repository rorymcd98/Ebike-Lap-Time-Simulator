import numpy as np
import openmdao.api as om
from scipy.sparse.linalg import gmres



class linearSystem(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.out_guess = np.zeros((4,1))

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('M', val=300.0, desc='total mass', units='kg')
        self.add_input('rt', val=0.1, desc='wheel toroid radius', units='m')
        self.add_input('rho', val=1.2, desc='air density', units='kg/m**3')
        self.add_input('Cd', val=0.41, desc='drag coefficient', units=None)
        self.add_input('h', val=0.6, desc='CoG height', units='m')
        self.add_input('g', val=9.81, desc='gravity', units='m/s**2')
        self.add_input('Cl', val=0.00, desc='lift coefficient', units=None)
        self.add_input('r', val=0.3, desc='wheel radius', units='m')
        self.add_input('Ixx', val=19, desc='roll moment of inertia', units='kg*m**2')
        self.add_input('Iyy', val=25, desc='pitch moment of inertia', units='kg*m**2') 
        self.add_input('Izz', val=25, desc='yaw moment of inertia', units='kg*m**2')
        self.add_input('Igy', val=2.1, desc='gyroscopic moment of inertia', units='kg*m**2')

        #states
        self.add_input('Phi', val=np.zeros(nn), desc='roll', units='rad')
        self.add_input('Phidot', val=np.zeros(nn), desc='roll rate', units='rad/s')
        self.add_input('Beta', val=np.zeros(nn), desc='drift angle', units='rad')
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
        self.add_input('ndot', val=np.zeros(nn), desc='road lateral velocity', units='m/s')
        self.add_input('omega_w', val=np.zeros(nn), desc='wheel spin', units='rad/s')
        self.add_input('Fy', val=np.zeros(nn), desc='lateral tyre load', units='N')
        self.add_input('N', val=np.zeros(nn), desc='tyre load', units='N')
        
        #outputs
        self.add_output('Vdot', val=np.ones(nn), desc='forward acceleration', units='m/s**2')
        self.add_output('Betadot', val=np.zeros(nn), desc='drift rate', units='rad/s')
        self.add_output('Phiddot', val=np.zeros(nn), desc='roll rate2', units='rad/s**2')
        self.add_output('zddot', val=np.zeros(nn), desc='vertical acceleration', units='m/s**2')

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12,show_sparsity=True)

    def apply_nonlinear(self, inputs, outputs, residuals):
        nn = self.options['num_nodes']
        M = inputs['M']*np.ones(nn)
        rt = inputs['rt']*np.ones(nn)
        h = inputs['h']*np.ones(nn)
        rho = inputs['rho']*np.ones(nn)
        Cd = inputs['Cd']*np.ones(nn)
        g = inputs['g']*np.ones(nn)
        Cl = inputs['Cl']*np.ones(nn)
        r = inputs['r']*np.ones(nn)
        Ixx = inputs['Ixx']*np.ones(nn)
        Iyy = inputs['Iyy']*np.ones(nn)
        Izz = inputs['Izz']*np.ones(nn)
        Igy = inputs['Igy']*np.ones(nn)

        Phi = inputs['Phi']
        Phidot = inputs['Phidot']
        Beta = inputs['Beta']
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
        omega_w = inputs['omega_w']
        Fy = inputs['Fy']
        ndot = inputs['ndot']
        N = inputs['N']
        
        Vdot = outputs['Vdot']
        zddot = outputs['zddot']
        Phiddot = outputs['Phiddot']
        Betadot = outputs['Betadot']
        
        A =np.array([[                                                                                                                                                                                                                                                                                                                                                                                 M + (M*(nu*np.cos(alpha) - tau*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha))*(rt - z + np.cos(Phi)*(h - rt)))/(k*n - 1),                                                 np.zeros(nn),                   np.zeros(nn),                                                                                                                                                                                                                                                                                                        (M*V*np.sin(alpha)*(nu*np.cos(alpha) - tau*np.sin(alpha))*(rt - z + np.cos(Phi)*(h - rt)))/(k*n - 1)],
        [                                                                                                                                                                                                                                                                                                                                                                          - Beta*M - (M*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha))*(rt - z + np.cos(Phi)*(h - rt)))/(k*n - 1),                               M*np.cos(Phi)*(h - rt),                   np.zeros(nn),                                                                                                                                                                                                                                                                                                - M*V - (M*V*np.sin(alpha)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(rt - z + np.cos(Phi)*(h - rt)))/(k*n - 1)],
        [                                                                                                                                                                                                                                                                                                                                         - (M*n*tau*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1) - (M*np.sin(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1),                               M*np.sin(Phi)*(h - rt),                   M,                                                                                                                                                                                                                                                                              - (M*V*n*tau*np.sin(alpha))/(k*n - 1) - (M*V*np.sin(Phi)*np.sin(alpha)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha)))/(k*n - 1)],
        [- M*(Beta*(rt - z) - (Beta*np.cos(Phi) - (n*tau*np.sin(Phi)*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1))*(h - rt) - ((tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha))*(- z**2 + 2*rt*z))/(k*n - 1) + ((h - rt)**2*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1) + (2*np.cos(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha))*(rt - z))/(k*n - 1)) - (Ixx*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1), Ixx + M*((h - rt)**2 + np.cos(Phi)*(h - rt)*(rt - z)), M*np.sin(Phi)*(h - rt), - M*(V*(rt - z) - (h - rt)*(V*np.cos(Phi) - (V*n*tau*np.sin(Phi)*np.sin(alpha))/(k*n - 1)) - (V*np.sin(alpha)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(- z**2 + 2*rt*z))/(k*n - 1) + (V*np.sin(alpha)*(h - rt)**2*(tau*np.cos(alpha) + nu*np.sin(alpha)))/(k*n - 1) + (2*V*np.cos(Phi)*np.sin(alpha)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(rt - z))/(k*n - 1)) - (Ixx*V*np.sin(alpha)*(tau*np.cos(alpha) + nu*np.sin(alpha)))/(k*n - 1)]])


        b =np.array([                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         [Fx + M*Omega_z*(Beta*V + sdot*(tau*np.cos(alpha) + nu*np.sin(alpha))*(rt - z)) - M*(sdot*(alphadot*nu*np.sin(alpha) + alphadot*tau*np.cos(alpha)) - (nu*np.cos(alpha) - tau*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2))*(rt - z + np.cos(Phi)*(h - rt)) - (Cd*V**2*rho)/2 - 2*M*sdot*zdot*(nu*np.cos(alpha) - tau*np.sin(alpha)) - M*g*sig*np.cos(alpha) + M*Omegadot_z*np.sin(Phi)*(h - rt) + 2*M*Phidot*(h - rt)*(Omega_z*np.cos(Phi) - sdot*np.sin(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))) - M*n*sdot**2*tau*(nu*np.cos(alpha) - tau*np.sin(alpha)) - M*sdot**2*np.sin(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(nu*np.cos(alpha) - tau*np.sin(alpha)) + M*Omega_z*sdot*np.cos(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [Fy + M*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2))*(rt - z + np.cos(Phi)*(h - rt)) - M*Omega_z*(V - sdot*(nu*np.cos(alpha) - tau*np.sin(alpha))*(rt - z)) + 2*M*sdot*zdot*(tau*np.cos(alpha) + nu*np.sin(alpha)) + M*np.sin(Phi)*(sdot**2*(tau*np.cos(alpha) + nu*np.sin(alpha))**2 + Omega_z**2)*(h - rt) + M*g*sig*np.sin(alpha) + (Cl*V**2*rho*np.sin(Phi))/2 + M*Phidot**2*np.sin(Phi)*(h - rt) + M*n*sdot**2*tau*(tau*np.cos(alpha) + nu*np.sin(alpha)) + M*Omega_z*sdot*np.cos(Phi)*(h - rt)*(nu*np.cos(alpha) - tau*np.sin(alpha)) + 2*M*Phidot*sdot*np.sin(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    [M*g - N - M*(n*tau*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2) + ndot*sdot*tau) + M*V*sdot*(nu*np.cos(alpha) - tau*np.sin(alpha)) - M*sdot**2*(tau*np.cos(alpha) + nu*np.sin(alpha))**2*(rt - z) - M*sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2*(rt - z) + M*np.sin(Phi)*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2))*(h - rt) - (Cl*V**2*rho*np.cos(Phi))/2 - M*Phidot**2*np.cos(Phi)*(h - rt) - M*np.cos(Phi)*(h - rt)*(sdot**2*(tau*np.cos(alpha) + nu*np.sin(alpha))**2 + sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2) + Beta*M*V*sdot*(tau*np.cos(alpha) + nu*np.sin(alpha)) - 2*M*Phidot*sdot*np.cos(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha)) - M*Omega_z*sdot*np.sin(Phi)*(h - rt)*(nu*np.cos(alpha) - tau*np.sin(alpha))],
        [Ixx*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2)) - (Iyy - Izz)*(np.cos(Phi)*np.sin(Phi)*(sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2 - Omega_z**2) - Omega_z*sdot*(nu*np.cos(alpha) - tau*np.sin(alpha))*(2*np.cos(Phi**2) - 1)) - M*((np.cos(Phi)*(- n*tau*(tau*np.cos(alpha) + nu*np.sin(alpha))*sdot**2 + 2*zdot*(tau*np.cos(alpha) + nu*np.sin(alpha))*sdot + Omega_z*V - g*sig*np.sin(alpha)) - np.sin(Phi)*(g - n*tau*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2) - ndot*sdot*tau + V*sdot*(nu*np.cos(alpha) - tau*np.sin(alpha)) + Beta*V*sdot*(tau*np.cos(alpha) + nu*np.sin(alpha))))*(h - rt) - (rt - z)*(n*tau*(tau*np.cos(alpha) + nu*np.sin(alpha))*sdot**2 + 2*zdot*(tau*np.cos(alpha) + nu*np.sin(alpha))*sdot - Omega_z*V + g*sig*np.sin(alpha)) + (sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2))*(- z**2 + 2*rt*z) - (h - rt)**2*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2) - np.cos(Phi)*np.sin(Phi)*(sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2 - Omega_z**2) + 2*sdot**2*np.cos(Phi)**2*(tau*np.cos(alpha) + nu*np.sin(alpha))*(nu*np.cos(alpha) - tau*np.sin(alpha))) - (h - rt)*(rt - z)*(2*np.cos(Phi)*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2)) + np.sin(Phi)*(Omega_z**2 - sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2 + Phidot**2 + 2*Phidot*sdot*(tau*np.cos(alpha) + nu*np.sin(alpha))) + 2*Omega_z*sdot*np.cos(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))) + Omega_z*sdot*(nu*np.cos(alpha) - tau*np.sin(alpha))*(h - z)*(h - 2*rt + z)) - Igy*omega_w*(Omega_z*np.cos(Phi) - sdot*np.sin(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))) + (Cl*V**2*rho*np.sin(Phi)*(rt - z))/2]])


        x = np.array([[Vdot],[Phiddot],[zddot],[Betadot]])
        residual_vector = (A[:,:,None]*x[:,:,None]).sum(1)-b

        residuals['Vdot'] = residual_vector[0,:,:]
        residuals['Phiddot'] = residual_vector[1,:,:]
        residuals['zddot'] = residual_vector[2,:,:]
        residuals['Betadot'] = residual_vector[3,:,:]


    def solve_nonlinear(self, inputs, outputs):
        nn = self.options['num_nodes']
        M = inputs['M']*np.ones(nn)
        rt = inputs['rt']*np.ones(nn)
        h = inputs['h']*np.ones(nn)
        rho = inputs['rho']*np.ones(nn)
        Cd = inputs['Cd']*np.ones(nn)
        g = inputs['g']*np.ones(nn)
        Cl = inputs['Cl']*np.ones(nn)
        r = inputs['r']*np.ones(nn)
        Ixx = inputs['Ixx']*np.ones(nn)
        Iyy = inputs['Iyy']*np.ones(nn)
        Izz = inputs['Izz']*np.ones(nn)
        Igy = inputs['Igy']*np.ones(nn)

        Phi = inputs['Phi']
        Phidot = inputs['Phidot']
        Beta = inputs['Beta']
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
        omega_w = inputs['omega_w']
        Fy = inputs['Fy']
        ndot = inputs['ndot']
        N = inputs['N']

        A =np.array([[                                                                                                                                                                                                                                                                                                                                                                                 M + (M*(nu*np.cos(alpha) - tau*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha))*(rt - z + np.cos(Phi)*(h - rt)))/(k*n - 1),                                                 np.zeros(nn),                   np.zeros(nn),                                                                                                                                                                                                                                                                                                        (M*V*np.sin(alpha)*(nu*np.cos(alpha) - tau*np.sin(alpha))*(rt - z + np.cos(Phi)*(h - rt)))/(k*n - 1)],
        [                                                                                                                                                                                                                                                                                                                                                                          - Beta*M - (M*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha))*(rt - z + np.cos(Phi)*(h - rt)))/(k*n - 1),                               M*np.cos(Phi)*(h - rt),                   np.zeros(nn),                                                                                                                                                                                                                                                                                                - M*V - (M*V*np.sin(alpha)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(rt - z + np.cos(Phi)*(h - rt)))/(k*n - 1)],
        [                                                                                                                                                                                                                                                                                                                                         - (M*n*tau*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1) - (M*np.sin(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1),                               M*np.sin(Phi)*(h - rt),                   M,                                                                                                                                                                                                                                                                              - (M*V*n*tau*np.sin(alpha))/(k*n - 1) - (M*V*np.sin(Phi)*np.sin(alpha)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha)))/(k*n - 1)],
        [- M*(Beta*(rt - z) - (Beta*np.cos(Phi) - (n*tau*np.sin(Phi)*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1))*(h - rt) - ((tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha))*(- z**2 + 2*rt*z))/(k*n - 1) + ((h - rt)**2*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1) + (2*np.cos(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha))*(rt - z))/(k*n - 1)) - (Ixx*(tau*np.cos(alpha) + nu*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1), Ixx + M*((h - rt)**2 + np.cos(Phi)*(h - rt)*(rt - z)), M*np.sin(Phi)*(h - rt), - M*(V*(rt - z) - (h - rt)*(V*np.cos(Phi) - (V*n*tau*np.sin(Phi)*np.sin(alpha))/(k*n - 1)) - (V*np.sin(alpha)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(- z**2 + 2*rt*z))/(k*n - 1) + (V*np.sin(alpha)*(h - rt)**2*(tau*np.cos(alpha) + nu*np.sin(alpha)))/(k*n - 1) + (2*V*np.cos(Phi)*np.sin(alpha)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(rt - z))/(k*n - 1)) - (Ixx*V*np.sin(alpha)*(tau*np.cos(alpha) + nu*np.sin(alpha)))/(k*n - 1)]])


        b =np.array([                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         [Fx + M*Omega_z*(Beta*V + sdot*(tau*np.cos(alpha) + nu*np.sin(alpha))*(rt - z)) - M*(sdot*(alphadot*nu*np.sin(alpha) + alphadot*tau*np.cos(alpha)) - (nu*np.cos(alpha) - tau*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2))*(rt - z + np.cos(Phi)*(h - rt)) - (Cd*V**2*rho)/2 - 2*M*sdot*zdot*(nu*np.cos(alpha) - tau*np.sin(alpha)) - M*g*sig*np.cos(alpha) + M*Omegadot_z*np.sin(Phi)*(h - rt) + 2*M*Phidot*(h - rt)*(Omega_z*np.cos(Phi) - sdot*np.sin(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))) - M*n*sdot**2*tau*(nu*np.cos(alpha) - tau*np.sin(alpha)) - M*sdot**2*np.sin(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))*(nu*np.cos(alpha) - tau*np.sin(alpha)) + M*Omega_z*sdot*np.cos(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [Fy + M*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2))*(rt - z + np.cos(Phi)*(h - rt)) - M*Omega_z*(V - sdot*(nu*np.cos(alpha) - tau*np.sin(alpha))*(rt - z)) + 2*M*sdot*zdot*(tau*np.cos(alpha) + nu*np.sin(alpha)) + M*np.sin(Phi)*(sdot**2*(tau*np.cos(alpha) + nu*np.sin(alpha))**2 + Omega_z**2)*(h - rt) + M*g*sig*np.sin(alpha) + (Cl*V**2*rho*np.sin(Phi))/2 + M*Phidot**2*np.sin(Phi)*(h - rt) + M*n*sdot**2*tau*(tau*np.cos(alpha) + nu*np.sin(alpha)) + M*Omega_z*sdot*np.cos(Phi)*(h - rt)*(nu*np.cos(alpha) - tau*np.sin(alpha)) + 2*M*Phidot*sdot*np.sin(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha))],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    [M*g - N - M*(n*tau*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2) + ndot*sdot*tau) + M*V*sdot*(nu*np.cos(alpha) - tau*np.sin(alpha)) - M*sdot**2*(tau*np.cos(alpha) + nu*np.sin(alpha))**2*(rt - z) - M*sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2*(rt - z) + M*np.sin(Phi)*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2))*(h - rt) - (Cl*V**2*rho*np.cos(Phi))/2 - M*Phidot**2*np.cos(Phi)*(h - rt) - M*np.cos(Phi)*(h - rt)*(sdot**2*(tau*np.cos(alpha) + nu*np.sin(alpha))**2 + sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2) + Beta*M*V*sdot*(tau*np.cos(alpha) + nu*np.sin(alpha)) - 2*M*Phidot*sdot*np.cos(Phi)*(h - rt)*(tau*np.cos(alpha) + nu*np.sin(alpha)) - M*Omega_z*sdot*np.sin(Phi)*(h - rt)*(nu*np.cos(alpha) - tau*np.sin(alpha))],
        [Ixx*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2)) - (Iyy - Izz)*(np.cos(Phi)*np.sin(Phi)*(sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2 - Omega_z**2) - Omega_z*sdot*(nu*np.cos(alpha) - tau*np.sin(alpha))*(2*np.cos(Phi**2) - 1)) - M*((np.cos(Phi)*(- n*tau*(tau*np.cos(alpha) + nu*np.sin(alpha))*sdot**2 + 2*zdot*(tau*np.cos(alpha) + nu*np.sin(alpha))*sdot + Omega_z*V - g*sig*np.sin(alpha)) - np.sin(Phi)*(g - n*tau*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2) - ndot*sdot*tau + V*sdot*(nu*np.cos(alpha) - tau*np.sin(alpha)) + Beta*V*sdot*(tau*np.cos(alpha) + nu*np.sin(alpha))))*(h - rt) - (rt - z)*(n*tau*(tau*np.cos(alpha) + nu*np.sin(alpha))*sdot**2 + 2*zdot*(tau*np.cos(alpha) + nu*np.sin(alpha))*sdot - Omega_z*V + g*sig*np.sin(alpha)) + (sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2))*(- z**2 + 2*rt*z) - (h - rt)**2*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2) - np.cos(Phi)*np.sin(Phi)*(sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2 - Omega_z**2) + 2*sdot**2*np.cos(Phi)**2*(tau*np.cos(alpha) + nu*np.sin(alpha))*(nu*np.cos(alpha) - tau*np.sin(alpha))) - (h - rt)*(rt - z)*(2*np.cos(Phi)*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) - (tau*np.cos(alpha) + nu*np.sin(alpha))*((V*(alphadot*np.sin(alpha) - Beta*alphadot*np.cos(alpha)))/(k*n - 1) + (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2)) + np.sin(Phi)*(Omega_z**2 - sdot**2*(nu*np.cos(alpha) - tau*np.sin(alpha))**2 + Phidot**2 + 2*Phidot*sdot*(tau*np.cos(alpha) + nu*np.sin(alpha))) + 2*Omega_z*sdot*np.cos(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))) + Omega_z*sdot*(nu*np.cos(alpha) - tau*np.sin(alpha))*(h - z)*(h - 2*rt + z)) - Igy*omega_w*(Omega_z*np.cos(Phi) - sdot*np.sin(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))) + (Cl*V**2*rho*np.sin(Phi)*(rt - z))/2]])

        for node in range(nn):
            x,exitCode = gmres(A[:,:,node],b[:,:,node],x0=self.out_guess)
            
            self.out_guess = x
            outputs['Vdot'][node] = x[0]
            outputs['Phiddot'][node] = x[1]
            outputs['zddot'][node] = x[2]
            outputs['Betadot'][node] = x[3]

