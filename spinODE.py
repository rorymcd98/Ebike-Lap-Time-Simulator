import numpy as np
import openmdao.api as om

#Equations 13a,b,c, 14, 15, 16, 18(3)
class spin(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('rt', val=0.1, desc='wheel toroid radius', units='m')
        self.add_input('r', val=0.3, desc='wheel radius', units='m')
        self.add_input('Iwy', val=0.7, desc='wheelspin moment of inertia', units='kg*m**2')
        self.add_input('mu_r', val=0.015, desc='tyre rolling resistance', units=None)

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
        self.add_input('omega_w', val=np.zeros(nn), desc='wheel spin', units='rad/s')
        self.add_input('tau_w', val=np.zeros(nn), desc='driving torque', units='N*m')
        self.add_input('N', val=np.zeros(nn), desc='tyre load', units='N')
        self.add_input('Fy', val=np.zeros(nn), desc='lateral tyre load', units='N')
        self.add_input('ndot', val=np.zeros(nn), desc='road lateral velocity', units='m/s')

        #outputs
        self.add_output('omegadot_w', val=np.zeros(nn), desc='wheel spin2', units='rad/s**2')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='omegadot_w', wrt='Phi', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='Phidot', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='Beta', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='Betadot', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='alphadot', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='nu', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='tau', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='n', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='k', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='z', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='Omega_z', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='Omegadot_z', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='Fx', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='tau_w', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='Vdot', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='N', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='Fy', rows=arange, cols=arange)
        self.declare_partials(of='omegadot_w', wrt='ndot', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        rt = inputs['rt']
        r = inputs['r']
        Iwy = inputs['Iwy']
        mu_r = inputs['mu_r']

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
        Vdot = inputs['Vdot']
        N = inputs['N']
        Fy = inputs['Fy']
        ndot = inputs['ndot']

        Omega_y = sdot*(nu*np.cos(alpha)-tau*np.sin(alpha))
        sddot = (k*V*ndot*(np.cos(alpha) + np.sin(alpha)*Beta))/(k*n - 1)**2 - (V*(np.sin(alpha)*Betadot - np.sin(alpha)*alphadot + np.cos(alpha)*Beta*alphadot))/(k*n - 1) - (Vdot*(np.cos(alpha) + np.sin(alpha)*Beta))/(k*n - 1)
        Omegadot_y = (nu*np.cos(alpha) - tau*np.sin(alpha))*sddot - sdot*(tau*np.cos(alpha)*alphadot + nu*np.sin(alpha)*alphadot)

        tau_r = (N*np.cos(Phi)+Fy*np.sin(Phi))*mu_r

        outputs['omegadot_w'] = (tau_w - tau_r - Fx*(r-rt*(1-np.cos(Phi))-z*np.cos(Phi)) \
                            + Iwy*(Omegadot_y*np.cos(Phi)+np.sin(Phi)*Omegadot_z+(Omega_z*np.cos(Phi)-Omega_y*np.sin(Phi))*Phidot))/Iwy

    def compute_partials(self, inputs, jacobian):
        rt = inputs['rt']
        r = inputs['r']
        Iwy = inputs['Iwy']
        mu_r = inputs['mu_r']

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
        Vdot = inputs['Vdot']
        N = inputs['N']
        Fy = inputs['Fy']
        ndot = inputs['ndot']

        jacobian['omegadot_w','Phi']=(Fx*(rt*np.sin(Phi) - z*np.sin(Phi)) + Iwy*(np.sin(Phi)*(sdot*(alphadot*nu*np.sin(alpha) + alphadot*tau*np.cos(alpha)) + (nu*np.cos(alpha) - tau*np.sin(alpha))*((Vdot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1) + (V*(Betadot*np.sin(alpha) - alphadot*np.sin(alpha) + Beta*alphadot*np.cos(alpha)))/(k*n - 1) - (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2)) + Omegadot_z*np.cos(Phi) - Phidot*(Omega_z*np.sin(Phi) + sdot*np.cos(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha)))) - mu_r*(Fy*np.cos(Phi) - N*np.sin(Phi)))/Iwy
        jacobian['omegadot_w','Phidot']=Omega_z*np.cos(Phi) - sdot*np.sin(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))
        jacobian['omegadot_w','Beta']=-np.cos(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))*((Vdot*np.sin(alpha))/(k*n - 1) + (V*alphadot*np.cos(alpha))/(k*n - 1) - (V*k*ndot*np.sin(alpha))/(k*n - 1)**2)
        jacobian['omegadot_w','Betadot']=-(V*np.cos(Phi)*np.sin(alpha)*(nu*np.cos(alpha) - tau*np.sin(alpha)))/(k*n - 1)
        jacobian['omegadot_w','alpha']=np.cos(Phi)*(sdot*(alphadot*tau*np.sin(alpha) - alphadot*nu*np.cos(alpha)) + (nu*np.cos(alpha) - tau*np.sin(alpha))*((Vdot*(np.sin(alpha) - Beta*np.cos(alpha)))/(k*n - 1) + (V*(alphadot*np.cos(alpha) - Betadot*np.cos(alpha) + Beta*alphadot*np.sin(alpha)))/(k*n - 1) - (V*k*ndot*(np.sin(alpha) - Beta*np.cos(alpha)))/(k*n - 1)**2) + (tau*np.cos(alpha) + nu*np.sin(alpha))*((Vdot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1) + (V*(Betadot*np.sin(alpha) - alphadot*np.sin(alpha) + Beta*alphadot*np.cos(alpha)))/(k*n - 1) - (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2)) + Phidot*sdot*np.sin(Phi)*(tau*np.cos(alpha) + nu*np.sin(alpha))
        jacobian['omegadot_w','alphadot']=-np.cos(Phi)*(sdot*(tau*np.cos(alpha) + nu*np.sin(alpha)) - (V*(nu*np.cos(alpha) - tau*np.sin(alpha))*(np.sin(alpha) - Beta*np.cos(alpha)))/(k*n - 1))
        jacobian['omegadot_w','nu']=- np.cos(Phi)*(np.cos(alpha)*((Vdot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1) + (V*(Betadot*np.sin(alpha) - alphadot*np.sin(alpha) + Beta*alphadot*np.cos(alpha)))/(k*n - 1) - (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2) + alphadot*sdot*np.sin(alpha)) - Phidot*sdot*np.sin(Phi)*np.cos(alpha)
        jacobian['omegadot_w','tau']=np.cos(Phi)*(np.sin(alpha)*((Vdot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1) + (V*(Betadot*np.sin(alpha) - alphadot*np.sin(alpha) + Beta*alphadot*np.cos(alpha)))/(k*n - 1) - (V*k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2) - alphadot*sdot*np.cos(alpha)) + Phidot*sdot*np.sin(Phi)*np.sin(alpha)
        jacobian['omegadot_w','V']=-np.cos(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))*((Betadot*np.sin(alpha) - alphadot*np.sin(alpha) + Beta*alphadot*np.cos(alpha))/(k*n - 1) - (k*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2)
        jacobian['omegadot_w','n']=np.cos(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))*((V*k*(Betadot*np.sin(alpha) - alphadot*np.sin(alpha) + Beta*alphadot*np.cos(alpha)))/(k*n - 1)**2 + (Vdot*k*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2 - (2*V*k**2*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**3)
        jacobian['omegadot_w','k']=np.cos(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))*((V*n*(Betadot*np.sin(alpha) - alphadot*np.sin(alpha) + Beta*alphadot*np.cos(alpha)))/(k*n - 1)**2 + (V*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2 + (Vdot*n*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2 - (2*V*k*n*ndot*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**3)
        jacobian['omegadot_w','z']=(Fx*np.cos(Phi))/Iwy
        jacobian['omegadot_w','sdot']=- np.cos(Phi)*(alphadot*nu*np.sin(alpha) + alphadot*tau*np.cos(alpha)) - Phidot*np.sin(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))
        jacobian['omegadot_w','Omega_z']=Phidot*np.cos(Phi)
        jacobian['omegadot_w','Omegadot_z']=np.sin(Phi)
        jacobian['omegadot_w','Fx']=-(r - z*np.cos(Phi) + rt*(np.cos(Phi) - 1))/Iwy
        jacobian['omegadot_w','tau_w']=1/Iwy
        jacobian['omegadot_w','Vdot']=-(np.cos(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)
        jacobian['omegadot_w','N']=-(mu_r*np.cos(Phi))/Iwy
        jacobian['omegadot_w','Fy']=-(mu_r*np.sin(Phi))/Iwy
        jacobian['omegadot_w','ndot']=(V*k*np.cos(Phi)*(nu*np.cos(alpha) - tau*np.sin(alpha))*(np.cos(alpha) + Beta*np.sin(alpha)))/(k*n - 1)**2