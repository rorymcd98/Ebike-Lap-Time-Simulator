import openmdao.api as om
import numpy as np

#Equations 18, 20
class tyre(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('Ks', val=2e5, desc='tyre radial stiffness', units=None)
        self.add_input('Kc', val=1e3, desc='tyre damping stiffness', units=None)
        self.add_input('Kk', val=12, desc='tyre longitudinal stiffness',units=None)#guess
        self.add_input('Kb', val=12, desc='tyre lateral stiffness', units=None)
        self.add_input('Kp', val=1, desc='tyre roll stiffness', units=None)

        self.add_input('r', val=0.3, desc='wheel radius', units='m')
        self.add_input('rt', val=0.1, desc='wheel toroid radius', units='m')
        self.add_input('h', val=0.6, desc='CoG height', units='m')

        #states
        self.add_input('omega_w', val=np.zeros(nn), desc='wheel spin', units='rad/s')
        self.add_input('Phi', val=np.zeros(nn), desc='roll', units='rad')
        self.add_input('Beta', val=np.zeros(nn), desc='drift angle', units='rad')
        self.add_input('V', val=np.zeros(nn), desc='forward velocity', units='m/s')
        self.add_input('z', val=np.zeros(nn), desc='vertical displacement', units='m')
        self.add_input('zdot', val=np.zeros(nn), desc='vertical velocity', units='m/s')

        #outputs
        self.add_output('N', val=np.zeros(nn), desc='tyre load', units='N')
        self.add_output('Fx', val=np.zeros(nn), desc='longitudinal tyre force', units='N')
        self.add_output('Fy', val=np.zeros(nn), desc='lateral tyre load', units='N')
 
        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='N', wrt='z', rows=arange, cols=arange)
        self.declare_partials(of='N', wrt='zdot', rows=arange, cols=arange)
        self.declare_partials(of='Fx', wrt='omega_w', rows=arange, cols=arange)
        self.declare_partials(of='Fx', wrt='Phi', rows=arange, cols=arange)
        self.declare_partials(of='Fx', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='Fx', wrt='z', rows=arange, cols=arange)
        self.declare_partials(of='Fx', wrt='zdot', rows=arange, cols=arange)
        self.declare_partials(of='Fy', wrt='Phi', rows=arange, cols=arange)
        self.declare_partials(of='Fy', wrt='Beta', rows=arange, cols=arange)
        self.declare_partials(of='Fy', wrt='z', rows=arange, cols=arange)
        self.declare_partials(of='Fy', wrt='zdot', rows=arange, cols=arange)
        

    def compute(self, inputs, outputs):
        Ks = inputs['Ks']
        Kc = inputs['Kc']
        Kk = inputs['Kk']
        Kb = inputs['Kb']
        Kp = inputs['Kp'] 
        r = inputs['r']
        rt = inputs['rt']
        h = inputs['h']
        omega_w = inputs['omega_w']
        Phi = inputs['Phi']
        Beta = inputs['Beta']
        V = inputs['V']
        z = inputs['z']
        zdot = inputs['zdot']

        kw = (omega_w*(r-rt+np.cos(Phi)*(rt-z))-V)/V
        N = Ks*z+Kc*zdot

        outputs['N'] = N
        outputs['Fy'] = N*(Kb*Beta+Kp*Phi)
        outputs['Fx'] = N*Kk*kw
    
    def compute_partials(self, inputs, jacobian):
        Ks = inputs['Ks']
        Kc = inputs['Kc']
        Kk = inputs['Kk']
        Kb = inputs['Kb']
        Kp = inputs['Kp'] 
        r = inputs['r']
        rt = inputs['rt']
        h = inputs['h']
        omega_w = inputs['omega_w']
        Phi = inputs['Phi']
        Beta = inputs['Beta']
        V = inputs['V']
        z = inputs['z']
        zdot = inputs['zdot']

        jacobian['N','z']=Ks
        jacobian['N','zdot']=Kc
        jacobian['Fx','omega_w']=(Kk*(Ks*z + Kc*zdot)*(r - rt + np.cos(Phi)*(rt - z)))/V
        jacobian['Fx','Phi']=-(Kk*omega_w*np.sin(Phi)*(Ks*z + Kc*zdot)*(rt - z))/V
        jacobian['Fx','V']=(Kk*(Ks*z + Kc*zdot)*(V - omega_w*(r - rt + np.cos(Phi)*(rt - z))))/V**2 - (Kk*(Ks*z + Kc*zdot))/V
        jacobian['Fx','z']=- (Kk*Ks*(V - omega_w*(r - rt + np.cos(Phi)*(rt - z))))/V - (Kk*omega_w*np.cos(Phi)*(Ks*z + Kc*zdot))/V
        jacobian['Fx','zdot']=-(Kc*Kk*(V - omega_w*(r - rt + np.cos(Phi)*(rt - z))))/V
        jacobian['Fy','Phi']=Kp*(Ks*z + Kc*zdot)
        jacobian['Fy','Beta']=Kb*(Ks*z + Kc*zdot)
        jacobian['Fy','z']=Ks*(Beta*Kb + Kp*Phi)
        jacobian['Fy','zdot']=Kc*(Beta*Kb + Kp*Phi)

