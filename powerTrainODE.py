import openmdao.api as om
import numpy as np

#Equations 18, 19
class powerTrain(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('Rb', val=30, desc='battery esistance', units='ohm')#guess
        self.add_input('Vb1', val=420, desc='battery full voltage', units='V')#semi guess
        self.add_input('Vb0', val=360, desc='battery empty voltage', units='V')#semi guess
        self.add_input('ei', val=50000, desc='initial battery charge', units='J')#semi guess

        self.add_input('Text', val=22, desc='external temperature', units='C')
        self.add_input('sig_R', val=0.1, desc='resistance temperature lapse', units=None)#semi guess
        self.add_input('sig_kV', val=-0.01, desc='bemf constant temperature lapse', units=None)
        self.add_input('sig_kt', val=-0.01, desc='torque constant temperature lapse', units=None)
        
        self.add_input('R0', val=30, desc='base resistance', units=None)#semi guess
        self.add_input('kV0', val=0.2, desc='base bemf constant', units=None)
        self.add_input('kt0', val=0.17, desc='base torque constant', units=None)
        self.add_input('T0', val=22, desc='reference temperature', units='C')
        self.add_input('im0', val=2, desc='engine current offset', units=None)

        self.add_input('q0', val=50, desc='base conduction coefficient', units=None)
        self.add_input('q1', val=150, desc='peak conduction coefficient', units=None)
        self.add_input('V1', val=41, desc='conduction coefficient speed', units=None)
        #add motor efficiency seperately 0.85
        self.add_input('C', val=500, desc='motor thermal capacity', units=None)#GUESS

        self.add_input('eta', val=0.8, desc='power train energy efficiency', units=None)
        self.add_input('gamma', val=3, desc='gear ratio', units=None)

        #states
        self.add_input('e', val=50000*np.ones(nn), desc='battery energy', units='J')
        self.add_input('T', val=22*np.ones(nn), desc='system temperature', units='C')
        self.add_input('omega_w', val=np.zeros(nn), desc='wheel spin', units='rad/s')
        self.add_input('tau_t', val=np.zeros(nn), desc='driving torque', units='N*m')
        self.add_input('V', val=np.zeros(nn), desc='forward velocity', units='m/s')


        #outputs
        self.add_output('Tdot', val=np.zeros(nn), desc='temperature rate', units='T/s')
        self.add_output('edot', val=np.zeros(nn), desc='battery discharge rate', units='W')
        self.add_output('im', val=np.zeros(nn), desc='motor current', units='A')

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=True)
        

    def compute(self, inputs, outputs):
        Rb = inputs['Rb']
        Vb1 = inputs['Vb1']
        Vb0 = inputs['Vb0']
        ei = inputs['ei']
        Text = inputs['Text']
        sig_R = inputs['sig_R']
        sig_kV = inputs['sig_kV']
        sig_kt = inputs['sig_kt']
        R0 = inputs['R0']
        kV0 = inputs['kV0']
        kt0 = inputs['kt0']
        T0 = inputs['T0']
        im0 = inputs['im0']
        q0 = inputs['q0']
        q1 = inputs['q1']
        V1 = inputs['V1']
        eta = inputs['eta']
        omega_w = inputs['omega_w']
        tau_t = inputs['tau_t']
        gamma = inputs['gamma']
        e = inputs['e']
        T = inputs['T']
        V = inputs['V']
        C = inputs['C']

        Vbn = Vb0+(Vb1-Vb0)*(e/ei)

        Rm = R0 + sig_R*(T-T0)
        kV = kV0 - sig_kV*(T-T0)
        kt = kt0 - sig_kt*(T-T0)

        omega_e = omega_w*gamma
        im = im0+tau_t/(eta*gamma)/kt
        tau_e = kt*(im-im0)

        q = q0+(q1-q0)*V/V1

        Vmm = omega_e*kV + (tau_e/kt)*Rm

        Qin = Vmm*im-omega_w*tau_e
        Qout = q*(T-Text)

        Vbn = Vb0+(Vb1-Vb0)*(e/ei)
        ib = (Vbn/(2*Rb))-((eta*(Vbn**2*eta-4*(tau_e/kt)**2*Rb*Rm-4*(tau_e/kt)*Rb*omega_e*kV))**0.5)/(2*Rb*eta)

        outputs['edot'] = -Vbn*ib
        outputs['Tdot'] = (Qin-Qout)/C
        outputs['im'] = im
