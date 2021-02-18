import openmdao.api as om
import numpy as np

class powerTrain(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('Rb', val=0.5, desc='battery esistance', units='ohm')#guess
        self.add_input('Vb1', val=420, desc='battery full voltage', units='V')
        self.add_input('Vb0', val=360, desc='battery empty voltage', units='V')
        self.add_input('ei', val=5e8, desc='initial battery charge', units='J')#guess

        self.add_input('Text', val=22, desc='external temperature', units='C')
        self.add_input('sig_R', val=0.00, desc='resistance temperature lapse', units=None)#guess
        self.add_input('sig_kV', val=-0.00, desc='bemf constant temperature lapse', units=None)
        self.add_input('sig_kt', val=-0.00, desc='torque constant temperature lapse', units=None)
        
        self.add_input('R0', val=1.9, desc='base resistance', units=None)#guess
        self.add_input('kV0', val=0.033333, desc='base bemf constant', units=None)
        self.add_input('kt0', val=0.724, desc='base torque constant', units=None)
        self.add_input('T0', val=22, desc='reference temperature', units='C')
        self.add_input('im0', val=2, desc='engine current offset', units=None)

        self.add_input('q0', val=500, desc='base conduction coefficient', units=None)
        self.add_input('q1', val=1500, desc='peak conduction coefficient', units=None)
        self.add_input('V1', val=41, desc='conduction coefficient calibration speed', units=None)
        #add motor efficiency seperately 0.85
        self.add_input('C', val=12000, desc='motor thermal capacity', units=None)#guess

        self.add_input('eta', val=0.8, desc='power train energy efficiency', units=None)
        self.add_input('gamma', val=3, desc='gear ratio', units=None)

        #states
        self.add_input('e', val=5e7*np.ones(nn), desc='battery energy', units='J')
        self.add_input('T', val=22*np.ones(nn), desc='system temperature', units='C')
        self.add_input('omega_w', val=np.zeros(nn), desc='wheel spin', units='rad/s')
        self.add_input('tau_t', val=np.zeros(nn), desc='driving torque', units='N*m')
        self.add_input('V', val=np.zeros(nn), desc='forward velocity', units='m/s')

        #outputs
        self.add_output('Tdot', val=np.zeros(nn), desc='temperature rate', units='T/s')
        self.add_output('edot', val=np.zeros(nn), desc='battery discharge rate', units='W')
        self.add_output('im', val=np.zeros(nn), desc='motor current', units='A')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='edot', wrt='e', rows=arange, cols=arange)
        self.declare_partials(of='edot', wrt='T', rows=arange, cols=arange)
        self.declare_partials(of='edot', wrt='tau_t', rows=arange, cols=arange)
        self.declare_partials(of='edot', wrt='omega_w', rows=arange, cols=arange)
        self.declare_partials(of='Tdot', wrt='T', rows=arange, cols=arange)
        self.declare_partials(of='Tdot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='Tdot', wrt='tau_t', rows=arange, cols=arange)
        self.declare_partials(of='Tdot', wrt='omega_w', rows=arange, cols=arange)
        self.declare_partials(of='im', wrt='T', rows=arange, cols=arange)
        self.declare_partials(of='im', wrt='tau_t', rows=arange, cols=arange)


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

        #Battery voltage decreases from 420 to 360 between e=ei and e=0
        Vbn = Vb0+(Vb1-Vb0)*(e/ei)

        #Temperature dependent parameters
        Rm = R0 + sig_R*(T-T0)
        kV = kV0 - sig_kV*(T-T0)
        kt = kt0 - sig_kt*(T-T0)

        #Relation between motor spin, current, and torque
        omega_e = omega_w*gamma
        im = im0+tau_t/(eta*gamma)/kt
        tau_e = kt*(im-im0)

        #Thermal conduction coefficient (speed dependent)
        q = q0+(q1-q0)*V/V1

        #Motor controller voltage
        Vmm = omega_e*kV + (tau_e/kt)*Rm

        #Motor heat balance
        Qin = Vmm*im-omega_w*tau_e
        Qout = q*(T-Text)

        #Battery current (equation 32)
        ib = (Vbn/(2*Rb))-((eta*(Vbn**2*eta-4*(tau_e/kt)**2*Rb*Rm-4*(tau_e/kt)*Rb*omega_e*kV))**0.5)/(2*Rb*eta)

        outputs['edot'] = -Vbn*ib
        outputs['Tdot'] = (Qin-Qout)/C
        outputs['im'] = im

    def compute_partials(self, inputs, jacobian):
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

        jacobian['edot','e']=((Vb0 - Vb1)/(2*Rb*ei) - (eta*(Vb0 - Vb1)*(Vb0 - (e*(Vb0 - Vb1))/ei))/(2*Rb*ei*(-eta*((4*Rb*omega_w*tau_t*(kV0 - sig_kV*(T - T0)))/(eta*(kt0 - sig_kt*(T - T0))) - eta*(Vb0 - (e*(Vb0 - Vb1))/ei)**2 + (4*Rb*tau_t**2*(R0 + sig_R*(T - T0)))/(eta**2*gamma**2*(kt0 - sig_kt*(T - T0))**2)))**(1/2)))*(Vb0 - (e*(Vb0 - Vb1))/ei) + (((Vb0 - (e*(Vb0 - Vb1))/ei)/(2*Rb) - (-eta*((4*Rb*tau_t**2*(R0 + sig_R*(T - T0)))/(eta**2*gamma**2*(kt0 - sig_kt*(T - T0))**2) - eta*(Vb0 - (e*(Vb0 - Vb1))/ei)**2 + (4*Rb*omega_w*tau_t*(kV0 - sig_kV*(T - T0)))/(eta*(kt0 - sig_kt*(T - T0)))))**(1/2)/(2*Rb*eta))*(Vb0 - Vb1))/ei
        jacobian['edot','T']=-((Vb0 - (e*(Vb0 - Vb1))/ei)*((4*Rb*sig_R*tau_t**2)/(eta**2*gamma**2*(kt0 - sig_kt*(T - T0))**2) - (4*Rb*omega_w*sig_kV*tau_t)/(eta*(kt0 - sig_kt*(T - T0))) + (4*Rb*omega_w*sig_kt*tau_t*(kV0 - sig_kV*(T - T0)))/(eta*(kt0 - sig_kt*(T - T0))**2) + (8*Rb*sig_kt*tau_t**2*(R0 + sig_R*(T - T0)))/(eta**2*gamma**2*(kt0 - sig_kt*(T - T0))**3)))/(4*Rb*(-eta*((4*Rb*omega_w*tau_t*(kV0 - sig_kV*(T - T0)))/(eta*(kt0 - sig_kt*(T - T0))) - eta*(Vb0 - (e*(Vb0 - Vb1))/ei)**2 + (4*Rb*tau_t**2*(R0 + sig_R*(T - T0)))/(eta**2*gamma**2*(kt0 - sig_kt*(T - T0))**2)))**(1/2))
        jacobian['edot','tau_t']=-((Vb0 - (e*(Vb0 - Vb1))/ei)*((4*Rb*omega_w*(kV0 - sig_kV*(T - T0)))/(eta*(kt0 - sig_kt*(T - T0))) + (8*Rb*tau_t*(R0 + sig_R*(T - T0)))/(eta**2*gamma**2*(kt0 - sig_kt*(T - T0))**2)))/(4*Rb*(-eta*((4*Rb*omega_w*tau_t*(kV0 - sig_kV*(T - T0)))/(eta*(kt0 - sig_kt*(T - T0))) - eta*(Vb0 - (e*(Vb0 - Vb1))/ei)**2 + (4*Rb*tau_t**2*(R0 + sig_R*(T - T0)))/(eta**2*gamma**2*(kt0 - sig_kt*(T - T0))**2)))**(1/2))
        jacobian['edot','omega_w']=-(tau_t*(kV0 - sig_kV*(T - T0))*(Vb0 - (e*(Vb0 - Vb1))/ei))/(eta*(kt0 - sig_kt*(T - T0))*(-eta*((4*Rb*omega_w*tau_t*(kV0 - sig_kV*(T - T0)))/(eta*(kt0 - sig_kt*(T - T0))) - eta*(Vb0 - (e*(Vb0 - Vb1))/ei)**2 + (4*Rb*tau_t**2*(R0 + sig_R*(T - T0)))/(eta**2*gamma**2*(kt0 - sig_kt*(T - T0))**2)))**(1/2))
        jacobian['Tdot','T']=((im0 + tau_t/(eta*gamma*(kt0 - sig_kt*(T - T0))))*((sig_R*tau_t)/(eta*gamma*(kt0 - sig_kt*(T - T0))) - gamma*omega_w*sig_kV + (sig_kt*tau_t*(R0 + sig_R*(T - T0)))/(eta*gamma*(kt0 - sig_kt*(T - T0))**2)) - q0 + (V*(q0 - q1))/V1 + (sig_kt*tau_t*(gamma*omega_w*(kV0 - sig_kV*(T - T0)) + (tau_t*(R0 + sig_R*(T - T0)))/(eta*gamma*(kt0 - sig_kt*(T - T0)))))/(eta*gamma*(kt0 - sig_kt*(T - T0))**2))/C
        jacobian['Tdot','V']=((T - Text)*(q0 - q1))/(C*V1)
        jacobian['Tdot','tau_t']=((gamma*omega_w*(kV0 - sig_kV*(T - T0)) + (tau_t*(R0 + sig_R*(T - T0)))/(eta*gamma*(kt0 - sig_kt*(T - T0))))/(eta*gamma*(kt0 - sig_kt*(T - T0))) - omega_w/(eta*gamma) + ((im0 + tau_t/(eta*gamma*(kt0 - sig_kt*(T - T0))))*(R0 + sig_R*(T - T0)))/(eta*gamma*(kt0 - sig_kt*(T - T0))))/C
        jacobian['Tdot','omega_w']=(gamma*(kV0 - sig_kV*(T - T0))*(im0 + tau_t/(eta*gamma*(kt0 - sig_kt*(T - T0)))) - tau_t/(eta*gamma))/C
        jacobian['im','T']=(sig_kt*tau_t)/(eta*gamma*(kt0 - sig_kt*(T - T0))**2)
        jacobian['im','tau_t']=1/(eta*gamma*(kt0 - sig_kt*(T - T0)))
