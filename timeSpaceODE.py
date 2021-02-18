import openmdao.api as om
import numpy as np



class timeSpace(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #states
        self.add_input('sdot', val=np.ones(nn), desc='road longitudinal velocity', units='m/s')
        self.add_input('Phidot', val=np.zeros(nn), desc='roll rate', units='rad/s')
        self.add_input('Phiddot', val=np.zeros(nn), desc='roll rate2', units='rad/s**2')
        self.add_input('ndot', val=np.zeros(nn), desc='road lateral velocity', units='m/s')
        self.add_input('alphadot', val=np.zeros(nn), desc='relative heading rate', units='rad/s')
        self.add_input('Vdot', val=np.zeros(nn), desc='forward acceleration', units='m/s**2')
        self.add_input('Betadot', val=np.zeros(nn), desc='drift rate', units='rad/s')
        self.add_input('zdot', val=np.zeros(nn), desc='vertical velocity', units='m/s')
        self.add_input('zddot', val=np.zeros(nn), desc='vertical acceleration', units='m/s**2')
        self.add_input('omegadot_w', val=np.zeros(nn), desc='wheel spin2', units='rad/s**2')
        self.add_input('Omegadot_z', val=np.zeros(nn), desc='yaw rate2', units='rad/s**2')
        self.add_input('Tdot', val=np.zeros(nn), desc='temperature rate', units='T/s')
        self.add_input('edot', val=np.zeros(nn), desc='battery discharge rate', units='W')

        #outputs
        self.add_output('dPhi_ds', val=np.zeros(nn), units = 'rad/m')
        self.add_output('dPhidot_ds', val=np.zeros(nn), units = 'rad/s/m')
        self.add_output('dn_ds', val=np.zeros(nn), units = 'm/m')
        self.add_output('dalpha_ds', val=np.zeros(nn), units = 'rad/m')
        self.add_output('dV_ds', val=np.zeros(nn), units = 'm/s/m')
        self.add_output('dBeta_ds', val=np.zeros(nn), units = 'rad/m')
        self.add_output('dz_ds', val=np.zeros(nn), units = 'm/m')
        self.add_output('dzdot_ds', val=np.zeros(nn), units = 'm/s/m')
        self.add_output('domega_w_ds', val=np.zeros(nn), units = 'rad/s/m')
        self.add_output('dOmega_z_ds', val=np.zeros(nn), units = 'rad/s/m')
        self.add_output('dT_ds', val=np.zeros(nn), units = 'C/m')
        self.add_output('de_ds', val=np.zeros(nn), units = 'J/m')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='dPhi_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dPhi_ds', wrt='Phidot', rows=arange, cols=arange)
        self.declare_partials(of='dPhidot_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dPhidot_ds', wrt='Phiddot', rows=arange, cols=arange)
        self.declare_partials(of='dn_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dn_ds', wrt='ndot', rows=arange, cols=arange)
        self.declare_partials(of='dalpha_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dalpha_ds', wrt='alphadot', rows=arange, cols=arange)
        self.declare_partials(of='dV_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dV_ds', wrt='Vdot', rows=arange, cols=arange)
        self.declare_partials(of='dBeta_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dBeta_ds', wrt='Betadot', rows=arange, cols=arange)
        self.declare_partials(of='dz_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dz_ds', wrt='zdot', rows=arange, cols=arange)
        self.declare_partials(of='dzdot_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dzdot_ds', wrt='zddot', rows=arange, cols=arange)
        self.declare_partials(of='domega_w_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='domega_w_ds', wrt='omegadot_w', rows=arange, cols=arange)
        self.declare_partials(of='dOmega_z_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dOmega_z_ds', wrt='Omegadot_z', rows=arange, cols=arange)
        self.declare_partials(of='dT_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='dT_ds', wrt='Tdot', rows=arange, cols=arange)
        self.declare_partials(of='de_ds', wrt='sdot', rows=arange, cols=arange)
        self.declare_partials(of='de_ds', wrt='edot', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        sdot = inputs['sdot']
        Phidot = inputs['Phidot']
        Phiddot = inputs['Phiddot']
        ndot = inputs['ndot']
        alphadot = inputs['alphadot']
        Vdot = inputs['Vdot']
        Betadot = inputs['Betadot']
        zdot = inputs['zdot']
        zddot = inputs['zddot']
        omegadot_w = inputs['omegadot_w']
        Omegadot_z = inputs['Omegadot_z']
        Tdot = inputs['Tdot']
        edot = inputs['edot']

        outputs['dPhi_ds'] = Phidot/sdot 
        outputs['dPhidot_ds'] = Phiddot/sdot
        outputs['dn_ds'] = ndot/sdot
        outputs['dalpha_ds'] = alphadot/sdot
        outputs['dV_ds'] = Vdot/sdot
        outputs['dBeta_ds'] = Betadot/sdot
        outputs['dz_ds'] = zdot/sdot
        outputs['dzdot_ds'] = zddot/sdot
        outputs['domega_w_ds'] = omegadot_w/sdot
        outputs['dOmega_z_ds'] = Omegadot_z/sdot
        outputs['dT_ds'] = Tdot/sdot
        outputs['de_ds'] = edot/sdot

    def compute_partials(self, inputs, jacobian):
        sdot = inputs['sdot']
        Phidot = inputs['Phidot']
        Phiddot = inputs['Phiddot']
        ndot = inputs['ndot']
        alphadot = inputs['alphadot']
        Vdot = inputs['Vdot']
        Betadot = inputs['Betadot']
        zdot = inputs['zdot']
        zddot = inputs['zddot']
        omegadot_w = inputs['omegadot_w']
        Omegadot_z = inputs['Omegadot_z']
        Tdot = inputs['Tdot']
        edot = inputs['edot']

        jacobian['dPhi_ds','sdot']=-Phidot/sdot**2
        jacobian['dPhi_ds','Phidot']=1/sdot
        jacobian['dPhidot_ds','sdot']=-Phiddot/sdot**2
        jacobian['dPhidot_ds','Phiddot']=1/sdot
        jacobian['dn_ds','sdot']=-ndot/sdot**2
        jacobian['dn_ds','ndot']=1/sdot
        jacobian['dalpha_ds','sdot']=-alphadot/sdot**2
        jacobian['dalpha_ds','alphadot']=1/sdot
        jacobian['dV_ds','sdot']=-Vdot/sdot**2
        jacobian['dV_ds','Vdot']=1/sdot
        jacobian['dBeta_ds','sdot']=-Betadot/sdot**2
        jacobian['dBeta_ds','Betadot']=1/sdot
        jacobian['dz_ds','sdot']=-zdot/sdot**2
        jacobian['dz_ds','zdot']=1/sdot
        jacobian['dzdot_ds','sdot']=-zddot/sdot**2
        jacobian['dzdot_ds','zddot']=1/sdot
        jacobian['domega_w_ds','sdot']=-omegadot_w/sdot**2
        jacobian['domega_w_ds','omegadot_w']=1/sdot
        jacobian['dOmega_z_ds','sdot']=-Omegadot_z/sdot**2
        jacobian['dOmega_z_ds','Omegadot_z']=1/sdot
        jacobian['dT_ds','sdot']=-Tdot/sdot**2
        jacobian['dT_ds','Tdot']=1/sdot
        jacobian['de_ds','sdot']=-edot/sdot**2
        jacobian['de_ds','edot']=1/sdot