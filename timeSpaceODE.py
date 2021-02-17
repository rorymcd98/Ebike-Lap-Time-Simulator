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

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12)

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