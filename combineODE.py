import openmdao.api as om
import numpy as np
from brakeThrottleODE import brakeThrottle
from powerTrainODE import powerTrain
from timeSpaceODE import timeSpace
from trackingODE import tracking
from tyreConstraintODE import tyreConstraint
from tyreODE import  tyre
from curvature import curvature
from timeAdder import TimeAdder
from spinODE import spin
from linearSystem import linearSystem


class combine(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='brakeThrottle',
                            subsys=brakeThrottle(num_nodes=nn),promotes_inputs=['tau_t','tau_b'],promotes_outputs=['tau_w'])     

        self.add_subsystem(name='powerTrain',
                           subsys=powerTrain(num_nodes=nn),promotes_inputs=['e','T','omega_w','tau_t','V'],promotes_outputs=['edot','Tdot','im'])     

        self.add_subsystem(name='curvature',
                    subsys=curvature(num_nodes=nn),promotes_outputs=['k'])   
        
        #self.connect('curvature.k','spin.k')
                
        #self.connect('curvature.k',['spin.k','implicitOutputs.k','tracking.k'])
        # self.connect('curvature.nu',['spin.nu','implicitOutputs.nu']) 
        # self.connect('curvature.tau',['spin.tau','implicitOutputs.tau'])
        # self.connect('curvature.sig',[           'implicitOutputs.sig'])  

        self.add_subsystem(name='tracking',
                           subsys=tracking(num_nodes=nn),promotes_inputs=['V','Beta','alpha','n','k','Omega_z'],promotes_outputs=['sdot','ndot','alphadot'])

        self.add_subsystem(name='tyre',
                           subsys=tyre(num_nodes=nn),promotes_inputs=['omega_w','Phi','Beta','V','z','zdot'],promotes_outputs=['N','Fy','Fx'])   

        self.add_subsystem(name='tyreConstraint',
                           subsys=tyreConstraint(num_nodes=nn),promotes_inputs=['N','Fx','Fy'],promotes_outputs=['TC'])    

        #Solve equations 13 and 14
        self.add_subsystem(name='linearSystem',
                           subsys=linearSystem(num_nodes=nn),promotes_inputs=['Phi','Phidot','Beta','alpha','alphadot','nu','tau','sig','V','n','k','z','zdot','sdot','Omega_z','Omegadot_z','Fx','Fy','omega_w','ndot','N'],
                           promotes_outputs=['Vdot','Betadot','Phiddot','zddot'])    

        self.add_subsystem(name='spin',subsys=spin(num_nodes=nn),promotes_inputs=['Phi','Phidot','Beta','Betadot','alpha','alphadot','nu','k','tau','V','Vdot','n','z','sdot','Omega_z','Omegadot_z','Fx','omega_w','tau_w','N','Fy'],
        promotes_outputs=['omegadot_w'])

        self.add_subsystem(name='timeSpace',
                    subsys=timeSpace(num_nodes=nn),promotes_inputs=['sdot','Phidot','Phiddot','ndot','alphadot','Vdot','Betadot','zdot','zddot','omegadot_w','Omegadot_z','Tdot','edot'],
                    promotes_outputs=['dPhi_ds','dPhidot_ds','dn_ds','dalpha_ds','dV_ds','dBeta_ds','dz_ds','dzdot_ds','domega_w_ds','dOmega_z_ds','dT_ds','de_ds'])

        self.add_subsystem(name='timeAdder',subsys=TimeAdder(num_nodes=nn),promotes_inputs=['sdot'],promotes_outputs=['dt_ds'])   
