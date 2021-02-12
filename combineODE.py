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
from implicitSubstep.combinedEOM import combinedEOM


class combine(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='brakeThrottle',
                            subsys=brakeThrottle(num_nodes=nn),promotes_inputs = ['tau_t','tau_b'],promotes_outputs=['tau_w'])

        self.add_subsystem(name='powerTrain',
                           subsys=powerTrain(num_nodes=nn),promotes_inputs=['omega_w','tau_t','V'],promotes_outputs=['Tdot','edot','im'])

        self.add_subsystem(name='tracking',
                           subsys=tracking(num_nodes=nn),promotes_inputs=['V','Beta','alpha','n','k','Omega_z'],promotes_outputs=['sdot','ndot','alphadot'])

        self.add_subsystem(name='tyre',
                           subsys=tyre(num_nodes=nn),promotes_inputs=['omega_w','Phi','Beta','V','z','zdot'],promotes_outputs=['N','Fx','Fy','kw'])

        self.add_subsystem(name='tyreConstraint',
                           subsys=tyreConstraint(num_nodes=nn),promotes_inputs=['N','Fx','Fy'],promotes_outputs=['TC'])

        self.add_subsystem(name='curvature',
                           subsys=curvature(num_nodes=nn),promotes_outputs=['tau','sig','kappa'])

        #Solve equations 13 and 14
        self.add_subsystem(name='implicitOutputs',
                           subsys=combinedEOM(num_nodes=nn),promotes=['Vdot','Phiddot','zddot','Betadot'])

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', ImpWithInitial())

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.ScipyKrylov()

        prob.setup()
        prob.run_model()

        self.add_subsystem(name='spin',subsys=spin(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['omegadot_w'])

        self.add_subsystem(name='timeSpace',
                    subsys=timeSpace(num_nodes=nn),promotes_inputs=['sdot','Phidot','Phiddot','ndot','alphadot','Vdot','Betadot','zdot','zddot','omegadot_w','Omegadot_z','Tdot','edot'],
                    promotes_outputs=['dPhi_ds','dPhidot_ds','dn_ds','dalpha_ds','dV_ds','dBeta_ds','dz_ds','dzdot_ds','domega_w_ds','dOmega_z_ds','dT_ds','de_ds'])

        self.add_subsystem(name='timeAdder',subsys=TimeAdder(num_nodes=nn),promotes_inputs=['sdot'],promotes_outputs=['dt_ds'])
