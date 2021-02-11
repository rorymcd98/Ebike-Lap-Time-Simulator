import openmdao.api as om
import numpy as np
from implicitSubstep.xEquation import xEquation
from implicitSubstep.yEquation import yEquation
from implicitSubstep.zEquation import zEquation
from implicitSubstep.eulerEquation import eulerEquation

class implicitSolver(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='xEquation',
                            subsys=xEquation(num_nodes=nn), promotes=['*'])
        
        self.add_subsystem(name='yEquation',
                            subsys=yEquation(num_nodes=nn), promotes=['*'])

        self.add_subsystem(name='zEquation',
                            subsys=zEquation(num_nodes=nn), promotes=['*'])

        self.add_subsystem(name='eulerEquation',
                            subsys=eulerEquation(num_nodes=nn), promotes=['*'])

        self.nonlinear_solver = om.NewtonSolver()
        self.nonlinear_solver.options['atol'] = 1e-14
        self.nonlinear_solver.options['rtol'] = 1e-14
        self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.options['err_on_non_converge'] = True
        self.nonlinear_solver.options['max_sub_solves'] = 10
        self.nonlinear_solver.options['maxiter'] = 150
        self.nonlinear_solver.options['iprint'] = -1
        self.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        self.nonlinear_solver.linesearch.options['print_bound_enforce'] = True