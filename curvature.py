import openmdao.api as om
import numpy as np

#track curvature imports
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
from spline import getTrackPoints,getSpline

class curvature(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        track = tracks.ovaltrack #remember to change here and in problemSolver.py

        points = getTrackPoints(track)
        print(track.getTotalLength())
        finespline,gates,gatesd,curv,slope = getSpline(points)

        self.curv = curv
        self.trackLength = track.getTotalLength()

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('s', val=np.zeros(nn), desc='distance along track', units='m')

        #outputs
        self.add_output('k', val=np.zeros(nn), desc='road curvature x-y', units='1/m')
        self.add_output('sig', val=np.zeros(nn), desc='road slope', units='rad')
        self.add_output('tau', val=np.zeros(nn), desc='road curvature torsional', units='1/m')
        self.add_output('nu', val=np.zeros(nn), desc='road curvature x-z', units='1/m')

        #no partials needed

    def compute(self, inputs, outputs):
        s = inputs['s']

        num_curv_points = len(self.curv)

        k = np.zeros(len(s))
        sig = np.zeros(len(s))
        

        for i in range(len(s)):
            index = np.floor((s[i]/self.trackLength)*num_curv_points)
            index = np.minimum(index,num_curv_points-1)
            k[i] = self.curv[index.astype(int)]

        outputs['k'] = k
        outputs['sig'] = sig
        outputs['tau'] = sig
        outputs['nu'] = sig

    def compute_partials(self, inputs, jacobian):
        pass

        








