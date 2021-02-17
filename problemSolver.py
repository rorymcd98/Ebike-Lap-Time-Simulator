import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from combineODE import combine
import matplotlib as mpl
import os

#track curvature imports
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
from spline import getSpline,getTrackPoints,getGateNormals,reverseTransformGates,setGateDisplacements,transformGates
from linewidthhelper import *

print('Config: RWD single thrust')

track = tracks.ovaltrack #change track here and in curvature.py. Tracks are defined in tracks.py
plot = False #plot track and telemetry

points = getTrackPoints(track) #generate nodes along the centerline for curvature calculation (different than collocation nodes)
finespline,gates,gatesd,curv,slope = getSpline(points,s=0.0) #fit the centerline spline. by default 10000 points
s_final = track.getTotalLength()

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class=combine,
		     transcription=dm.GaussLobatto(num_segments=50, order=3,compressed=True))

traj.add_phase(name='phase0', phase=phase)

# Set the time options, in this problem we perform a change of variables. So 'time' is actually 's' (distance along the track centerline)
# This is done to fix the collocation nodes in space, which saves us the calculation of the rate of change of curvature.
# The state equations are written with respect to time, the variable change occurs in timeODE.py
phase.set_time_options(fix_initial=True,fix_duration=True,duration_val=s_final,targets=['curvature.s'],units='m',duration_ref=s_final,duration_ref0=10)

#Define states
phase.add_state('t', fix_initial=True, fix_final=False, units='s', lower = 0,rate_source='dt_ds',ref=100) #time
phase.add_state('Phi', fix_initial=False, fix_final=False, units='rad', rate_source='dPhi_ds',targets=['Phi'])
phase.add_state('Phidot', fix_initial=False, fix_final=False, units='rad/s', rate_source='dPhidot_ds',targets=['Phidot'])
phase.add_state('n', fix_initial=False, fix_final=False, units='m', upper = 4.0, lower = -4.0, rate_source='dn_ds',targets=['n'],ref=4.0) #normal distance to centerline. The bounds on n define the width of the track
phase.add_state('alpha', fix_initial=False, fix_final=False, units='rad', rate_source='dalpha_ds',targets=['alpha'],ref=0.15) #vehicle heading angle with respect to centerline
phase.add_state('V', fix_initial=False, fix_final=False, units='m/s', ref = 0.1, ref0=5,rate_source='dV_ds', targets=['V']) #velocity
phase.add_state('Beta', fix_initial=False, fix_final=False, units='rad', rate_source='dBeta_ds',targets=['Beta'])
phase.add_state('z', fix_initial=False, fix_final=False, units='m', rate_source='dz_ds',targets=['z'])
phase.add_state('zdot', fix_initial=False, fix_final=False, units='m/s', rate_source='dzdot_ds',targets=['zdot'])
phase.add_state('omega_w', fix_initial=False, fix_final=False, units='rad/s', rate_source='domega_w_ds',targets=['omega_w'])
phase.add_state('Omega_z', fix_initial=False, fix_final=False, units='rad/s', rate_source='dOmega_z_ds',targets=['Omega_z'])
phase.add_state('T', fix_initial=False, fix_final=False, units='C', rate_source='dT_ds',targets=['T'])
phase.add_state('e', fix_initial=True, fix_final=False, units='J', rate_source='de_ds',targets=['e'])
#Add lower = 0

#Define Controls
phase.add_control(name='Omegadot_z', units='rad/s**2', lower=None, upper=None,fix_initial=False,fix_final=False, targets=['Omegadot_z'],ref=0.04) #steering angle
phase.add_control(name='tau_t', lower=None, upper=None, units='N*m',fix_initial=False,fix_final=False, targets=['tau_t']) #the thrust controls the longitudinal force of the rear tires and is positive while accelerating, negative while braking
phase.add_control(name='tau_b', lower=None, upper=None ,units='N*m',fix_initial=False,fix_final=False, targets=['tau_b'])

#Physical Constraints
phase.add_path_constraint('im',shape=(1,),units='A',lower=2,upper=160)#Max motor current
phase.add_path_constraint('TC',shape=(1,),units=None,upper=1)#Max tyre constraint
phase.add_path_constraint('N',shape=(1,),units=None,lower=0)#Enforce positive load
#Add max power


#Some of the vehicle design parameters are available to set here. Other parameters can be found in their respective ODE files.
phase.add_design_parameter('ei',val=50000.0,units='J',opt=False,targets=['powerTrain.ei'],dynamic=False) 

#Minimize final time.
phase.add_objective('t', loc='final') #note that we use the 'state' time instead of Dymos 'time'

#Add output timeseries
#phase.add_timeseries_output('Phi',units='rad/s',shape=(1,))

#Link the states at the start and end of the phase in order to ensure a continous lap
traj.link_phases(phases=['phase0', 'phase0'], vars=['Phi','Phidot','n','alpha','V','Beta','z','zdot','omega_w','Omega_z','T'], locs=('final', 'initial'))



IPOPT = True

# Set the driver. IPOPT or SNOPT are recommended but SLSQP might work.

if IPOPT:
	p.driver = om.pyOptSparseDriver(optimizer='IPOPT')

	# p.driver.opt_settings['mu_init'] = 1e-3
	# p.driver.opt_settings['max_iter'] = 1
	# p.driver.opt_settings['acceptable_tol'] = 1e-3
	# p.driver.opt_settings['constr_viol_tol'] = 1e-3
	# p.driver.opt_settings['compl_inf_tol'] = 1e-3
	# p.driver.opt_settings['acceptable_iter'] = 0
	# p.driver.opt_settings['tol'] = 1e-3
	# p.driver.opt_settings['hessian_approximation'] = 'exact'
	# p.driver.opt_settings['nlp_scaling_method'] = 'none'
	p.driver.opt_settings['print_level'] = 5
	p.driver.options['user_terminate_signal'] = None
else:
	p.driver = om.ScipyOptimizeDriver()

p.driver.declare_coloring()
path = os.getcwd()
coloring_path = path + '/coloring_files/total_coloring.pkl'
#p.driver.use_fixed_coloring(coloring=coloring_path)
p.setup(check=True) #force_alloc_complex=True

#States
p.set_val('traj.phase0.states:t',phase.interpolate(ys=[0.0,100.0], nodes='state_input'),units='s') #initial guess for what the final time should be
p.set_val('traj.phase0.states:Phi',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
p.set_val('traj.phase0.states:Phidot',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad/s')
p.set_val('traj.phase0.states:n',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m')
p.set_val('traj.phase0.states:alpha',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
p.set_val('traj.phase0.states:V',phase.interpolate(ys=[20,20], nodes='state_input'),units='m/s') #non-zero velocity in order to protect against 1/0 errors.
p.set_val('traj.phase0.states:Beta',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
p.set_val('traj.phase0.states:z',phase.interpolate(ys=[0.1,0.1], nodes='state_input'),units='m')
p.set_val('traj.phase0.states:zdot',phase.interpolate(ys=[0.1,0.1], nodes='state_input'),units='m/s')
p.set_val('traj.phase0.states:omega_w',phase.interpolate(ys=[0.1,0.1], nodes='state_input'),units='rad/s')
p.set_val('traj.phase0.states:Omega_z',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad/s')
p.set_val('traj.phase0.states:T',phase.interpolate(ys=[22,22], nodes='state_input'),units='C')
p.set_val('traj.phase0.states:e',phase.interpolate(ys=[50000,0.0], nodes='state_input'),units='J')

#Controls
p.set_val('traj.phase0.controls:Omegadot_z',phase.interpolate(ys=[0.1,0.1], nodes='control_input'),units='rad/s**2')
p.set_val('traj.phase0.controls:tau_t',phase.interpolate(ys=[0.1, 0.1], nodes='control_input'),units='N*m')
p.set_val('traj.phase0.controls:tau_b',phase.interpolate(ys=[0, 0], nodes='control_input'),units='N*m')


p.run_driver()
print('Optimization finished')

#Get optimized time series
n = p.get_val('traj.phase0.timeseries.states:n')
t = p.get_val('traj.phase0.timeseries.states:t')
s = p.get_val('traj.phase0.timeseries.time')
V = p.get_val('traj.phase0.timeseries.states:V')



#Plotting
if plot:
	print("Plotting")


	#We know the optimal distance from the centerline (n). To transform this into the racing line we fit a spline to the displaced points. This will let us plot the racing line in x/y coordinates
	trackLength = track.getTotalLength()
	normals = getGateNormals(finespline,slope)
	newgates = []
	newnormals = []
	newn = []
	for i in range(len(n)):
		index = ((s[i]/s_final)*np.array(finespline).shape[1]).astype(int) #interpolation to find the appropriate index
		if index[0]==np.array(finespline).shape[1]:
			index[0] = np.array(finespline).shape[1]-1
		if i>0 and s[i] == s[i-1]:
			continue
		else:
			newgates.append([finespline[0][index[0]],finespline[1][index[0]]])
			newnormals.append(normals[index[0]])
			newn.append(n[i][0])

	newgates = reverseTransformGates(newgates)
	displacedGates = setGateDisplacements(newn,newgates,newnormals)
	displacedGates = np.array((transformGates(displacedGates)))

	npoints = 1000
	displacedSpline,gates,gatesd,curv,slope = getSpline(displacedGates,1/npoints,0) #fit the racing line spline to npoints

	plt.rcParams.update({'font.size': 12})


	def plotTrackWithData(state,s):
		#this function plots the track
		state = np.array(state)[:,0]
		s = np.array(s)[:,0]
		s_new = np.linspace(0,s_final,npoints)

		#Colormap and norm of the track plot
		cmap = mpl.cm.get_cmap('viridis')
		norm = mpl.colors.Normalize(vmin=np.amin(state),vmax=np.amax(state))

		fig, ax = plt.subplots(figsize=(15,6))
		plt.plot(displacedSpline[0],displacedSpline[1],linewidth=0.1,solid_capstyle="butt") #establishes the figure axis limits needed for plotting the track below

		plt.axis('equal')
		plt.plot(finespline[0],finespline[1],'k',linewidth=linewidth_from_data_units(8.5,ax),solid_capstyle="butt") #the linewidth is set in order to match the width of the track
		plt.plot(finespline[0],finespline[1],'w',linewidth=linewidth_from_data_units(8,ax),solid_capstyle="butt") #8 is the width, and the 8.5 wide line draws 'kerbs'
		plt.xlabel('x (m)')
		plt.ylabel('y (m)')

		#plot spline with color
		for i in range(1,len(displacedSpline[0])):
			s_spline = s_new[i]
			index_greater = np.argwhere(s>=s_spline)[0][0]
			index_less = np.argwhere(s<s_spline)[-1][0]

			x = s_spline
			xp = np.array([s[index_less],s[index_greater]])
			fp = np.array([state[index_less],state[index_greater]])
			interp_state = np.interp(x,xp,fp) #interpolate the given state to calculate the color

			#calculate the appropriate color
			state_color = norm(interp_state)
			color = cmap(state_color)
			color = mpl.colors.to_hex(color)

			#the track plot consists of thousands of tiny lines:
			point = [displacedSpline[0][i],displacedSpline[1][i]]
			prevpoint = [displacedSpline[0][i-1],displacedSpline[1][i-1]]
			if i <=5 or i == len(displacedSpline[0])-1:
				plt.plot([point[0],prevpoint[0]],[point[1],prevpoint[1]],color,linewidth=linewidth_from_data_units(1.5,ax),solid_capstyle="butt",antialiased=True)
			else:
				plt.plot([point[0],prevpoint[0]],[point[1],prevpoint[1]],color,linewidth=linewidth_from_data_units(1.5,ax),solid_capstyle="projecting",antialiased=True)

		clb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),fraction = 0.02, pad=0.04) #add colorbar

		if np.array_equal(state,V[:,0]):
			clb.set_label('Velocity (m/s)')

		plt.tight_layout()
		plt.grid()

	#Create the plots
	plotTrackWithData(V,s)


	#Plot the main vehicle telemetry
	fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 8))

	#Velocity vs s
	axes[0].plot(s,
			p.get_val('traj.phase0.timeseries.states:V'), label='solution')

	axes[0].set_xlabel('s (m)')
	axes[0].set_ylabel('V (m/s)')
	axes[0].grid()
	axes[0].set_xlim(0,s_final)

	#n vs s
	axes[1].plot(s,
			p.get_val('traj.phase0.timeseries.states:n', units='m'), label='solution')

	axes[1].set_xlabel('s (m)')
	axes[1].set_ylabel('n (m)')
	axes[1].grid()
	axes[1].set_xlim(0,s_final)


	plt.tight_layout()
	plt.show()
