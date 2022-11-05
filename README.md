# Ebike-LTS
Run through problemSolver.py

At the time Linux/Unix install was neccessary as although some solvers (e.g. IPOPT) could be run on windows, the tools to make them compatible with Dymos were only for Linux.

I was not great with specifying depencies but I believe you need (I can't remember which versions):

OpenMDAO + Dymos, NumPy, Matplotlib, SciPy
And a solver:  IPOPT (or another such as ma57)

# Purpose of the tool

This tool was designed to simulate a vehicle to simultaneously find:

1) Vehicle inputs to minimise laptime
2) The vehicle parameters (battery weight and capacity, coolant flow rate and volume, gear ratio) to minimise laptop

The combination of (1) and (2) I believe were something missing from the literature for electric vehicles - previously people ran simulations dozens of times to optimise one value at a time.

The motivation for our team in particular was missing out on the 2020 Isle of Man TT Zero podium finish due to unoptimised gear ratios. As COVID was ongoing during this project, a simulation gave our team an excellent way to test the designs we were developing.

# Cool diagrams

Here is one of the main outputs of the tool - the acceleration and trajectory of the vehicle around the lap. Notice how we approach on the outside then hug the inside, and that we brake into corners then accelerate out. We have managed to find the optimal behaviour that real drivers use to maximise speed around corners. 

![Acceleration trajectory](https://i.imgur.com/2ix0gn6.png)

As well as trajectory diagrams we can plot things like thermals, battery state of charge, and so on...

![Thermal plot](https://i.imgur.com/1FdNUFn.png)

Here's one of the parameters we optimised, initial battery charge. Notice how excessive battery charge makes you slower (due to increased battery weight), and leaves some charge in the cell by the end of the lap.

![Battery optimisation](https://i.imgur.com/mIZyujs.png)

Here is one of the models for the vehicle on the track (Omega - yaw rate, alpha - direction relative to track, n - distance from centerline, V/U - velocity components, r - radius from road section center). 

![Vehicle diagram](https://i.imgur.com/dVNA70V.png)

And this is the model of the cooling system we planned to develop (but were unable to due to COVID).

![Cooling diagram](https://i.imgur.com/dIj1cZA.png)



