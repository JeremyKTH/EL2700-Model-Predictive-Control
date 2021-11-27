import numpy as np

from astrobee_1d import Astrobee
from controller import Controller
from simulation import EmbeddedSimEnvironment
from control.matlab import *
import casadi as ca


# Create pendulum and controller objects
abee = Astrobee(h=0.1) #h being the sampling time (e^(Ach))
ctl = Controller()

"""Question 1"""
# Get the system discrete-time dynamics (Task 1)
A, B = abee.one_axis_ground_dynamics()

"""Question 2"""
C = ca.DM.zeros(1, 2) #Initializing output and feethrough matrices
D = ca.DM.zeros(1)

# TODO: Get the discrete time system with casadi_c2d
Ad, Bd, Cd, Dd = abee.casadi_c2d(A, B, C, D) #Using Casadi functions to discretize continuous system (by means of jacobian)
abee.set_discrete_dynamics(Ad, Bd, Cd, Dd) #setting self class object with discretized matrices

print("Discretized A:", Ad, end = '\n\n') #Print discretized matrix 2x2
print("Discretized B:", Bd, end = '\n\n') #Print discretized matrix 2x2

"""Question 3"""
C[0] = 1 # Per instruction, C = [1, 0], D = 0 already
sys = ss2tf(Ad, Bd, Cd, Dd) #Calculating the Transfer Function [b,a] = C(SI-A)^(-1)B + D
print("Transfer Function: ", sys)

# Plot poles and zeros...

#NEED TO ADJUST dt IN abee.poles_zeros depending on cont. or discrete..
# dt == 0 -> Continuous time system
# dt != 0 -> Discrete time system
#abee.poles_zeros(A, B, C, D) #continuous pole/zero
abee.poles_zeros(Ad, Bd, Cd, Dd) #Discrete pole/zero

"""Question 4"""
# Get control gains
ctl.set_system(Ad, Bd, Cd, Dd)

p = ctl.set_poles([.981, .98]) #Pole placement to meet desired parameters
K = ctl.get_closed_loop_gain(p)

x0 = [1.0, 0.0] #initial reference

# Set the desired reference based on the dock position and zero velocity on docked position
dock_target = np.array([[0.0, 0.0]]).T #x target array
ctl.set_reference(dock_target)

# Starting position
x0 = [1.0, 0.0]

# Initialize simulation environment
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.control_law,
                                 time=40.0)
t, y, u = sim_env.run(x0)
sim_env.visualize()

# Disturbance effect
abee.set_disturbance()
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.control_law,
                                 time=40.0)
t, y, u = sim_env.run(x0)
sim_env.visualize()

# Activate feed-forward gain
ctl.activate_integral_action(dt=0.1, ki=0.027) 
t, y, u = sim_env.run(x0)
sim_env.visualize()
