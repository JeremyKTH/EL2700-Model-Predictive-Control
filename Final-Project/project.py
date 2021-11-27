import numpy as np

from astrobee import Astrobee
from mpc import MPC
from simulation import EmbeddedSimEnvironment
import casadi as ca

# TODO: Set the path to the trajectory file:
#       eg.: trajectory_quat = '/home/roque/Project Assignment/Dataset/trajectory_quat.txt'
trajectory_quat = './Dataset/trajectory_quat.txt'

# TODO: complete the 'tuning_file_path' variable to the path of your tuning.yaml
#       eg.: tuning_file_path = '/home/roque/Project Assignment/tuning.yaml'
tuning_file_path = './tuning.yaml'

#BRYSON RULE Calc for cost matrices
x_calc = ca.MX.ones(12,1)
u_calc = ca.MX.ones(6,1)

x_calc[0:3] = (1/.06)**2 #position
x_calc[3:6] = (1/.03)**2 #vel
x_calc[6:10] = (1/(10**(-7)))**2#quaternion
x_calc[10:] = (1/1)**2 #angular vel

u_calc[0:3] = (1/.85)**2
u_calc[3:6] = (1/.04)**2

print("Calculated tuning for Q: \n", x_calc, "\n")
print("Calculated tuning for R: \n", u_calc)

# Q1
# TODO: Set the Astrobee dynamics in Astrobee->astrobee_dynamics_quat
abee = Astrobee(trajectory_file=trajectory_quat)

# If successful, test-dynamics should not complain
abee.test_dynamics()

# Instantiate controller
u_lim, x_lim = abee.get_limits()

# Create MPC Solver
# TODO: Select the parameter type with the argument param='P1'  - or 'P2', 'P3'
MPC_HORIZON = 10
ctl = MPC(model=abee,
          dynamics=abee.model_test,
          param='P4',
          N=MPC_HORIZON,
          ulb=-u_lim, uub=u_lim,
          xlb=-x_lim, xub=x_lim,
          tuning_file=tuning_file_path)

# Q2: REFERENCE TRACKING (STATIC)
x_d = abee.get_static_setpoint() #setting the reference 

 
ctl.set_reference(x_d)
print(x_d,"ref Q2")
# Set initial state
x0 = abee.get_initial_pose()

# Q3: ACTIVATE REFERENCE TRACKING (TIME-VARYING)
tracking_ctl = MPC(model=abee,
                   dynamics=abee.model_test,
                   param='P4',
                   N=MPC_HORIZON,
                   trajectory_tracking=True,
                   ulb=-u_lim, uub=u_lim,
                   xlb=-x_lim, xub=x_lim)
sim_env_tracking = EmbeddedSimEnvironment(model=abee,
                                          dynamics=abee.model,
                                          controller=tracking_ctl.mpc_controller,
                                          time=40)
#t, y, u = sim_env_tracking.run(x0)
print("Q3 state tracking")
#sim_env_tracking.visualize()  # Visualize state propagation
#sim_env_tracking.visualize_error()

# Q4: iMPLEMENT FEED FORWARE PROPOGATION LAW
print("Q4")
# Test 3: Activate forward propagation
abee.test_forward_propagation()
tracking_ctl.set_forward_propagation()
t, y, u = sim_env_tracking.run(x0)
sim_env_tracking.visualize()  # Visualize state propagation
sim_env_tracking.visualize_error()
