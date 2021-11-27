from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca
from casadi.casadi import horzcat, vertcat
import numpy as np
import numpy.matlib as nmp
from util import *


class Astrobee(object):
    def __init__(self,
                 trajectory_file,
                 mass=9.6,
                 inertia=np.diag([0.1534, 0.1427, 0.1623]),
                 h=0.1,
                 **kwargs):
        """
        Astrobee Robot, NMPC tester class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param inertia: inertia tensor of the Astrobee
        :type inertia: np.diag
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        :param model: select between 'euler' or 'quat'
        :type model: str
        """

        # Model
        self.nonlinear_model = self.astrobee_dynamics_quat
        self.n = 13
        self.m = 6
        self.dt = h

        # Model prperties
        self.mass = mass
        self.inertia = inertia

        # Set CasADi functions
        self.set_casadi_options()

        # Set nonlinear model with a RK4 integrator
        self.model = self.rk4_integrator(self.nonlinear_model)
        self.model_test =self.rk4_integrator(self.astrobee_dynamics_quat_test)

        # Set path for trajectory file
        self.trajectory_file = trajectory_file

      

    def set_casadi_options(self):
        """
        Helper function to set casadi options.
        """
        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

    def astrobee_dynamics_quat(self, x, u):
        """
        Astrobee nonlinear dynamics with Quaternions.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:]
       

        # 3D Force
        f = u[0:3]

        # 3D Torque
        tau = u[3:]

        # Model
        pdot = ca.MX.zeros(3, 1)
        vdot = ca.MX.zeros(3, 1)
        qdot = ca.MX.zeros(4, 1)
        wdot = ca.MX.zeros(3, 1)

        #Q1
        #Assert dynamics of the model
        """pdot = v
        vdot = (1/self.mass)*ca.mtimes(R_q,f)
        qdot = 0.5*ca.mtimes(Attitude_jacobian,w)
        wdot = np.linalg.inv(self.inertia)@(tau-w*(self.inertia@w)"""

        pdot = v
        vdot = ca.mtimes(r_mat_q(q), f) / self.mass
        qdot =ca.mtimes(xi_mat(q), w) / 2
        #wdot = ca.mtimes(ca.inv(self.inertia), tau - w, ca.mtimes(self.inertia,w))
        #wdot = ca.mtimes(ca.inv(self.inertia), tau - w*ca.mtimes(self.inertia,w))
        
        #wdot = ca.mtimes(np.linalg.inv(self.inertia), (tau - ca.cross(w, ca.mtimes(self.inertia,w))))

        wdot = ca.mtimes(ca.inv(self.inertia), tau - ca.mtimes(skew(w),
                         ca.mtimes(self.inertia, w)))
        

        #vdot = (1/self.m)


        dxdt = [pdot, vdot, qdot, wdot]

        return ca.vertcat(*dxdt)

    def astrobee_dynamics_quat_test(self, x, u):
        """
        Astrobee nonlinear dynamics with Quaternions.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:]
       

        # 3D Force
        f = u[0:3]

        # 3D Torque
        tau = u[3:]

        # Model
        pdot = ca.MX.zeros(3, 1)
        vdot = ca.MX.zeros(3, 1)
        qdot = ca.MX.zeros(4, 1)
        wdot = ca.MX.zeros(3, 1)

        #Q1
        
        mass_perc = 1
        intertia_perc = 1
        #Assert dynamics of the model
        """pdot = v
        vdot = (1/self.mass)*ca.mtimes(R_q,f)
        qdot = 0.5*ca.mtimes(Attitude_jacobian,w)
        wdot = np.linalg.inv(self.inertia)@(tau-w*(self.inertia@w)"""

        pdot = v
        vdot = ca.mtimes(r_mat_q(q), f) / (self.mass*mass_perc)
        qdot =ca.mtimes(xi_mat(q), w) / 2
        #wdot = ca.mtimes(ca.inv(self.inertia), tau - w, ca.mtimes(self.inertia,w))
        #wdot = ca.mtimes(ca.inv(self.inertia), tau - w*ca.mtimes(self.inertia,w))
        
        #wdot = ca.mtimes(np.linalg.inv(self.inertia), (tau - ca.cross(w, ca.mtimes(self.inertia,w))))

        wdot = ca.mtimes(ca.inv(self.inertia*intertia_perc), tau - ca.mtimes(skew(w),
                         ca.mtimes(self.inertia*intertia_perc, w)))
        

        #vdot = (1/self.m)


        dxdt = [pdot, vdot, qdot, wdot]

        return ca.vertcat(*dxdt)

    def rk4_integrator(self, dynamics):
        """
        Runge-Kutta 4th Order discretization.
        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state at next step
        :rtype: ca.MX
        """
        x0 = ca.MX.sym('x0', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)

        x = x0

        k1 = dynamics(x, u)
        k2 = dynamics(x + self.dt / 2 * k1, u)
        k3 = dynamics(x + self.dt / 2 * k2, u)
        k4 = dynamics(x + self.dt * k3, u)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion: TODO(Pedro-Roque): check best way to propagate
        rk4 = ca.Function('RK4', [x0, u], [xdot], self.fun_options)

        return rk4

    def forward_propagate(self, x_s, npoints, radius=0.5):
        """
        Forward propagate the observed state given a constant velocity.

        The output should be a self.n x npoints matrix, with the
        desired offset track.

        :param x_s: starting state
        :type x_s: np.ndarray
        :param npoints: number of points to propagate
        :type npoints: int
        :return: forward propagated trajectory
        :rtype: np.ndarray
        """
        p = x_s[0:3]
        v = x_s[3:6]
        q = x_s[6:10]
        w = x_s[10:]
        
        

        pdot = np.zeros((3, 1))
        vdot = np.zeros((3, 1))
        qdot = np.zeros((4, 1))
        wdot = np.zeros((3, 1))
        Rmat = np.zeros((4, 3))
        pb = np.zeros((3,1))
        x_r = np.zeros((self.n, npoints))
        #print(x_r)
        
        #v_c = 0.1*np.eye(3,1)
        #w_c = 0.1*np.eye(3,1)
        # TODO: do the forward propagation of the measured state x_s for npoints.
        #pdot = v
        #qdot = xi_mat_np(q)@w / 2
        pdot = v
        qdot = np.dot(xi_mat_np(q),w) * 0.5
        for i in range(npoints):
            
            p_t = (p + pdot*(self.dt*i)).reshape(3,1)
            #print(p,"p")
            q_t = q + qdot*(self.dt*i)
            q_t = q_t/np.linalg.norm(q_t)
            #print(q,"q")
            Rmat = (r_mat_q_np(q_t)[:,0]*radius).reshape(3,1)
            #print(Rmat,"Xi")
            #pb = np.add(p , (Rmat[:,0].T)*radius)
            pb = p_t + Rmat
            #print(pb,"pbbb")
            #print(np.array(vertcat(pb,v,q,w)).T)
            #x_r[:,i] = np.vstack((pb,v,q,w)).resize(13,)
            x_r[:,i] = np.array(vertcat(pb,v,q_t,w).T)
           
            #print(x_r,"x_r")
            #if i == 24:
                #print("p_t", p_t.T)
                #print("q_t", q_t.T)
                #print("x_r", x_r[:, 24])
       # print(pb[:,0],"pb")
        #print(v)
        #print(q)
        #print(w)
        #print(x_r)
       # print(x_r)
        return x_r
    
        """for t in range(npoints):
            # Calculate new position and quaternion
            p_t = p + pdot*(self.dt*t)
            q_t = q + qdot*(self.dt*t)
            q_t /= np.linalg.norm(q_t)
            p_B = p_t + (r_mat_q_np(q_t)[:,0]*radius).reshape(3,1)

            x_r[:, t] = np.vstack((p_B, v, q_t, w)).T

            if t == 24:
                print("p_t", p_t.T)
                print("q_t", q_t.T)
                #print("x_r", x_r[:, 24])
            #x_r[:, t] = np.array(ca.vertcat(p_B, v, q, w)).resize(13,)

        return x_r"""
    def get_trajectory(self, t, npoints, forward_propagation=False):
        """
        Provide trajectory to be followed.
        :param t0: starting time
        :type t0: float
        :param npoints: number of trajectory points
        :type npoints: int
        :return: trajectory with shape (Nx, npoints)
        :rtype: np.array
        """

        if t == 0.0:
            tmp = np.loadtxt(self.trajectory_file, ndmin=2)
            self.trajectory = tmp.reshape((self.n, int(tmp.shape[0] / self.n)), order="F")

        if forward_propagation is False:
            id_s = int(round(t / self.dt))
            id_e = int(round(t / self.dt)) + npoints
            x_r = self.trajectory[:, id_s:id_e]
        else:
            # Take a point and propagate the kinematics
            id_s = int(round(t / self.dt))
            x_start = self.trajectory[:, id_s]
            x_r = self.forward_propagate(x_start, npoints)

        return x_r

    def get_initial_pose(self):
        """
        Helper function to get a starting state, depending on the dynamics type.

        :return: starting state
        :rtype: np.ndarray
        """
        x0 = np.zeros((self.n, 1))
        x0[0] = 11.0
        x0[1] = -7.5
        x0[2] = 5.2
        x0[8] = 1.0

        return x0

    def get_static_setpoint(self):
        """
        Helper function to get the initial state of Honey for setpoint stabilization.
        """
        xd = np.array([[11, -7, 4.8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).T
        return xd

    def get_limits(self):
        """
        Get Astrobee control and state limits for ISS

        :return: state and control limits
        :rtype: np.ndarray, np.ndarray
        """
        u_lim = np.array([[0.85, 0.41, 0.41, 0.085, 0.041, 0.041]]).T

        x_lim = np.array([[13, 13, 13,
                           0.5, 0.5, 0.5,
                           1, 1, 1, 1,
                           0.1, 0.1, 0.1]]).T

        return u_lim, x_lim

    # ----------------------------------------
    #               Unit Tests
    # ----------------------------------------

    def test_forward_propagation(self):
        """
        Unit test to check if forward propagation is correctly implemented.
        """

        x0 = np.array([[11, 0.3, 0.4, 0, 0.1, 0, 0, 0, 0, 1, 0.1, 0, 0]]).T
        x_r = self.forward_propagate(x0, 30)
        xd = np.array([11.5, 0.54, 0.4, 0.0, 0.1, 0.0, 0.11971121, 0.0, 0.0, 0.99280876, 0.1, 0.0, 0.0])
        eps = np.linalg.norm(x_r[:, 24] - xd)
        if eps > 1e-2:
            print("Forward propagation has a large error. Double check your dynamics.")
            exit()

    def test_dynamics(self):
        """
        Unit test to check if the Astrobee dynamics are correctly set.
        """
        x0 = np.array([[11, 0.3, 0.4, 0, 0.1, 0, 0, 0, 0, 1, 0.1, 0, 0]]).T
        u0 = np.array([[.1, .1, .1, .01, .01, .01]])
        xd = np.array([[11.0001, 0.310052, 0.400052,
                        0.00104168, 0.101036, 0.00104685,
                        0.00516295, 0.000174902, 0.000154287, 0.999987,
                        0.106519, 0.0070057, 0.00615902]]).T
        xt = self.model(x0, u0)
        eps = np.linalg.norm(np.array(xt) - xd)
        if eps > 1e-4:
            print("Forward propagation has a large error. Double check your dynamics.")
            exit()
