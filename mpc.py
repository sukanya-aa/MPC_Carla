import carla
import numpy as np
import scipy.optimize as opt

class KinematicBicycleMPC:
    def __init__(self, vehicle, dt=0.05, prediction_horizon=10, n_controls=2, n_states=4, bounds=None):
        self.vehicle = vehicle
        self.dt = dt
        self.prediction_horizon = prediction_horizon
        self.n_controls = n_controls
        self.n_states = n_states
        self.bounds = bounds
        
        self.x_k = np.zeros(n_states)  # current state
        self.u_k = np.zeros(n_controls)  # current controls
        
        self.x_pred = np.zeros((prediction_horizon+1, n_states))  # predicted states
        self.u_pred = np.zeros((prediction_horizon, n_controls))  # predicted controls
        self.cost = np.zeros(prediction_horizon)  # predicted costs
        
        self.set_initial_state()
        
    def set_initial_state(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        self.x_k = np.array([transform.location.x, transform.location.y,
                             transform.rotation.yaw, np.linalg.norm(velocity)])
        
    def update_state(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        self.x_k = np.array([transform.location.x, transform.location.y,
                             transform.rotation.yaw, np.linalg.norm(velocity)])
        
    def get_model_prediction(self, x0, u):
        x = np.zeros((self.prediction_horizon+1, self.n_states))
        x[0,:] = x0
        
        for i in range(self.prediction_horizon):
            x[i+1,:] = self.propagate_dynamics(x[i,:], u[i,:])
            
        return x
    
    def propagate_dynamics(self, x, u):
        L = 2.5  # wheelbase
        a = L/2.0  # distance from center of mass to front axle
        b = L/2.0  # distance from center of mass to rear axle
        
        x_next = np.zeros(self.n_states)
        x_next[0] = x[0] + x[3]*np.cos(x[2])*self.dt
        x_next[1] = x[1] + x[3]*np.sin(x[2])*self.dt
        x_next[2] = x[2] + (x[3]/L)*np.tan(u[0])*self.dt
        x_next[3] = x[3] + u[1]*self.dt - 0.1*x[3]*self.dt
        
        return x_next
    
    def cost_function(self, u, x_k, x_ref):
        Q = np.diag([10, 10, 1, 1])
        R = np.diag([1, 1])
        
        cost = 0.0
        x = x_k.copy()
        
        for i in range(self.prediction_horizon):
            cost += np.dot((x - x_ref[i,:]), np.dot(Q, (x - x_ref[i,:]).T))
            cost += np.dot(u[i,:], np.dot(R, u[i,:].T))
            x = self.propagate_dynamics(x, u[i,:])
            
        return cost
                                 
    def optimize_controls(self, x_ref):
        u0 = np.zeros((self.prediction_horizon, self.n_controls))
        bounds = self.bounds or [(None, None) for _ in range(self.prediction_horizon*self.n_controls)]
        options = {'maxiter': 100, 'ftol': 1e-04, 'iprint': 0, 'disp': True}
        res = opt.minimize(self.cost_function, u0, args=(self.x_k, x_ref),
                           bounds=bounds, method='SLSQP', options=options)
        
        self.u_pred = res.x.reshape(self.prediction_horizon, self.n_controls)
        self.cost = res.fun
        self.x_pred = self.get_model_prediction(self.x_k, self.u_pred)
        
    def get_control_action(self):
        return self.u_pred[0,:]
    
    def update(self):
        self.update_state()
        self.optimize_controls(self.x_pred[1:,:])
        
    def apply_control_action(self, control_action):
        steer = control_action[0]
        throttle = control_action[1]
        brake = 0.0
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        
    def run_mpc(self):
        while True:
            control_action = self.get_control_action()
            self.apply_control_action(control_action)
            self.update()
