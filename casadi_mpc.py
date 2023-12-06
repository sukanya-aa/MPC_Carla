#*********************************************************************#
#This is an implementation of mpc design using single shooting method 
# in carla simulator. The program designs MPC by transforming the 
# optimal control problem in to a nonlinear programming problem. 
#*********************************************************************#
import glob
import sys
import os
import numpy as np
import random
import cv2
import time
import queue
from casadi import *
import matplotlib.pyplot as plt
import pygame
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

sys.path.append(r'C:\CARLA_0.9.8\WindowsNoEditor\PythonAPI\carla\agents\navigation')
from plannerbyme import GlobalRoutePlanMe


def handle_image(disp, image):
    og_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(og_array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:,:,::-1]
    array = array.swapaxes(0,1)
    surface = pygame.surfarray.make_surface(array)

    disp.blit(surface, (0,0))
    pygame.display.flip()

def convert_theta(theta):
    if theta < 0:
        theta = 360 - abs(theta)
    return theta


#Get initial state
# def initial_state(vehicle):
#     x0 = vehicle.get_location().x
#     y0 = vehicle.get_location().y
#     theta0 = convert_theta(vehicle.get_transform().rotation.yaw)
#     v0 = np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
#     return np.array([x0, y0, theta0, v0])

# def shift(T, t0, x0, u, f):
#     st = x0
#     con = u[0,:].T
#     f_value = f(st, con)
#     st = st + (T*f_value)
#     x0 = st.full()
#     t0 = t0 + T
#     u0 = vertcat(u[1:, :], repmat(u[-1, :], 1, 1))


actor_list = []

#set up the carla environment
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

#Set up the world
world = client.get_world()
blueprint_library = world.get_blueprint_library()


start = carla.Transform(carla.Location(x=-148.585938, y=112.971313, z=0.000000), carla.Rotation(pitch=0.000000, yaw=80.012032, roll=0.000000))
world.debug.draw_string(start.location, 'O',
                        color=carla.Color(255,0,0),
                        life_time=30.0,
                        persistent_lines=True)
#spawn the vehicle
start.location.z += 1
vehicle = world.spawn_actor(blueprint_library.find('vehicle.tesla.model3'), start)
time.sleep(3)
actor_list.append(vehicle)
spectator = world.get_spectator()
vt = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4, z=3)), vehicle.get_transform().rotation)
spectator.set_transform(vt)
#spawn the camera
# cam = world.spawn_actor(blueprint_library.find('sensor.camera.rgb'),
#                                carla.Transform(carla.Location(x=2.5, z=0.7)),
#                                attach_to = vehicle)
# actor_list.append(cam)

# #display the camera
# display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
# cam.listen(lambda image: handle_image(display, image))

#define trajectory
wps = world.get_map().get_topology()
route = wps[113][0].next_until_lane_end(5.0) 

# print(route)
# for i in route:
#     print(i)


#quit()
#print(start.location)
for i in route:
    world.debug.draw_string(i.transform.location, 'O', 
                            draw_shadow=False, 
                            color=carla.Color(0,255,0), 
                            life_time=20.0, 
                            persistent_lines=True)
    

end = route[-1].transform
    


#*********************************************************************#
#****************************MPC Design*******************************#
#*********************************************************************#

#Define the parameters
T = 1 #sampling time
N = 10 #prediction horizon
#sim_time = 50

#States
x = SX.sym('x')  #x position
y = SX.sym('y')  #y position
theta =SX.sym('theta') #heading angle
v = SX.sym('v') #velocity
states = vertcat(x,y,theta,v)
n_states = states.numel()

#Controls
a = SX.sym('a') #throttle
delta = SX.sym('delta') #steering angle
controls = vertcat(a,delta)
n_controls = controls.numel()

#Control bounds
a_min = 0
a_max = 0.5
delta_min = -0.5
delta_max = 0.5



#Kinematic bicycle model from centre of mass of the vehicle
lr = 1.4 #distance from centre of mass to rear axle
lf = 1.6 #distance from centre of mass to front axle
#beta = arctan(lr*tan(delta)/(lr + lf))
rhs = vertcat((v*cos(theta)),
                (v*sin(theta)),
                (v*tan(delta)),
                a)

#Discrete time dynamics
f = Function('f', [states, controls], [rhs])  #Nonlinear mapping function f(x,u)
U = SX.sym('U', n_controls, N)
P = SX.sym('P', n_states+n_states)
X = SX.sym('X', n_states, (N+1))
X[:,0] = P[0:4]

for k in range(N):
    st = X[:,k]
    con = U[:,k]
    f_value = f(st, con)
    st_next = st + (T*f_value)
    X[:,k+1] = st_next

ff = Function('ff', [U,P], [X])


#Objective function and constraints
obj = 0
g = SX.sym('g', 4, (N+1))  #constraints vector
Q = diag([3600, 3600, 1900, 2])   #State weights
R = diag([0, 8000])        #Control weights
# st = X[:,0]  #Initial state
# g.append(st-P[0:4])  #Initial condition constraints
for k in range(N):
    st = X[:,k]
    con = U[:,k]
    obj = obj + \
          (st-P[4:8]).T @ Q @ (st-P[4:8]) + \
          (con.T @ R @ con)
    st_next = X[:,k+1]
    k1 = f(st, con)
    k2 = f(st + (T/2)*k1, con)
    k3 = f(st + (T/2)*k2, con)
    k4 = f(st + T*k3, con) 
    st_next_RK = st + (T/6)*(k1 + 2*k2 + 2*k3 + k4)
    # f_value = f(st, con)
    # st_next_euler = st + (T*f_value)
    g.append(st_next-st_next_RK)

# for k in range(0, N+1):
#     g[0,k] = X[0,k]
#     g[1,k] = X[1,k]
#     g[2,k] = X[2,k]
#     g[3,k] = X[3,k]
# g = reshape(g, 4*(N+1), 1)

#Make the decision variables one column vector
OPT_variables = reshape(U, 2*N, 1)
#OPT_variables = vertcat(reshape(X, 4*(N+1), 1), reshape(U, 2*N, 1))
nlp_prob = {
    'f': obj,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6},
    'print_time': 0}

solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = []
ubx = []
for i in range(2*N):
    if i%2 == 0:
        lbx.append(a_min)
        ubx.append(a_max)
    else:
        lbx.append(delta_min)
        ubx.append(delta_max)

lbx = np.transpose(lbx)
ubx = np.transpose(ubx)

lbg = []
ubg = []
for i in range(0, 4*(N+1), 4):
    lbg.append(-inf)
    lbg.append(-inf)
    lbg.append(0)
    lbg.append(0)
    ubg.append(inf)
    ubg.append(inf)
    ubg.append(2*pi)
    ubg.append(3)


x0 = np.transpose([start.location.x, start.location.y, convert_theta(start.rotation.yaw), 0])
xs = np.transpose([end.location.x, end.location.y, convert_theta(start.rotation.yaw), 3])
p = np.transpose([start.location.x, start.location.y, convert_theta(start.rotation.yaw), 0, end.location.x, end.location.y, convert_theta(start.rotation.yaw), 3])
c = 0
mpciter = 0
u0 = (DM.zeros(2*N, 1))
ucl = []
# while c < len(route):
#     if (norm_2(x0[0:2]- p[4:6])) < 3:
#         c += 1
#         end = route[c].transform
#         world.debug.draw_string(end.location, 'o', draw_shadow=False,
#                                 color=carla.Color(r=0, g=0, b=255), life_time=5,
#                                 persistent_lines=True)
#     print(x0,"---",p[4:8])
#     u0 = reshape(u0, 2*N, 1)
#     p[0:4] = x0
#     p[4:8] = [end.location.x, end.location.y, convert_theta(end.rotation.yaw), 3]

#     sol = solver(x0=u0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)

#     u = reshape(sol['x'].T, 2, N).T
#     ff_value = ff(u.T, p)
#     for k in range(N):
#         world.debug.draw_string(carla.Location(x=float(ff_value[0,k]), y=float(ff_value[1,k]), z=0.0),
#                                 'x', draw_shadow=False,
#                                 color=carla.Color(r=255, g=0, b=0), life_time=0.5,
#                                 persistent_lines=True)
#     ucl.append(u[0,:])
#     print(u)
#     print('a:', u[0,0], 'delta:', u[0,1])
#     quit()
#     vehicle.apply_control(carla.VehicleControl(throttle=float(u[0,0]), steer=float(u[0,1])))
#     ut = vehicle.get_transform().rotation.yaw
#     x0 = np.transpose([vehicle.get_transform().location.x, 
#                        vehicle.get_transform().location.y, convert_theta(ut), 
#                        norm_2([vehicle.get_velocity().x, vehicle.get_velocity().y])])
#     u0 = reshape(u0, N, 2)
#     u0[0:N-1,:] = u[1:N,:]
#     u0[N-1,:]=u[N-1,:]
#     i += 1
while c < len(route):
    if (norm_2(x0[0:2]-p[4:6]))<3:
        c += 1
        end = route[c].transform
        print('end:', end)
        world.debug.draw_string(end.location, 'O', draw_shadow=False,
                                                color=carla.Color(r=0, g=0, b=255), life_time=3,
                                                persistent_lines=True)
    u0 = reshape(u0, 2*N,1)
    p[0:4] = x0
    p[4:8] = [end.location.x, end.location.y, convert_theta(end.rotation.yaw), 3]#6
    
    sol = solver(x0=u0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)

    u = reshape(sol['x'].T, 2, N).T
    ff_value = ff(u.T, p)
    print(ff_value)
    time.sleep(1)
    for k in range(N):
        world.debug.draw_string(carla.Location(x=float(ff_value[0,k]), y=float(ff_value[1,k]),z=0.0), 'O', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=0.01,
                                    persistent_lines=True)
        time.sleep(0.5)
    ucl.append(u[0,:])

    vehicle.apply_control(carla.VehicleControl(throttle =float(u[0,0]) , steer = float(u[0,1])))
    # print(vehicle.get_transform().rotation.yaw)
    u_theta = vehicle.get_transform().rotation.yaw
    # if u_theta < 0:
    #     u_theta = 360 - abs(u_theta)
    x0 = np.transpose([vehicle.get_transform().location.x, vehicle.get_transform().location.y, convert_theta(u_theta), norm_2([vehicle.get_velocity().x, vehicle.get_velocity().y])])

    u0 = reshape(u0, N, 2)
    u0[0:N-1,:] = u[1:N,:]
    u0[N-1,:]=u[N-1,:]
    mpciter += 1


time.sleep(20)
for actor in actor_list:
    actor.destroy()
print("All cleaned up!")
















# lbx = DM.zeros((n_states*(N+1) + n_controls*N, 1))   #Lower bound on variables
# ubx = DM.zeros((n_states*(N+1) + n_controls*N, 1))

# lbx[0: n_states*(N+1): n_states] = -inf     # X lower bound
# ubx[0: n_states*(N+1): n_states] = inf     # X upper bound
# lbx[1: n_states*(N+1): n_states] = -inf     # Y lower bound
# ubx[1: n_states*(N+1): n_states] = inf     # Y upper bound
# lbx[2: n_states*(N+1): n_states] = -inf     # theta lower bound
# ubx[2: n_states*(N+1): n_states] = inf     # theta upper bound
# lbx[3: n_states*(N+1): n_states] = 0     # v lower bound
# ubx[3: n_states*(N+1): n_states] = 20    # v upper bound

# lbx[n_states*(N+1):] = th_min
# ubx[n_states*(N+1):] = th_max
# lbx[n_states*(N+1)+1::2] = delta_min
# ubx[n_states*(N+1)+1::2] = delta_max

# args = {
#     'lbg': DM.zeros((n_states*(N+1), 1)),
#     'ubg': DM.zeros((n_states*(N+1), 1)),
#     'lbx': lbx,
#     'ubx': ubx
# }

# t0 = 0
# for i in range(len(route) + 1):
#     start = route[i][0].transform.location
#     start_theta = route[i][0].tra_stringnsform.rotation.yaw
#     target = route[i+1][0].transform.location
#     target_theta = route[i+1][0].transform.rotation.yaw

#     world.debug.draw_string(target, 'o',
#                             draw_shadow=False,
#                             color=carla.Color(r=255, g=0, b=0),
#                             life_time=20.0,
#                             persistent_lines=True)

#     state_init = DM([start.x, start.y, start_theta, 0])
#     state_target = DM([target.x, target.y, target_theta, 0])
    
#     t = DM(t0)
#     u0 = DM.zeros((n_controls, N))
#     X0 = repmat(state_init, 1, N+1)
#     mpciter = 0
#     cat_states = dm_to_arr(X0)
#     cat_controls = dm_to_arr(u0[:,0])
#     times = np.array([[0]])
#     while (norm_2(state_init - state_target) > 1e-1) and (mpciter*T < sim_time):
#         t1 = time.time()
#         args['p'] = vertcat(state_init, state_target)
#         args['x0'] = vertcat(
#             reshape(X0, 4*(N+1), 1),
#             reshape(u0, 2*N, 1)
#         )

#         sol = solver(
#             x0=args['x0'],
#             lbx=args['lbx'],
#             ubx=args['ubx'],
#             lbg=args['lbg'],
#             ubg=args['ubg'],
#             p=args['p']
#         )

#         u = reshape(sol['x'][n_states*(N+1):], n_controls, N)
#         X0 = reshape(sol['x'][:n_states*(N+1)], n_states, N+1)

#         cat_states = np.dstack((cat_states, dm_to_arr(X0)))
#         cat_controls = np.dstack((cat_controls, dm_to_arr(u[:,0])))

#         t = np.vstack((t, t0))

#     mpciter += 1
#     vehicle.apply_control(carla.VehicleControl(throttle))

# def simulate(cat_states, cat_controls, t, dt, N, reference, save=False):
#     def inint():
#         return path, horizon, current_state, target_state
    
#     def animate(i):
#         # get variables
#         x = cat_states[0, 0, i]
#         y = cat_states[1, 0, i]
#         th = cat_states[2, 0, i]
#         v = cat_states[3, 0, i]
#         if i ==0:
#             path.set_data(np.array([]), np.array([]))
#         x_new = np.hstack((path.get_xdata(), x))
#         y_new = np.hstack((path.get_ydata(), y))
#         path.set_data(x_new, y_new)

#         #update horizon
#         x_new = cat_states[0, :, i]
#         y_new = cat_states[1, :, i]
#         horizon.set_data(x_new, y_new)

#         #update current state
#         current_state.set_xy((x, y))

#         return path, horizon, current_state, target_state

    






# t0 = 0
# state_init = initial_state(vehicle)
# state_target = DM([endpoint.location.x, endpoint.location.y, convert_theta(startpoint.rotation.yaw), 3])
# p =  np.transpose([startpoint.location.x, startpoint.location.y, convert_theta(startpoint.rotation.yaw), \
#                    0,endpoint.location.x, endpoint.location.y, convert_theta(startpoint.rotation.yaw), 3])

# u0 = DM.zeros(n_controls, N)
# X0 = repmat(state_init, 1, N+1)

# mpciter = 0

# c=0
# #while (norm_2(state_init - state_target) > 1e-1) and (mpciter * T < 20):
 
# while c < len(route):
#     if norm_2(x0[0:2]-p[4:8])<3:
#         c += 1
#         endpoint = route[c].transform
#         world.debug.draw_string(endpoint.location, '-',
#                                 draw_shadow=False,
#                                 color=carla.Color(0,0,255),
#                                 life_time=20.0,
#                                 persistent_lines=True)
    
    
#     p[0:4] = x0
#     p[4:8] = [endpoint.location.x, endpoint.location.y, convert_theta(endpoint.rotation.yaw), 3]
#     sol = solver(x0=u0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
#     u = reshape(sol['x'].T, 2, N).T
#     ff_value = ff(u.T, p)
#     for k in range(N):
#         world.debug.draw_string(carla.Location(x=float(ff_value[0,k]), y=float(ff_value[1,k]), z=0.5), '-',
#                                 draw_shadow=False,
#                                 color=carla.Color(255,0,0),
#                                 life_time=20.0,
#                                 persistent_lines=True)
        
#         vehicle.apply_control(carla.VehicleControl(throttle=float(u[0,0]), steer=float(u[0,1])))
#         u_theta = vehicle.get_transform().rotation.yaw
#         x0 = np.transpose([vehicle.get_transform().location.x,
#                            vehicle.get_transform().location.y,
#                            convert_theta(u_theta),
#                            norm_2([vehicle.get_velocity().x, vehicle.get_velocity().y])])
#         #u0 = reshape(u0, N, 2)
#         u0[0:N-1,:] = u[1:,:]
#         u0[N-1,:] = u[N-1,:]

#         mpciter += 1

# time.sleep(20)

# #Destroy actors
# for actor in actor_list:
#     actor.destroy()
# print('All cleaned up!')

