from hilo_mpc import NMPC, Model, set_plot_backend, SimpleControlLoop
from casadi import *
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, gridplot

def traj(z):
    return np.sin(z), np.sin(2*z)
    
set_plot_backend('bokeh')


model = Model(name='MPC trajectory', plot_backend='bokeh')
X = model.set_dynamical_states(['x', 'y', 'theta', 'v'])
model.set_measurements(['xi', 'yi', 'thetai', 'vi'])
model.set_measurement_equations([X[0], X[1], X[2], X[3]])
x = X[0]
y = X[1]
theta = X[2]
v = X[3]

U = model.set_inputs(['a', 'delta'])
a = U[0]
delta = U[1]

lr = 1.5
lf = 1.5
beta = arctan(lr / (lr + lf) * tan(delta))
#w = v * tan(delta) / (lr + lf)
dx = v * cos(theta + beta)
dy = v * sin(theta + beta)
dyaw = v * cos(beta) * tan(delta) / (lr + lf)
dv = a

model.set_dynamical_equations([dx, dy, dyaw, dv])

x0 = [0, 0, 0, 0]
u0 = [0, 0]

dt = 0.1
model.setup(dt=dt, options={'objective_function': 'discrete'})

nmpc = NMPC(model)
t = nmpc.get_time_variable()

nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10,10],
                     ref=vertcat(sin(t), sin(2*t)),
                     trajectory_tracking=True)
nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10,10],
                        ref=vertcat(sin(t), sin(2*t)),
                        trajectory_tracking=True)

# nmpc.quad_stage_cost.add_states(names='v', ref=1, weights=10)
# nmpc.quad_terminal_cost.add_states(names='v', ref=1, weights=10)
# nmpc.set_box_constraints(x_lb=[-inf, -inf, -inf, 0], x_ub =[inf, inf, inf, 1])
nmpc.horizon = 10
nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
nmpc.setup(options={'objective_function': 'discrete', 'print_level': 0})
model.set_initial_conditions(x0=x0)

ss = SimpleControlLoop(model, nmpc)
ss.run(100)
ss.plot()


'''
for step in range(200):
    u = nmpc.optimize(x0)
    model.simulate(u=u)
    x0 = model.solution['x:f']

plots
xt = []
yt = []
for i in range(1000):
    x, y = traj(i/100)
    xt.append(x)
    yt.append(y)
p1 = figure(title='Lissajous')
p1.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze(), color='#189ad3',
          line_width=3, line_dash='dashed', legend_label='tracked')
p1.line(x=xt, y=yt, line_dash='dashed', line_width=3, color='#fc913a', legend_label='trajectory')

show(p1)

n_steps = 100
for step in range(n_steps):
    u = nmpc.optimize(x0)
    model.simulate(u=u)
    x0 = model.solution['x:f']

xt = []
yt = []
for i in range(1000):
    x, y = traj(i/100)
    xt.append(x)
    yt.append(y)

p1 = figure(title='Trajectory following')
p1.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
p1.line(x=xt, y=yt, color='red', line_dash='dashed')
p1.xaxis.axis_label = 'x'
p1.yaxis.axis_label = 'y'
show(p1)
pt =[]
for state in model.dynamical_state_names:
    p = figure(background_fill_color="#fafafa", width=400, height=400)
    p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution[state]).squeeze(), 
           legend_label=state, line_width=2)
    p.line(x=xt, y=yt, color='red', line_dash='dashed')
    for i in range(len(nmpc.quad_stage_cost._references_list)):
        if state in nmpc.quad_stage_cost._references_list[i]['names']:
            position = nmpc.quad_stage_cost._references_list[i]['names'].index(state)
            value = nmpc.quad_stage_cost._references_list[i]['ref'][position]
            p.line([np.array(model.solution['t'][1]).squeeze(), np.array(model.solution['t'][-1]).squeeze()],
                   [value, value], legend_label=state + '_ref',
                   line_dash='dashed', line_color="red", line_width=2)
    p.xaxis.axis_label = 't'
    p.yaxis.axis_label = state
    pt.append(p)

show(gridplot(pt, ncols=2))
'''