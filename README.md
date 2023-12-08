# What is it?

This project is a low-level implementation of Nonlinear Model Predictive Control in autonomous cars in a simulated environment provided by [CARLA](http://carla.org/). The car is modelled to track a set of trajectory reference points on CARLA's maps. Additionally, the algorithm, as such can be used to track any trajectory that can be expressed as a function of time. This project demonstrates trajectory tracking the following curves: 
* Circular trajectory
* Lissajous curve
* Reference points trajectory.

# Abstract

The estimated number of deaths due to road crashes in the year 2019 happens to be 36,096 in the United States. Autonomous Vehicles (AV) have the potential to revolutionize 
transportation by reducing accidents and easing the traffic flow. An autonomous vehicle system comprises three units - **Perception, Planning** and **Control**. 

* Perception involves the identification of the vehicle’s state and the environment using sensors. 
* Planning refers to the process of generating a trajectory or a path to a target destination based on the perceived environment. 
* The control system is responsible for executing the planned trajectory by adjusting the throttle, 
brake and steering of the vehicle. 

Developing autonomous vehicle systems requires robust training data and advanced algorithms for perception and control. However, such training data 
are expensive and risky to create, label, and process, owing to on-road safety concerns, and time-consuming work. The proposed model is implemented on a Software-in-the-loop (SIL) 
simulation, where a software model of the car and its control system is present to simulate different driving scenarios. This project focuses on the control system of the autonomous 
vehicle system. 

Nonlinear Model Predictive Control (NMPC) is a powerful control technique that estimates the vehicle's trajectory, optimizing a cost function over a finite time horizon. It uses 
a predictive model to forecast future behaviour and find the control inputs that minimize the cost function- reducing the time to reach the destination while satisfying safety constraints. 
This process is repeated to generate new predictions and control sets for every time step. The project aims to develop a vehicle dynamics model that describes its motion and implements 
the MPC controller by designing the control parameters in CARLA Simulator. Further, the validation results can be used to refine the controller by tuning the parameters, modifying the 
constraints, or improving the dynamics model.




