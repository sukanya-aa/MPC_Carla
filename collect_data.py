from concurrent.futures import process
import glob
import os
import sys
import time
import numpy as np
import cv2
import threading
import queue
import math
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

actor_list = []


def process_image(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    i3 = i2[:, :, :3]
    image.save_to_disk(f"C:\CARLA_0.9.8\WindowsNoEditor\PythonAPI\out_data\{image.frame}.png")

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    #settings.synchronous_mode = True
    world.apply_settings(settings)

    spawnpoints = world.get_map().get_spawn_points()
    #startpoint = spawnpoints[0]
    startpoint = random. choice(spawnpoints)

    vehicle_blueprint= world.get_blueprint_library().find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_blueprint, startpoint)
    actor_list.append(vehicle)

    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    actor_list.append(cam)  

    vehicle.set_autopilot(True)
    cam.listen(lambda data: process_image(data))
    print("Collecting images....")

    time.sleep(10)

finally:
    print('destroying all actors')
    exit = True
    for actor in actor_list:
        actor.destroy()
    print('done')