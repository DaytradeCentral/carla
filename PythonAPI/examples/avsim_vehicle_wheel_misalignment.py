#!/usr/bin/env python


import glob
import os
import sys
import argparse
import time
from pxr import Gf

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


rgb_image = None
wheel_screen_translations = None
on_tick_callback_id = None

WINDOW_WIDTH = 896
WINDOW_HEIGHT = 504
CAM_FOV = 45


class CarlaUser(object):
    def __init__(self, world, fps, sync):
        self.world = world
        self.delta_seconds = 1.0 / fps
        self._settings = None
        self._sync = sync

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=self._sync,
            fixed_delta_seconds=self.delta_seconds))
        return self

    def tick(self, timeout):
        if self._sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)


def draw_image(surface, image, blend=False, offset=(0,0)):
    if not (image is None):
        image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))     
        if image_surface.get_height() != WINDOW_HEIGHT or image_surface.get_width() != WINDOW_WIDTH:
            # scale image
            image_surface = pygame.transform.scale(image_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, offset)


def convert_carla_matrix_to_Gf(mtx: list[list[float]]):
    m = [v for l in mtx for v in l]
    return Gf.Matrix4f(
        m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14], m[3], m[7], m[11], m[15]
    )


def carla_to_usd_coord(matrix: Gf.Matrix4f):
    # Extract rotation angles and translation, this is the world matrix in Unreal
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotation().Decompose(
        Gf.Vec3d.ZAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.XAxis()
    )

    # Convert the Unreal matrix to USD, flip translation in Y-axis, and roll, yaw values
    usd_mtx = Gf.Matrix4f().SetRotate(
        Gf.Rotation(Gf.Vec3d.XAxis(), -rotation[2])
        * Gf.Rotation(Gf.Vec3d.YAxis(), rotation[1])
        * Gf.Rotation(Gf.Vec3d.ZAxis(), -rotation[0])
    )
    usd_mtx *= Gf.Matrix4f().SetTranslate(
        Gf.Vec3f(translation[0], -translation[1], translation[2])
    )
    return usd_mtx


def get_wheel_screen_translations(vehicle, sensor):
    if not vehicle.is_alive or not sensor.is_alive:
        return

    # View matrix, UE camera is +X forward, +Z upward
    cam_transform = sensor.get_transform()
    cam_matrix = cam_transform.get_matrix()
    cam_matrix: Gf.Matrix4f = convert_carla_matrix_to_Gf(cam_matrix)

    sensor_corrected_mtx = carla_to_usd_coord(cam_matrix)

    # Camera is +X forward, +Z up in Unreal and -Z forward and +Y up in USD
    cam_correction_mtx = Gf.Matrix4f().SetRotate(
        Gf.Rotation(Gf.Vec3d.XAxis(), 90) * Gf.Rotation(Gf.Vec3d.ZAxis(), -90)
    )
    sensor_corrected_mtx = cam_correction_mtx * sensor_corrected_mtx
    view_mtx = sensor_corrected_mtx.GetInverse()

    # Projection matrix
    fov_horizontal = np.radians(CAM_FOV)
    aspect = WINDOW_WIDTH / WINDOW_HEIGHT
    fov_vertical = 2 * np.arctan(np.tan(fov_horizontal / 2) * (WINDOW_HEIGHT/WINDOW_WIDTH))
    e = 1 / np.tan(fov_vertical/2)
    
    far = 200
    near = 0.1
    proj_mtx = Gf.Matrix4f(
        e/aspect, 0, 0, 0,
        0, e, 0, 0,
        0, 0, (far+near)/(near-far), -1,
        0, 0, (2*far*near)/(near-far), 0
    )

    # Visualize wheel location
    bone_names = vehicle.get_bone_names()
    bone_transforms = vehicle.get_bone_world_transforms()

    wheel_screen_translations = []

    for bone_name, bone_transform in zip(bone_names, bone_transforms):
        if bone_name.startswith("Wheel_") and bone_name.endswith("Left"):
            wheel_mtx = convert_carla_matrix_to_Gf(bone_transform.get_matrix())
            wheel_mtx = carla_to_usd_coord(wheel_mtx)

            wheel_ndc_mtx = wheel_mtx * view_mtx * proj_mtx
            clip_coord = wheel_ndc_mtx.GetRow(3)
            ndc_coord = Gf.Vec3f(clip_coord[0]/clip_coord[3], clip_coord[1]/clip_coord[3], clip_coord[2]/clip_coord[3])

            x_screen = (ndc_coord[0] + 1) * 0.5 * WINDOW_WIDTH
            y_screen = (1 - ndc_coord[1]) * 0.5 * WINDOW_HEIGHT

            wheel_screen_translations.append((x_screen, y_screen))

    return wheel_screen_translations


def display_content(display):
    global rgb_image
    
    display.fill((0, 0, 0))
    
    draw_image(display, rgb_image, False, (0, 0))
    
    # Visualize wheels
    surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    if wheel_screen_translations is not None:
        for wheel_screen_t in wheel_screen_translations:
            pygame.draw.circle(surface, (0, 0, 255, 140), wheel_screen_t, 50)

    display.blit(surface, (0, 0))

    pygame.display.flip()


def handle_key_event():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return pygame.QUIT
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return pygame.QUIT
            elif event.key == pygame.K_r:
                return pygame.K_r
            elif event.key == pygame.K_s:
                return pygame.K_s
            elif event.key == pygame.K_d:
                return pygame.K_d
            elif event.key == pygame.K_o:
                return pygame.K_o
    return 0


def save_rgb_image(image):
    global rgb_image
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    rgb_image = array


def start_pygame(width, height):
    pygame.init()
    display = pygame.display.set_mode(
        (width, height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    return display, clock


def spawn_actor(world, id, start_pose, actor_list, attach_to=None, attributes={}):
    blueprint_library = world.get_blueprint_library()
    actor_bp = blueprint_library.find(id)
    for attr_name, value in attributes.items():
        actor_bp.set_attribute(attr_name, value)

    actor = world.spawn_actor(actor_bp, start_pose, attach_to)
    actor_list.append(actor)
    return actor


def on_tick_callback(world_snapshot, vehicle, camera):
    time.sleep(0.2)  # Comment out this line to compare the wheel transforms
    global wheel_screen_translations
    wheel_screen_translations = get_wheel_screen_translations(vehicle, camera)


def main():
    argparser = argparse.ArgumentParser(
        description='AVSim nv_rtx Sensor tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-f', '--maxframe',
        default=50,
        type=int,
        help='Max Frames to run (default: 50)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')

    argparser.set_defaults(sync=True)

    args = argparser.parse_args()

    global rgb_image
    actor_list = []

    client = carla.Client(args.host, args.port)
    # need a big timeout for avsim_client_library
    client.set_timeout(100.0)

    world = client.get_world()

    map_name = "Town10HD_Opt"

    print(f"sync={args.sync} map={map_name}")
    world = client.load_world(map_name)

    try:
        start_transform = world.get_map().get_spawn_points()[10]

        # start pygame for sensors to use
        display, clock = start_pygame(WINDOW_WIDTH, WINDOW_HEIGHT)

        cam_xform = carla.Transform(carla.Location(x=0, y=-8, z=1), carla.Rotation(pitch=0, yaw=90))

        camera_attributes = {
            "image_size_x": str(WINDOW_WIDTH),
            "image_size_y": str(WINDOW_HEIGHT),
            "fov": str(CAM_FOV)
        }

        # Create a synchronous mode context.
        frame = 0

        with CarlaUser(world, fps=30, sync=args.sync) as sync_mode:
            # Spawn a vehicle
            vehicle_ctrl = carla.VehicleControl()
            vehicle = spawn_actor(world, "vehicle.dodge.charger_2020", start_transform, actor_list)

            # Spawn a sensor
            camera_rgb = spawn_actor(
                world,
                'sensor.camera.rgb',
                cam_xform,
                actor_list,
                attach_to=vehicle,
                attributes=camera_attributes
            )
            camera_rgb.listen(save_rgb_image)

            on_tick_callback_id = sync_mode.world.on_tick(lambda snapshot, v=vehicle, c=camera_rgb: on_tick_callback(snapshot, v, c))

            while True:
                key = handle_key_event()
                if key == pygame.QUIT or frame > args.maxframe:
                    break

                clock.tick()
                # Advance the simulation and wait for the data.
                sync_mode.tick(timeout=2.0)

                frame += 1

                # Apply vehicle control
                vehicle_ctrl.throttle = 0.6
                vehicle.apply_control(vehicle_ctrl)

                display_content(display)

    finally:
        if on_tick_callback_id is not None:
            world.remove_on_tick(on_tick_callback_id)

        for actor in actor_list:
            actor.destroy()

        # Always disable sync mode before the script ends to prevent the server blocking whilst waiting for a tick
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        
        pygame.quit()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
