import glob
import os
import sys
import time
import random
import queue
import numpy as np
import cv2
import argparse
import math

# Try to find the CARLA library
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Image parameters
WIDTH = 800
HEIGHT = 600
BASE_SAVE_DIR = "./datasets/carla_data"

# Queues for storing images
rgb_queue = queue.Queue()
seg_queue = queue.Queue()

def process_rgb_image(image):
    rgb_queue.put(image)

def process_seg_image(image):
    seg_queue.put(image)

def get_weather_presets():
    # Return a list of weather presets for data diversity
    return [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.ClearSunset
    ]

def spawn_static_actors(world, blueprint_library, num_vehicles=3, num_walkers=6):
    """Spawn a small set of static vehicles and pedestrians for scene diversity."""
    static_actors = []

    # Static vehicles
    vehicle_bps = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    spawned_vehicles = 0
    for spawn_point in spawn_points:
        if spawned_vehicles >= num_vehicles:
            break
        if not vehicle_bps:
            break

        vehicle_bp = random.choice(vehicle_bps)
        if vehicle_bp.has_attribute('role_name'):
            vehicle_bp.set_attribute('role_name', 'static')

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            continue

        try:
            vehicle.set_autopilot(False)
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            vehicle.set_simulate_physics(False)
        except Exception:
            pass

        static_actors.append(vehicle)
        spawned_vehicles += 1

    # Static pedestrians
    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    spawned_walkers = 0

    for _ in range(num_walkers):
        if not walker_bps:
            break

        nav_location = world.get_random_location_from_navigation()
        if nav_location is None:
            continue

        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')

        walker = world.try_spawn_actor(walker_bp, carla.Transform(nav_location))
        if walker is None:
            continue

        try:
            walker.set_simulate_physics(False)
        except Exception:
            pass

        static_actors.append(walker)
        spawned_walkers += 1

    print(f"Spawned static actors: {spawned_vehicles} vehicles, {spawned_walkers} pedestrians")
    return static_actors

def main():
    parser = argparse.ArgumentParser(description="CARLA Data Collection Script (Single Map)")
    parser.add_argument('--host', default='127.0.0.1', help='Server IP')
    parser.add_argument('-p', '--port', default=2000, type=int, help='TCP Port')
    parser.add_argument('-n', '--nb_images', default=1000, type=int, help='Total images to save for this map')
    parser.add_argument('--map', default='Town03', type=str, help='Target map to collect data from (e.g., Town03)')
    args = parser.parse_args()

    # Create directories based on the map name
    map_save_dir = os.path.join(BASE_SAVE_DIR, args.map)
    rgb_dir = os.path.join(map_save_dir, "rgb")
    mask_dir = os.path.join(map_save_dir, "mask")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"=== Starting data collection for {args.map} ===")
    print(f"Data will be saved to: {map_save_dir}")

    world = None
    tm = None
    static_actor_list = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0) 
        
        # Check if requested map exists
        available_maps = client.get_available_maps()
        if not any(args.map in m for m in available_maps):
            print(f"Error: Map '{args.map}' is not available on the server.")
            return

        print(f"Loading map: {args.map}...")
        world = client.load_world(args.map)
        
        # Setup synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Setup Traffic Manager
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.0)
        
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]

        #障碍参数Parameters for static actors
        static_actor_list = spawn_static_actors(world, blueprint_library, num_vehicles=15, num_walkers=10)
        
        weathers = get_weather_presets()
        images_per_weather = math.ceil(args.nb_images / len(weathers))
        
        print(f"Target: {args.nb_images} images.")
        print(f"Distribution: {len(weathers)} weathers, ~{images_per_weather} images per weather.")

        total_saved_images = 0
        global_frame_index = 0 

        for weather_idx, weather_params in enumerate(weathers):
            if total_saved_images >= args.nb_images:
                break 
                
            print(f"\n[{weather_idx+1}/{len(weathers)}] Applying weather: {weather_params}")
            world.set_weather(weather_params)
            
            images_current_weather = 0
            
            # Keep trying until we gather enough images for this weather
            while images_current_weather < images_per_weather and total_saved_images < args.nb_images:
                
                spawn_points = world.get_map().get_spawn_points()
                if not spawn_points:
                    print("Error: No spawn points found.")
                    break
                    
                spawn_point = random.choice(spawn_points)
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                
                if vehicle is None:
                    continue # Failed to spawn (blocked), try another point
                    
                vehicle.set_autopilot(True)
                actor_list = [vehicle]
                
                # Clear residual queues
                while not rgb_queue.empty(): rgb_queue.get()
                while not seg_queue.empty(): seg_queue.get()
                
                # Spawn RGB Camera
                cam_bp = blueprint_library.find('sensor.camera.rgb')
                cam_bp.set_attribute('image_size_x', str(WIDTH))
                cam_bp.set_attribute('image_size_y', str(HEIGHT))
                cam_bp.set_attribute('fov', '90')
                cam_bp.set_attribute('sensor_tick', '0.0') 
                
                transform = carla.Transform(carla.Location(x=1.5, z=2.4))
                rgb_cam = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
                actor_list.append(rgb_cam)
                rgb_cam.listen(process_rgb_image)

                # Spawn Segmentation Camera
                seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
                seg_bp.set_attribute('image_size_x', str(WIDTH))
                seg_bp.set_attribute('image_size_y', str(HEIGHT))
                seg_bp.set_attribute('fov', '90')
                seg_bp.set_attribute('sensor_tick', '0.0')

                seg_cam = world.spawn_actor(seg_bp, transform, attach_to=vehicle)
                actor_list.append(seg_cam)
                seg_cam.listen(process_seg_image)
                
                # Spawn Collision Sensor
                col_bp = blueprint_library.find('sensor.other.collision')
                col_cam = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
                actor_list.append(col_cam)
                
                has_collided = [False] 
                
                def on_collision(event):
                    has_collided[0] = True
                    print(f"\n[!] Collision detected with {event.other_actor.type_id}. Respawning...")
                
                col_cam.listen(on_collision)
                
                # Warm-up phase (Drop to ground and check bad spawn)
                bad_spawn = False
                for _ in range(40):
                    world.tick()
                    if has_collided[0]:
                        bad_spawn = True
                        break
                        
                if bad_spawn:
                    print("--> Invalid spawn point (immediate collision). Discarding...")
                    for actor in reversed(actor_list):
                        if actor.is_alive:
                            if hasattr(actor, 'stop'): actor.stop()
                            actor.destroy()
                    continue # Go to the next iteration of the while loop (respawn)
                    
                # Collection phase
                frames_skipped = 0
                
                while images_current_weather < images_per_weather and total_saved_images < args.nb_images:
                    world.tick()
                    
                    if has_collided[0]:
                        break # Exit the inner loop to respawn
                    
                    try:
                        rgb_img = rgb_queue.get(timeout=2.0)
                        seg_img = seg_queue.get(timeout=2.0)
                    except queue.Empty:
                        continue
                        
                    # Save 1 image every 20 frames
                    frames_skipped += 1
                    if frames_skipped >= 20:
                        frames_skipped = 0
                        
                        # Process RGB
                        array_rgb = np.frombuffer(rgb_img.raw_data, dtype=np.dtype("uint8"))
                        array_rgb = np.reshape(array_rgb, (rgb_img.height, rgb_img.width, 4))
                        array_rgb = array_rgb[:, :, :3]
                        rgb_path = os.path.join(rgb_dir, f"{global_frame_index:06d}.png")
                        cv2.imwrite(rgb_path, array_rgb)

                        # Process Segmentation
                        array_seg = np.frombuffer(seg_img.raw_data, dtype=np.dtype("uint8"))
                        array_seg = np.reshape(array_seg, (seg_img.height, seg_img.width, 4))
                        labels = array_seg[:, :, 2]
                        seg_path = os.path.join(mask_dir, f"{global_frame_index:06d}.png")
                        cv2.imwrite(seg_path, labels)

                        images_current_weather += 1
                        total_saved_images += 1
                        global_frame_index += 1
                        
                        if total_saved_images % 20 == 0:
                            print(f"   Progress: {total_saved_images}/{args.nb_images}")

                # Destroy actors before spawning a new one or changing weather
                for actor in reversed(actor_list):
                    if actor.is_alive:
                        if hasattr(actor, 'stop'):
                            actor.stop() 
                        actor.destroy()

        print("\n=== Data collection finished successfully! ===")
        print(f"Total images saved: {total_saved_images}")

    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        for actor in reversed(static_actor_list):
            if actor.is_alive:
                actor.destroy()

        if tm is not None:
            tm.set_synchronous_mode(False)

        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print("Script terminated.")

if __name__ == '__main__':
    main()
