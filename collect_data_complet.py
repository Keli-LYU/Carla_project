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

# Try to find the carla library
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Image parameters
LARGEUR = 800
HAUTEUR = 600
DOSSIER_SAUVEGARDE = "./datasets/carla_data"

# Queues to store images
file_rgb = queue.Queue()
file_seg = queue.Queue()

def process_img_rgb(image):
    file_rgb.put(image)

def process_img_seg(image):
    file_seg.put(image)

def get_weather_presets():
    # Returns a list of interesting weather presets for diversity
    return [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.HardRainNoon,
        # carla.WeatherParameters.WetCloudyMidnight,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.ClearSunset
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', help='Server IP')
    parser.add_argument('-p', '--port', default=2000, type=int, help='TCP Port')
    parser.add_argument('-n', '--nb_images', default=2000, type=int, help='Total number of images to save')
    parser.add_argument('--maps', nargs='+', default=['Town03', 'Town10HD', 'Town12'], help='List of maps to use (e.g. Town01 Town02)')
    args = parser.parse_args()

    # Create directories
    if not os.path.exists(os.path.join(DOSSIER_SAUVEGARDE, "rgb")):
        os.makedirs(os.path.join(DOSSIER_SAUVEGARDE, "rgb"))
    if not os.path.exists(os.path.join(DOSSIER_SAUVEGARDE, "mask")):
        os.makedirs(os.path.join(DOSSIER_SAUVEGARDE, "mask"))

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0) # Longer timeout for loading maps (can take time)
        
        available_maps = client.get_available_maps()
        # Filter with what the user requested
        maps_to_use = [m for m in available_maps if any(user_map in m for user_map in args.maps)]
        
        if not maps_to_use:
            print("None of the requested maps are available on the server.")
            print("Available maps:", available_maps)
            return
            
        print(f"Selected maps: {maps_to_use}")
        
        weathers = get_weather_presets()
        
        # Calculate number of images per condition (map + weather)
        nb_conditions = len(maps_to_use) * len(weathers)
        images_par_condition = math.ceil(args.nb_images / nb_conditions)
        
        print(f"Total to generate: {args.nb_images} images.")
        print(f"Breakdown: {len(maps_to_use)} maps x {len(weathers)} weathers = {nb_conditions} different conditions.")
        print(f"Approximately {images_par_condition} images per condition.")

        images_sauvegardees_total = 0
        global_frame_index = 0 # Used for continuous image naming (000000.png, 000001.png...)

        # Security: check if the server is stuck in synchronous mode due to a previous crash
        try:
            temp_world = client.get_world()
            settings = temp_world.get_settings()
            if settings.synchronous_mode:
                settings.synchronous_mode = False
                temp_world.apply_settings(settings)
        except Exception:
            pass

        for map_name in maps_to_use:
            if images_sauvegardees_total >= args.nb_images:
                break
                
            print(f"\n--- Loading map: {map_name} ---")
            world = client.load_world(map_name)
            
            # Synchronous settings for this map
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

            tm = client.get_trafficmanager(8000)
            tm.set_synchronous_mode(True)
            # Safety distance for autopilot
            tm.set_global_distance_to_leading_vehicle(5.0) #add distance -Keli
            
            blueprint_library = world.get_blueprint_library()
            bp_vehicule = blueprint_library.filter('model3')[0]
            
            for weather_index, weather_params in enumerate(weathers):
                if images_sauvegardees_total >= args.nb_images:
                    break # Global quota reached
                    
                print(f" -> [{weather_index+1}/{len(weathers)}] Applying weather: {weather_params}")
                world.set_weather(weather_params)
                
                # --- Spawn vehicle for this session ---
                spawn_points = world.get_map().get_spawn_points()
                if not spawn_points:
                    print("Error: No spawn points found on this map.")
                    continue
                    
                spawn_point = random.choice(spawn_points)
                vehicule = world.spawn_actor(bp_vehicule, spawn_point)
                vehicule.set_autopilot(True, tm.get_port())
                
                liste_acteurs = [vehicule]
                
                # Clear potentially residual queues
                while not file_rgb.empty(): file_rgb.get()
                while not file_seg.empty(): file_seg.get()
                
                # --- Spawn sensors ---
                cam_bp = blueprint_library.find('sensor.camera.rgb')
                cam_bp.set_attribute('image_size_x', str(LARGEUR))
                cam_bp.set_attribute('image_size_y', str(HAUTEUR))
                cam_bp.set_attribute('fov', '90')
                cam_bp.set_attribute('sensor_tick', '0.0') # Capture at every tick for perfect sync
                
                transform = carla.Transform(carla.Location(x=1.5, z=2.4))
                cam_rgb = world.spawn_actor(cam_bp, transform, attach_to=vehicule)
                liste_acteurs.append(cam_rgb)
                cam_rgb.listen(process_img_rgb)

                seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
                seg_bp.set_attribute('image_size_x', str(LARGEUR))
                seg_bp.set_attribute('image_size_y', str(HAUTEUR))
                seg_bp.set_attribute('fov', '90')
                seg_bp.set_attribute('sensor_tick', '0.0') # Strict sync with simulation

                cam_seg = world.spawn_actor(seg_bp, transform, attach_to=vehicule)
                liste_acteurs.append(cam_seg)
                cam_seg.listen(process_img_seg)
                
                # Run the game for a few frames so the vehicle drops and initializes
                for _ in range(100):
                    world.tick()
                    
                # Start collection for this specific condition
                images_cette_condition = 0
                frames_skipped = 0
                
                while images_cette_condition < images_par_condition and images_sauvegardees_total < args.nb_images:
                    world.tick()
                    
                    try:
                        # Blocking wait for the frame exactly matching this tick
                        rgb_img = file_rgb.get(timeout=2.0)
                        seg_img = file_seg.get(timeout=2.0)
                    except queue.Empty:
                        continue
                        
                    # Save only 1 out of 20 images to space out captures (1 per simulated second)
                    frames_skipped += 1
                    if frames_skipped >= 20:
                        frames_skipped = 0
                        
                        # RGB Processing
                        array_rgb = np.frombuffer(rgb_img.raw_data, dtype=np.dtype("uint8"))
                        array_rgb = np.reshape(array_rgb, (rgb_img.height, rgb_img.width, 4))
                        array_rgb = array_rgb[:, :, :3]
                        chemin_rgb = os.path.join(DOSSIER_SAUVEGARDE, "rgb", f"{global_frame_index:06d}.png")
                        cv2.imwrite(chemin_rgb, array_rgb)

                        # Segmentation Processing
                        array_seg = np.frombuffer(seg_img.raw_data, dtype=np.dtype("uint8"))
                        array_seg = np.reshape(array_seg, (seg_img.height, seg_img.width, 4))
                        labels = array_seg[:, :, 2]
                        chemin_seg = os.path.join(DOSSIER_SAUVEGARDE, "mask", f"{global_frame_index:06d}.png")
                        cv2.imwrite(chemin_seg, labels)

                        images_cette_condition += 1
                        images_sauvegardees_total += 1
                        global_frame_index += 1
                        
                        if images_sauvegardees_total % 20 == 0:
                            print(f"   Global progress: {images_sauvegardees_total}/{args.nb_images}")

                # End of condition: destroy actors before moving to the next one
                for acteur in reversed(liste_acteurs):
                    if acteur.is_alive:
                        if hasattr(acteur, 'stop'):
                            acteur.stop() # Stop sensor listening
                        acteur.destroy()
            
            # Restore asynchronous mode before changing map to avoid server freezing
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            
            tm.set_synchronous_mode(False)


        print("\n=== Data collection completed successfully! ===")
        print(f"Total: {images_sauvegardees_total} images saved.")

    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        print("End of script.")

if __name__ == '__main__':
    main()
