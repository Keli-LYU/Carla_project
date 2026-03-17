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

# On essaie de trouver la lib carla
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Parametres de l'image
LARGEUR = 800
HAUTEUR = 600
DOSSIER_SAUVEGARDE = "./datasets/carla_data"

# Files pour stocker les images
file_rgb = queue.Queue()
file_seg = queue.Queue()

def process_img_rgb(image):
    file_rgb.put(image)

def process_img_seg(image):
    file_seg.put(image)

def get_weather_presets():
    # Retourne une liste de preset météo intéressants pour la diversité
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
    parser.add_argument('--host', default='127.0.0.1', help='IP du serveur')
    parser.add_argument('-p', '--port', default=2000, type=int, help='Port TCP')
    parser.add_argument('-n', '--nb_images', default=2000, type=int, help='Nombre total d\'images a sauvegarder')
    parser.add_argument('--maps', nargs='+', default=['Town03', 'Town10HD', 'Town12'], help='Liste des maps a utiliser (ex: Town01 Town02)')
    args = parser.parse_args()

    # Creation des dossiers
    if not os.path.exists(os.path.join(DOSSIER_SAUVEGARDE, "rgb")):
        os.makedirs(os.path.join(DOSSIER_SAUVEGARDE, "rgb"))
    if not os.path.exists(os.path.join(DOSSIER_SAUVEGARDE, "mask")):
        os.makedirs(os.path.join(DOSSIER_SAUVEGARDE, "mask"))

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0) # Un peu plus long pour le chargement des maps (peut prendre du temps)
        
        available_maps = client.get_available_maps()
        # Filtrer avec ce que l'utilisateur a demandé
        maps_to_use = [m for m in available_maps if any(user_map in m for user_map in args.maps)]
        
        if not maps_to_use:
            print("Aucune des maps demandées n'est disponible sur le serveur.")
            print("Maps disponibles :", available_maps)
            return
            
        print(f"Maps sélectionnées : {maps_to_use}")
        
        weathers = get_weather_presets()
        
        # Calcul du nombre d'images par condition (map + météo)
        nb_conditions = len(maps_to_use) * len(weathers)
        images_par_condition = math.ceil(args.nb_images / nb_conditions)
        
        print(f"Total à générer : {args.nb_images} images.")
        print(f"Répartition : {len(maps_to_use)} maps x {len(weathers)} météos = {nb_conditions} conditions différentes.")
        print(f"Soit environ {images_par_condition} images par condition.")

        images_sauvegardees_total = 0
        global_frame_index = 0 # Utilisé pour nommer les images continuellement (000000.png, 000001.png...)

        for map_name in maps_to_use:
            if images_sauvegardees_total >= args.nb_images:
                break
                
            print(f"\n--- Chargement de la map : {map_name} ---")
            world = client.load_world(map_name)
            
            # Paramétrage synchrone pour cette map
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

            tm = client.get_trafficmanager(8000)
            tm.set_synchronous_mode(True)
            # Distance de sécurité pour le pilote auto
            tm.set_global_distance_to_leading_vehicle(2.0)
            
            blueprint_library = world.get_blueprint_library()
            bp_vehicule = blueprint_library.filter('model3')[0]
            
            for weather_index, weather_params in enumerate(weathers):
                if images_sauvegardees_total >= args.nb_images:
                    break # On a atteint le quota global
                    
                print(f" -> [{weather_index+1}/{len(weathers)}] Application de la météo : {weather_params}")
                world.set_weather(weather_params)
                
                # --- Spawn du véhicule pour cette session ---
                spawn_points = world.get_map().get_spawn_points()
                if not spawn_points:
                    print("Erreur: Aucun point de spawn trouvé sur cette map.")
                    continue
                    
                spawn_point = random.choice(spawn_points)
                vehicule = world.spawn_actor(bp_vehicule, spawn_point)
                vehicule.set_autopilot(True)
                
                liste_acteurs = [vehicule]
                
                # Vider les files d'attente potentiellement résiduelles
                while not file_rgb.empty(): file_rgb.get()
                while not file_seg.empty(): file_seg.get()
                
                # --- Spawn des capteurs ---
                cam_bp = blueprint_library.find('sensor.camera.rgb')
                cam_bp.set_attribute('image_size_x', str(LARGEUR))
                cam_bp.set_attribute('image_size_y', str(HAUTEUR))
                cam_bp.set_attribute('fov', '90')
                cam_bp.set_attribute('sensor_tick', '0.0') # On capture à chaque tick pour assurer la synchro parfaite
                
                transform = carla.Transform(carla.Location(x=1.5, z=2.4))
                cam_rgb = world.spawn_actor(cam_bp, transform, attach_to=vehicule)
                liste_acteurs.append(cam_rgb)
                cam_rgb.listen(process_img_rgb)

                seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
                seg_bp.set_attribute('image_size_x', str(LARGEUR))
                seg_bp.set_attribute('image_size_y', str(HAUTEUR))
                seg_bp.set_attribute('fov', '90')
                seg_bp.set_attribute('sensor_tick', '0.0') # Synchro stricte avec la simulation

                cam_seg = world.spawn_actor(seg_bp, transform, attach_to=vehicule)
                liste_acteurs.append(cam_seg)
                cam_seg.listen(process_img_seg)
                
                # Mettre le jeu en route pour quelques frames afin que le véhicule tombe au sol et s'initialise
                for _ in range(40):
                    world.tick()
                    
                # Début de la collecte pour cette condition précise
                images_cette_condition = 0
                frames_skipped = 0
                
                while images_cette_condition < images_par_condition and images_sauvegardees_total < args.nb_images:
                    world.tick()
                    
                    try:
                        # Attente bloquante de la frame correspondant exactement à ce tick
                        rgb_img = file_rgb.get(timeout=2.0)
                        seg_img = file_seg.get(timeout=2.0)
                    except queue.Empty:
                        continue
                        
                    # On ne sauvegarde qu'une image sur 20 pour espacer les prises de vue (1 par seconde simulée)
                    frames_skipped += 1
                    if frames_skipped >= 20:
                        frames_skipped = 0
                        
                        # Traitement RGB
                        array_rgb = np.frombuffer(rgb_img.raw_data, dtype=np.dtype("uint8"))
                        array_rgb = np.reshape(array_rgb, (rgb_img.height, rgb_img.width, 4))
                        array_rgb = array_rgb[:, :, :3]
                        chemin_rgb = os.path.join(DOSSIER_SAUVEGARDE, "rgb", f"{global_frame_index:06d}.png")
                        cv2.imwrite(chemin_rgb, array_rgb)

                        # Traitement Segmentation
                        array_seg = np.frombuffer(seg_img.raw_data, dtype=np.dtype("uint8"))
                        array_seg = np.reshape(array_seg, (seg_img.height, seg_img.width, 4))
                        labels = array_seg[:, :, 2]
                        chemin_seg = os.path.join(DOSSIER_SAUVEGARDE, "mask", f"{global_frame_index:06d}.png")
                        cv2.imwrite(chemin_seg, labels)

                        images_cette_condition += 1
                        images_sauvegardees_total += 1
                        global_frame_index += 1
                        
                        if images_sauvegardees_total % 20 == 0:
                            print(f"   Progression globale : {images_sauvegardees_total}/{args.nb_images}")

                # Fin de la condition : on détruit les acteurs avant de passer à la condition suivante
                for acteur in reversed(liste_acteurs):
                    if acteur.is_alive:
                        if hasattr(acteur, 'stop'):
                            acteur.stop() # On arrête l'écoute des capteurs
                        acteur.destroy()


        print("\n=== Collecte de données terminée avec succès ! ===")
        print(f"Total : {images_sauvegardees_total} images sauvegardées.")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        
    finally:
        print("Fin du script.")

if __name__ == '__main__':
    main()
