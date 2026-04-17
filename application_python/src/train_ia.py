from ultralytics import YOLO
import os
import glob
import numpy as np

IMGSZ = 1024

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================
CHEMIN_DATA_YAML = '../../dataset_autocollant/data.yaml' 
CHEMIN_DOSSIER_TEST = '../../dataset_autocollant/test/images'

if __name__ == '__main__':
    print("[IA] Chargement du modèle YOLOv8-Pose (Nano)...")
    # On charge le tout petit modèle de base "nano"
    model = YOLO('yolov8n-pose.pt') 

    print("[IA] Lancement de l'entraînement avec le dataset Roboflow...")
    results = model.train(
        data=CHEMIN_DATA_YAML, 
        epochs=50,
        imgsz=IMGSZ,
        batch=6,
        device='cpu',
        
        # Vu que Roboflow a potentiellement déjà fait de la Data Augmentation (génération), 
        # on désactive celles de YOLO pour éviter de "casser" les points clés
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        degrees=0.0,
        
        # Pénalité de l'absence de détection (box) et de points clés (pose)
        pose=35.0,
        box=10.0
    )
    
    # ==========================================
    # LE TABLEAU DE BORD DES RÉSULTATS
    # ==========================================
    print("\n" + "="*60)
    print("Résumé des performances du modèle :")
    print("="*60)
    
    chemin_best_pt = f"{results.save_dir}/weights/best.pt"
    print(f"Dossier de sauvegarde : {results.save_dir}")
    print(f"Fichier 'Cerveau' à utiliser : {chemin_best_pt}\n")

    box_map50 = results.box.map50 * 100   
    box_map95 = results.box.map * 100     
    pose_map50 = results.pose.map50 * 100
    pose_map95 = results.pose.map * 100

    print("=== Détection Globale (Boîte) ===")
    print(f"Score Global (Tolérance 50%)             : {box_map50:.1f} %")
    print(f"Score Strict (Tolérance 50-95%)          : {box_map95:.1f} %\n")

    print("=== Détection des 4 Coins (Pose) ===")
    print(f"Score Global (Tolérance 50%)             : {pose_map50:.1f} %")
    print(f"Score Strict (Tolérance 50-95%)          : {pose_map95:.1f} %\n")
    print("="*60 + "\n")

    # ==========================================
    # REALITY CHECK
    # ==========================================
    print("\n" + "="*60)
    print("Lancement du Reality Check...")
    print("="*60 + "\n")

    best_model = YOLO(chemin_best_pt)

    # Récupération automatique de la première image du dossier test de Roboflow
    images_test_disponibles = glob.glob(f"{CHEMIN_DOSSIER_TEST}/*.jpg")
    
    if len(images_test_disponibles) == 0:
        print(f"Erreur : Aucune image .jpg trouvée dans le dossier test ({CHEMIN_DOSSIER_TEST}).")
    else:
        image_test = images_test_disponibles[0]
        print(f"[Test] Analyse de l'image issue du dataset de test : {image_test}")
        
        predictions = best_model.predict(
            source=image_test, 
            conf=0.5,
            show=False,
            save=True
        )

        resultat_ia = predictions[0]

        if len(resultat_ia.boxes) > 0:
            print("\nAutocollant détecté ! Extraction des points...")
            
            boite = resultat_ia.boxes.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = boite
            
            points_cles = resultat_ia.keypoints.data[0].cpu().numpy() 
            confiance_moyenne = np.mean(points_cles[:, 2]) * 100
            
            points_finaux = []
            
            if confiance_moyenne >= 95.0:
                print(f"\n[Succès] Confiance élevée ({confiance_moyenne:.1f}%). Utilisation des points de l'IA :")
                for i, point in enumerate(points_cles):
                    x, y, confiance = point

                    x_securise = np.clip(x, x_min, x_max)
                    y_securise = np.clip(y, y_min, y_max)
                    
                    points_finaux.append([x_securise, y_securise])
                    print(f" -> Point {i+1} : X={int(x_securise)}, Y={int(y_securise)} (Sûr à {confiance*100:.1f}%)")
            else:
                print(f"\n[Sécurité] Confiance trop faible ({confiance_moyenne:.1f}% < 95%). Utilisation des coins de la boîte :")
                points_finaux = [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max] 
                ]
                
                noms = ["Haut-Gauche", "Haut-Droit", "Bas-Droit", "Bas-Gauche"]
                for i, pt in enumerate(points_finaux):
                    print(f" -> Point {i+1} ({noms[i]}) : X={int(pt[0])}, Y={int(pt[1])} (Sûr à 100% via Box)")
                
            print("\nFormat Python pour notre script :")
            print(f"pts_ia = np.float32({points_finaux})")
            
        else:
            print("\nAucun autocollant détecté sur cette image.")
            print("L'IA n'est pas encore assez entraînée ou l'image est trop compliquée.")

    # ==========================================
    # EXPORTATION AUTOMATIQUE EN TFLITE
    # ==========================================
    print("\n" + "="*60)
    print("Exportation du modèle au format TFLite pour Flutter...")
    print("="*60 + "\n")
    
    try:
        fichier_export = best_model.export(
            format='tflite',
            imgsz=IMGSZ,
            optimize=True
        )
        print(f"\n[Succès] Modèle exporté avec succès ! Fichier disponible ici : {fichier_export}")
    except Exception as e:
        print(f"\n[Erreur] L'exportation a échoué : {e}")