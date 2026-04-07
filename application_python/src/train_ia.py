from ultralytics import YOLO
import os
import numpy as np

if __name__ == '__main__':
    print("[IA] Chargement du modèle YOLOv8-Pose (Nano)...")
    # On charge le tout petit modèle de base "nano"
    model = YOLO('yolov8n-pose.pt') 

    print("[IA] Lancement de l'entraînement...")
    results = model.train(
        data='../../dataset_autocollant/data.yaml', 
        epochs=75,
        imgsz=1024,      
        batch=6,
        device='cpu',
        
        # On désactive la géométrie et les transformations pour se concentrer sur l'apprentissage des points clés
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        degrees=0.0,
        
        # Pénalité de l'absence de détection (box) et de points clés (pose)
        pose=30.0,
        box=5.0
    )
    
    # ==========================================
    # LE TABLEAU DE BORD DES RÉSULTATS AVANCÉ
    # ==========================================
    print("\n" + "="*60)
    print("Résumer des performances du modèle :")
    print("="*60)
    
    chemin_best_pt = f"{results.save_dir}/weights/best.pt"
    print(f"Dossier de sauvegarde : {results.save_dir}")
    print(f"Fichier 'Cerveau' à utiliser : {chemin_best_pt}\n")

    box_map50 = results.box.map50 * 100   
    box_map95 = results.box.map * 100     
    pose_map50 = results.pose.map50 * 100
    pose_map95 = results.pose.map * 100

    print("=== Détection Globale ===")
    print(f"Score Global (Tolérance 50%)             : {box_map50:.1f} %")
    print(f"Score Strict (Tolérance 50-95%)          : {box_map95:.1f} %\n")

    print("=== Détection des 4 Coins ===")
    print(f"Score Global (Tolérance 50%)             : {pose_map50:.1f} %")
    print(f"Score Strict (Tolérance 50-95%)          : {pose_map95:.1f} %\n")
    print("="*60 + "\n")

    # ==========================================
    # REALITY CHECK
    # ==========================================
    print("\n" + "="*60)
    print("Lancement du Reality Check...")
    print("="*60 + "\n")

    # On charge le modèle
    best_model = YOLO(chemin_best_pt)

    # On choisit une image de test
    image_test = "../img_test/IMG_20260401_090116.jpg" 
    
    if not os.path.exists(image_test):
        print(f"Erreur : L'image de test {image_test} est introuvable.")
    else:
        print(f"[Test] Analyse de l'image : {image_test}")
        
        # L'IA regarde l'image et fait sa prédiction
        predictions = best_model.predict(
            source=image_test, 
            conf = 0.5,     # On lui demande d'être sûre à au moins 50%
            show = False,
            save = True     # Va sauvegarder l'image résultat dans le dossier 'runs/pose/predict'
        )

        # On extrait les données mathématiques de la première image analysée
        resultat_ia = predictions[0]

        if len(resultat_ia.boxes) > 0:
            print("\nAutocollant détecté ! Extraction des points...")
            
            # On récupère les limites strictes de la boîte englobante [x_min, y_min, x_max, y_max]
            boite = resultat_ia.boxes.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = boite
            
            points_cles = resultat_ia.keypoints.data[0].cpu().numpy() 
            
            # Calcul de la confiance moyenne des points clés (en %) pour évaluer la fiabilité de la prédiction
            # La colonne 2 contient les confiances de chaque point
            confiance_moyenne = np.mean(points_cles[:, 2]) * 100
            
            points_finaux = []
            
            # Choix de la stratégie en fonction de la confiance moyenne des points clés
            if confiance_moyenne >= 80.0:
                print(f"\n[Succès] Confiance élevée ({confiance_moyenne:.1f}%). Utilisation des points de l'IA :")
                for i, point in enumerate(points_cles):
                    x, y, confiance = point

                    # Filet de sécurité classique
                    x_securise = np.clip(x, x_min, x_max)
                    y_securise = np.clip(y, y_min, y_max)
                    
                    points_finaux.append([x_securise, y_securise])
                    print(f" -> Point {i+1} : X={int(x_securise)}, Y={int(y_securise)} (Sûr à {confiance*100:.1f}%)")
            else:
                print(f"\n[Sécurité] Confiance trop faible ({confiance_moyenne:.1f}% < 80%). Utilisation des coins de la boîte :")
                # On crée les 4 coins parfaits du rectangle de la Bounding Box
                points_finaux = [
                    [x_min, y_min], # Haut-Gauche
                    [x_max, y_min], # Haut-Droit
                    [x_max, y_max], # Bas-Droit
                    [x_min, y_max]  # Bas-Gauche
                ]
                
                noms = ["Haut-Gauche", "Haut-Droit", "Bas-Droit", "Bas-Gauche"]
                for i, pt in enumerate(points_finaux):
                    print(f" -> Point {i+1} ({noms[i]}) : X={int(pt[0])}, Y={int(pt[1])} (Sûr à 100% via Box)")
                
            print("\nFormat Python pour notre script :")
            print(f"pts_ia = np.float32({points_finaux})")
            
        else:
            print("\nAucun autocollant détecté sur cette image.")
            print("L'IA n'est pas encore assez entraînée ou l'image est trop compliquée.")