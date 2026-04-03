from ultralytics import YOLO
import os

if __name__ == '__main__':
    print("[IA] Chargement du modèle YOLOv8-Pose (Nano)...")
    # On charge le tout petit modèle de base "nano"
    model = YOLO('yolov8n-pose.pt') 

    print("[IA] Lancement de l'entraînement...")
    results = model.train(
        data='../../dataset_autocollant/data.yaml',
        epochs=150,
        imgsz=640,
        batch=2,
        device='cpu'
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
            conf=0.5,     # On lui demande d'être sûre à au moins 50%
            show=True,
            save=True     # Va sauvegarder l'image résultat dans le dossier 'runs/pose/predict'
        )

        # On extrait les données mathématiques de la première image analysée
        resultat_ia = predictions[0]

        if len(resultat_ia.boxes) > 0:
            print("\nAutocollant détecté ! Extraction des points clés...")
            
            # On récupère les 4 points clés
            # keypoints.data contient [x, y, confiance] pour chaque point
            points_cles = resultat_ia.keypoints.data[0].cpu().numpy() 
            
            print("\nVoici les 4 coordonnées à envoyer à OpenCV (traitement_image.py) :")
            # On ne garde que les X et Y
            points_finaux = []
            for i, point in enumerate(points_cles):
                x, y, confiance = point
                points_finaux.append([x, y])
                print(f" -> Point {i+1} : X={int(x)}, Y={int(y)} (Sûr à {confiance*100:.1f}%)")
                
            print("\nFormat Python pour notre script :")
            print(f"pts_ia = np.float32({points_finaux})")
            
        else:
            print("\nAucun autocollant détecté sur cette image.")
            print("L'IA n'est pas encore assez entraînée ou l'image est trop compliquée.")