from ultralytics import YOLO

if __name__ == '__main__':
    print("[IA] Chargement du modèle YOLOv8-Pose (Nano)...")
    # On charge le tout petit modèle de base "nano" spécialisé dans les points-clés
    model = YOLO('yolov8n-pose.pt') 

    print("[IA] Lancement de l'entraînement...")
    # On lance l'apprentissage
    results = model.train(
        data='../../dataset_autocollant/data.yaml', # Le chemin vers ton fichier YAML
        epochs=100,      # Le nombre de fois que l'IA va étudier tes 6 images
        imgsz=640,      # La taille de redimensionnement des images pour l'IA
        batch=2,        # Combien d'images elle traite en même temps
        device='cpu'    # Force l'utilisation du processeur classique (idéal pour un test rapide)
    )
    
    print("[IA] Entraînement terminé ! Le cerveau est sauvegardé dans le dossier 'runs/pose/train/weights/best.pt'")