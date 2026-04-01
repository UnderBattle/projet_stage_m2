import cv2
import numpy as np
from src.traitement_image import incruster_climatisation

# ==========================================
# CONFIGURATION ET CONSTANTES
# ==========================================
AUTOCOLLANT_TAILLE_REELLE = (100, 50) # Hauteur, Largeur en mm
TAKAO_PLUS_DIMENSION_REELLE = (270, 798, 240) # Hauteur, Largeur, Profondeur en mm
TAKAO_PLUS_IMG_BLANC = './installations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png'
TAKAO_PLUS_IMG_NOIR = './installations/clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png'
# TARGET_WALL = './img_test/Cette-simple-astuce-de-decorateur-rendra-votre-interieur-encore-plus-beau_avec_autocollant_fictif.jpg'
TARGET_WALL = './img_test/IMG_20260401_090116.jpg'
RESULT_PATH = "./export/resultat_noir_echelle_reelle_luminosite.jpg"

# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================
print("Chargement des images...")
src_unit = cv2.imread(TAKAO_PLUS_IMG_NOIR, cv2.IMREAD_UNCHANGED)
target_wall = cv2.imread(TARGET_WALL, cv2.IMREAD_COLOR)

if src_unit is None or target_wall is None:
    print(f"Erreur: Impossible de charger {TAKAO_PLUS_IMG_NOIR} ou {TARGET_WALL}.")
    exit()

# Définition des coordonnées de l'autocollant (simulation de l'IA)
pts_autocollant = np.float32([[1129, 1128], [1208, 1128], [1206, 1288], [1128, 1287]]) # Pour clim Image Réelle
# pts_autocollant = np.float32([[667, 115], [698, 115], [698, 169], [667, 169]]) # Pour clim Image Fictif

# ==========================================
# EXÉCUTION DU TRAITEMENT
# ==========================================
try:
    image_finale = incruster_climatisation(
        mur_img = target_wall,
        clim_img = src_unit,
        pts_autocollant = pts_autocollant,
        dim_clim_mm = TAKAO_PLUS_DIMENSION_REELLE,
        dim_autocollant_mm = AUTOCOLLANT_TAILLE_REELLE
    )
    
    # Rendu et sauvegarde
    print("Sauvegarde du résultat final...")
    cv2.imwrite(RESULT_PATH, image_finale)
    print(f"Image sauvegardée avec succès dans : {RESULT_PATH}")

except ValueError as e:
    # Si la fonction lève une erreur (ex: dépassement des limites), on l'attrape ici
    print(f"Erreur lors du traitement : {e}")