import cv2
import numpy as np

AUTOCOLLANT_TAILLE_REELLE = (100, 50) # Hauteur, Largeur en mm
TAKAO_PLUS_DIMENSION_REELLE = (270, 798, 240) # Hauteur, Largeur, Profondeur en mm
TAKAO_PLUS_IMG_BLANC = './clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png'
TAKAO_PLUS_IMG_NOIR = './clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png'
TARGET_WALL = 'gettyimages-1837566278-612x612_avec_autocollant_fictif.jpg'

print("Chargement des images...")
src_unit = cv2.imread(TAKAO_PLUS_IMG_BLANC, cv2.IMREAD_UNCHANGED)
target_wall = cv2.imread(TARGET_WALL, cv2.IMREAD_COLOR)

if src_unit is None or target_wall is None:
    print(f"Erreur: Impossible de charger {TAKAO_PLUS_IMG_BLANC} ou {TARGET_WALL}.")
    exit()

# ==========================================
# PHASE 1 : CALCULS ET COORDONNÉES
# ==========================================

# Définition des coordonnées de l'autocollant sur la photo du mur
# Format: [Haut-Gauche, Haut-Droit, Bas-Droit, Bas-Gauche]
pts_autocollant = np.float32([[18, 75], [28, 75], [28, 91], [18, 91]])

# Calcule la nouvelle taille de la clim en utilisant l'échelle de l'autocollant
# 1. On récupère les points Haut-Gauche et Haut-Droit de l'autocollant
pt_haut_gauche = pts_autocollant[0]
pt_haut_droit = pts_autocollant[1]

# 2. On calcule la distance en pixels entre ces deux points (Théorème de Pythagore)
largeur_autocollant_px = np.sqrt((pt_haut_droit[0] - pt_haut_gauche[0])**2 + (pt_haut_droit[1] - pt_haut_gauche[1])**2)

# 3. On calcule le ratio (Pixels par millimètre)
largeur_autocollant_mm = AUTOCOLLANT_TAILLE_REELLE[1] # 50 mm
ratio_px_mm = largeur_autocollant_px / largeur_autocollant_mm

# 4. On applique ce ratio aux dimensions réelles de la clim
largeur_clim_mm = TAKAO_PLUS_DIMENSION_REELLE[1] # 798 mm
hauteur_clim_mm = TAKAO_PLUS_DIMENSION_REELLE[0] # 270 mm

nouvelle_largeur = int(largeur_clim_mm * ratio_px_mm)
nouvelle_hauteur = int(hauteur_clim_mm * ratio_px_mm)

print(f"Ratio calculé : {ratio_px_mm:.2f} pixels/mm")
print(f"Redimensionnement à l'échelle : {nouvelle_largeur}x{nouvelle_hauteur} pixels")

# On choisit le meilleur algorithme mathématique selon si on agrandit ou on rétrécit l'image
if nouvelle_largeur > src_unit.shape[1]:
    # Upscaling
    print("Agrandissement de l'image (Interpolation Lanczos4)...")
    methode_interpolation = cv2.INTER_LANCZOS4 
else:
    # Downscaling
    print("Réduction de l'image (Interpolation Area)...")
    methode_interpolation = cv2.INTER_AREA

# On applique le redimensionnement avec la méthode choisie
clim_redimensionnee = cv2.resize(src_unit, (nouvelle_largeur, nouvelle_hauteur), interpolation=methode_interpolation)

# On convertit les coordonnées flottantes en entiers pour les pixels (Point de collage final)
x_offset = int(pt_haut_gauche[0]) 
y_offset = int(pt_haut_gauche[1]) 

print(f"Positionnement de la clim en X:{x_offset}, Y:{y_offset}")

# ==========================================
# PHASE 2 : MANIPULATION D'IMAGES
# ==========================================

# EFFACEMENT DE L'AUTOCOLLANT (INPAINTING)
# Placé ici, on prépare la "toile de fond" juste avant d'y coller la clim
# ==========================================
# PHASE 2 : MANIPULATION D'IMAGES
# ==========================================

print("Effacement de l'autocollant vert par inpainting...")
mask = np.zeros(target_wall.shape[:2], dtype=np.uint8)
pts_int = np.int32(pts_autocollant).reshape((-1, 1, 2))
cv2.fillPoly(mask, [pts_int], 255)

print("Dilatation du masque pour éviter les bords verts...")
# 1. On crée un "kernel" (un outil mathématique qui agit comme un pinceau épais de 5x5 pixels)
kernel = np.ones((5, 5), np.uint8)

# On "dilate" le blanc du masque. 
# 'iterations=1' agrandit le masque d'environ 2 pixels de chaque côté. 
# Tu peux passer à iterations=2 ou 3 si tu vois encore du vert !
mask = cv2.dilate(mask, kernel, iterations=1)

# On applique l'inpainting avec le masque agrandi
# Tu peux aussi passer le inpaintRadius à 5 pour qu'il cherche la couleur du mur un peu plus loin
target_wall = cv2.inpaint(target_wall, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

# Séparation de l'image de la clim et de sa transparence
clim_rgb = clim_redimensionnee[:, :, 0:3]
# On normalise le masque alpha entre 0 et 1
alpha_mask = clim_redimensionnee[:, :, 3] / 255.0 

# Définition de la "Region of Interest" sur le mur
# On vérifie d'abord que la clim ne dépasse pas de l'image du mur
if (y_offset + nouvelle_hauteur > target_wall.shape[0]) or (x_offset + nouvelle_largeur > target_wall.shape[1]):
    print("Erreur : La clim dépasse de l'image du mur avec ces coordonnées ou cette taille !")
    exit()

roi = target_wall[y_offset:y_offset+nouvelle_hauteur, x_offset:x_offset+nouvelle_largeur]

# Superposition avec la transparence
print("Incrustation en cours...")
for c in range(0, 3):
    # La formule du mélange (blending) : (Pixel Clim * Transparence) + (Pixel Mur * (1 - Transparence))
    roi[:, :, c] = (alpha_mask * clim_rgb[:, :, c] + (1.0 - alpha_mask) * roi[:, :, c])

# On remet la ROI modifiée dans l'image du mur originale
target_wall[y_offset:y_offset+nouvelle_hauteur, x_offset:x_offset+nouvelle_largeur] = roi

# Rendu et sauvegarde
print("Sauvegarde du résultat final...")
cv2.imwrite('resultat_echelle.jpg', target_wall)
print("Image 'resultat_echelle.jpg' créée avec succès !")