import cv2
import numpy as np

def incruster_climatisation(mur_img, clim_img, pts_autocollant, dim_clim_mm, dim_autocollant_mm):
    """
    Incruste une image de climatisation sur un mur à la place d'un autocollant repère.
    
    :param mur_img: Image OpenCV du mur (BGR)
    :param clim_img: Image OpenCV de la clim avec transparence (BGRA)
    :param pts_autocollant: Coordonnées des 4 coins de l'autocollant
    :param dim_clim_mm: Tuple (Hauteur, Largeur, Profondeur) de la clim en mm
    :param dim_autocollant_mm: Tuple (Hauteur, Largeur) de l'autocollant en mm
    :return: Image OpenCV du résultat final
    """
    # ==========================================
    # PHASE 1 : CALCULS ET COORDONNÉES
    # ==========================================
    print("[Traitement] Calcul de l'échelle...")
    
    pt_haut_gauche = pts_autocollant[0]
    pt_haut_droit = pts_autocollant[1]

    # Distance en pixels (Théorème de Pythagore)
    largeur_autocollant_px = np.sqrt((pt_haut_droit[0] - pt_haut_gauche[0])**2 + (pt_haut_droit[1] - pt_haut_gauche[1])**2)
    
    # Calcul du ratio
    largeur_autocollant_mm = dim_autocollant_mm[1]
    ratio_px_mm = largeur_autocollant_px / largeur_autocollant_mm

    # Nouvelles dimensions de la clim
    largeur_clim_mm = dim_clim_mm[1]
    hauteur_clim_mm = dim_clim_mm[0]

    nouvelle_largeur = int(largeur_clim_mm * ratio_px_mm)
    nouvelle_hauteur = int(hauteur_clim_mm * ratio_px_mm)

    print(f"[Traitement] Redimensionnement de la clim à : {nouvelle_largeur}x{nouvelle_hauteur} pixels")

    # Choix de l'algorithme d'interpolation
    if nouvelle_largeur > clim_img.shape[1]:
        methode_interpolation = cv2.INTER_LANCZOS4 
    else:
        methode_interpolation = cv2.INTER_AREA

    # Redimensionnement
    clim_redimensionnee = cv2.resize(clim_img, (nouvelle_largeur, nouvelle_hauteur), interpolation=methode_interpolation)

    # Point de collage final (converti en entiers)
    x_offset = int(pt_haut_gauche[0]) 
    y_offset = int(pt_haut_gauche[1]) 

    # ==========================================
    # PHASE 2 : MANIPULATION D'IMAGES
    # ==========================================
    # On fait une copie de l'image du mur pour ne pas modifier l'originale
    result_img = mur_img.copy()

    print("[Traitement] Effacement de l'autocollant par inpainting...")
    mask = np.zeros(result_img.shape[:2], dtype=np.uint8)
    pts_int = np.int32(pts_autocollant).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts_int], 255)

    # Dilatation du masque
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpainting
    result_img = cv2.inpaint(result_img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Séparation RGB et Alpha
    clim_rgb = clim_redimensionnee[:, :, 0:3]
    alpha_mask = clim_redimensionnee[:, :, 3] / 255.0 

    # Vérification des limites
    if (y_offset + nouvelle_hauteur > result_img.shape[0]) or (x_offset + nouvelle_largeur > result_img.shape[1]):
        raise ValueError("La clim dépasse de l'image du mur avec ces coordonnées ou cette taille !")

    print("[Traitement] Incrustation en cours...")
    roi = result_img[y_offset:y_offset+nouvelle_hauteur, x_offset:x_offset+nouvelle_largeur]

    # Mélange des pixels
    for c in range(0, 3):
        roi[:, :, c] = (alpha_mask * clim_rgb[:, :, c] + (1.0 - alpha_mask) * roi[:, :, c])

    result_img[y_offset:y_offset+nouvelle_hauteur, x_offset:x_offset+nouvelle_largeur] = roi

    return result_img