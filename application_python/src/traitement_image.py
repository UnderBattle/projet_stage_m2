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

    print("[Traitement] Création du masque ciblé sur le vert...")
    
    # On crée le masque géométrique de base
    mask_geo = np.zeros(result_img.shape[:2], dtype=np.uint8)
    pts_int = np.int32(pts_autocollant).reshape((-1, 1, 2))
    cv2.fillPoly(mask_geo, [pts_int], 255)

    # On convertit la photo en HSV pour mieux détecter les couleurs
    hsv_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2HSV)

    # On définit la fourchette de notre vert spécifique en HSV
    # Le vert est autour de la teinte (Hue) 40 à 85 dans OpenCV
    vert_bas = np.array([40, 50, 20])   # Tolérance basse (vert sombre/ombragé)
    vert_haut = np.array([85, 255, 255]) # Tolérance haute (vert très éclairé)

    # On crée un masque qui cible uniquement ces pixels verts dans toute l'image
    mask_couleur = cv2.inRange(hsv_img, vert_bas, vert_haut)

    # On combine les deux masques
    # On garde les pixels qui sont verts (mask_couleur) et qui sont dans la zone (mask_geo)
    mask_final = cv2.bitwise_and(mask_couleur, mask_geo)

    # On dilate ce masque ultra-précis pour englober les bords flous
    kernel = np.ones((5, 5), np.uint8)
    mask_final = cv2.dilate(mask_final, kernel, iterations=2) # On passe à 2 pour être sûr de bien tout manger

    # Inpainting avec le masque parfait
    print("[Traitement] Effacement par inpainting...")
    result_img = cv2.inpaint(result_img, mask_final, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Séparation RGB et Alpha
    clim_rgb = clim_redimensionnee[:, :, 0:3]
    alpha_mask = clim_redimensionnee[:, :, 3] / 255.0

    # ==========================================
    # PHASE 3 : RENDU RÉALISTE (LUMIÈRE ET OMBRE ADAPTATIVE)
    # ==========================================
    print("[Traitement] Ajout de l'ombre portée adaptative...")
    
    # Paramètres de base de l'ombre
    decalage_x = 10  # Réduit pour rapprocher l'ombre
    decalage_y = 20  # Réduit pour rapprocher l'ombre
    flou = 61        # On garde un flou large pour une diffusion douce
    
    # Préparation de la forme de l'ombre
    padding = flou
    masque_elargi = cv2.copyMakeBorder(alpha_mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    ombre_floue = cv2.GaussianBlur(masque_elargi, (flou, flou), 0)
    
    # Position de l'ombre sur le mur
    x_ombre = x_offset + decalage_x - padding
    y_ombre = y_offset + decalage_y - padding

    # Application avec intensité adaptative
    if x_ombre >= 0 and y_ombre >= 0 and (y_ombre + ombre_floue.shape[0] < result_img.shape[0]) and (x_ombre + ombre_floue.shape[1] < result_img.shape[1]):
        
        # On récupère la zone du mur exacte où l'ombre va tomber
        roi_ombre = result_img[y_ombre:y_ombre+ombre_floue.shape[0], x_ombre:x_ombre+ombre_floue.shape[1]]
        
        # Calcul de la luminosité
        roi_grise = cv2.cvtColor(roi_ombre, cv2.COLOR_BGR2GRAY)
        luminosite_moyenne = np.mean(roi_grise)
        
        # Base à 0.05 (5% d'opacité min) + proportionnel jusqu'à 0.25 (25% d'opacité max)
        intensite_adaptative = 0.05 + (luminosite_moyenne / 255.0) * 0.20
        
        print(f"[Traitement] Luminosité du mur : {luminosite_moyenne:.1f}/255 -> Nouvelle Intensité de l'ombre : {intensite_adaptative:.2f}")
        
        # On applique cette intensité mathématique à notre ombre floue
        ombre_alpha_adaptee = ombre_floue * intensite_adaptative
        
        # On dessine l'ombre sur le mur
        for c in range(0, 3):
            roi_ombre[:, :, c] = roi_ombre[:, :, c] * (1.0 - ombre_alpha_adaptee)
            
        result_img[y_ombre:y_ombre+ombre_floue.shape[0], x_ombre:x_ombre+ombre_floue.shape[1]] = roi_ombre
    else:
        print("[Traitement] Attention : L'ombre sort de l'image, elle est ignorée.")
        
    # ==========================================
    # PHASE 3.5 : ADAPTATION DE LA COULEUR AMBIANTE ET LUMINOSITÉ
    # ==========================================
    print("[Traitement] Adaptation de la colorimétrie et luminosité...")
    
    # On vérifie d'abord que la clim ne dépasse pas du mur
    if (y_offset + nouvelle_hauteur <= result_img.shape[0]) and (x_offset + nouvelle_largeur <= result_img.shape[1]):
        
        # On récupère la zone exacte du mur où sera la clim
        roi_mur_clim = result_img[y_offset:y_offset+nouvelle_hauteur, x_offset:x_offset+nouvelle_largeur]
        
        # On calcule la couleur moyenne (BGR) du mur derrière la clim
        avg_color_per_row = np.average(roi_mur_clim, axis=0)
        avg_color_mur = np.average(avg_color_per_row, axis=0)
        
        # On crée un "calque" uni de cette couleur ambiante
        calque_ambiance = np.full(clim_rgb.shape, avg_color_mur, dtype=np.uint8)
        
        # On mélange la clim avec ce filtre (ex: 15% de la couleur du mur, 85% de la clim d'origine)
        # C'est ce qui "casse" le blanc pur numérique et l'intègre à la pièce
        influence_mur = 0.15
        clim_rgb = cv2.addWeighted(clim_rgb, 1.0 - influence_mur, calque_ambiance, influence_mur, 0)
        
        # On compare la vraie luminosité du mur avec la luminosité de la clim modifiée
        roi_mur_grise = cv2.cvtColor(roi_mur_clim, cv2.COLOR_BGR2GRAY)
        lum_mur = np.mean(roi_mur_grise)
        
        clim_grise = cv2.cvtColor(clim_rgb, cv2.COLOR_BGR2GRAY)
        mask_binaire = (alpha_mask * 255).astype(np.uint8)
        lum_clim = cv2.mean(clim_grise, mask=mask_binaire)[0]
        
        # Ratio mathématique direct
        ratio_lum = lum_mur / (lum_clim + 1e-5)
        
        # On adoucit ce ratio (on n'applique que 40% de la différence pour ne pas griser la clim)
        ratio_adouci = 1.0 - ((1.0 - ratio_lum) * 0.40) 
        ratio_adouci = np.clip(ratio_adouci, 0.7, 1.1) # Garde-fou de sécurité
        
        print(f"[Traitement] Lumière Mur: {lum_mur:.1f}, Clim: {lum_clim:.1f} -> Ratio adouci: {ratio_adouci:.2f}")
        
        # Application sur le canal Valeur (Lumière) du HSV
        clim_hsv = cv2.cvtColor(clim_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
        clim_hsv[:, :, 2] = clim_hsv[:, :, 2] * ratio_adouci
        clim_hsv[:, :, 2] = np.clip(clim_hsv[:, :, 2], 0, 255)
        clim_rgb = cv2.cvtColor(clim_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ==========================================
    # PHASE 4 : INCRUSTATION FINALE
    # ==========================================
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