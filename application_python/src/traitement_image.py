import cv2
import numpy as np

def incruster_climatisation(mur_img, clim_img, pts_autocollant, dim_clim_mm, dim_autocollant_mm, position_cible=None):
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
    if position_cible is not None:
        x_offset = int(position_cible[0])
        y_offset = int(position_cible[1])
    else:
        x_offset = int(pt_haut_gauche[0]) 
        y_offset = int(pt_haut_gauche[1]) 

    print(f"[Traitement] Positionnement de la clim en X:{x_offset}, Y:{y_offset}")

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
    
    decalage_x = 10
    decalage_y = 20
    flou = 61
    padding = flou
    
    masque_elargi = cv2.copyMakeBorder(alpha_mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    ombre_floue = cv2.GaussianBlur(masque_elargi, (flou, flou), 0)
    
    x_ombre = x_offset + decalage_x - padding
    y_ombre = y_offset + decalage_y - padding

    # Calcul de l'intersection de l'ombre avec les bords de l'écran ---
    sy1 = max(0, y_ombre)
    sy2 = min(result_img.shape[0], y_ombre + ombre_floue.shape[0])
    sx1 = max(0, x_ombre)
    sx2 = min(result_img.shape[1], x_ombre + ombre_floue.shape[1])

    # Si l'ombre est au moins partiellement visible à l'écran
    if sx1 < sx2 and sy1 < sy2:
        roi_ombre = result_img[sy1:sy2, sx1:sx2]
        ombre_visible = ombre_floue[sy1 - y_ombre:sy2 - y_ombre, sx1 - x_ombre:sx2 - x_ombre]
        
        roi_grise = cv2.cvtColor(roi_ombre, cv2.COLOR_BGR2GRAY)
        luminosite_moyenne = np.mean(roi_grise)
        intensite_adaptative = 0.05 + (luminosite_moyenne / 255.0) * 0.20
        
        ombre_alpha_adaptee = ombre_visible * intensite_adaptative
        
        for c in range(0, 3):
            roi_ombre[:, :, c] = roi_ombre[:, :, c] * (1.0 - ombre_alpha_adaptee)
            
        result_img[sy1:sy2, sx1:sx2] = roi_ombre

    # ==========================================
    # PHASE 4 : LUMINOSITÉ ET INCRUSTATION DYNAMIQUE
    # ==========================================
    print("[Traitement] Incrustation dynamique (gestion des bords)...")
    
    # Calcul de l'intersection de la CLIM avec les bords de l'écran
    y1 = max(0, y_offset)
    y2 = min(result_img.shape[0], y_offset + nouvelle_hauteur)
    x1 = max(0, x_offset)
    x2 = min(result_img.shape[1], x_offset + nouvelle_largeur)

    # Si la clim est au moins partiellement visible sur l'image
    if x1 < x2 and y1 < y2:
        
        # On découpe le bout de mur visible
        roi_mur_clim = result_img[y1:y2, x1:x2]

        # On calcule quelle partie de la clim on a le droit d'afficher (si elle est coupée en haut/gauche)
        clim_y1 = y1 - y_offset
        clim_y2 = y2 - y_offset
        clim_x1 = x1 - x_offset
        clim_x2 = x2 - x_offset

        # On isole uniquement les pixels visibles de la clim
        visible_clim_rgb = clim_rgb[clim_y1:clim_y2, clim_x1:clim_x2].copy()
        visible_alpha = alpha_mask[clim_y1:clim_y2, clim_x1:clim_x2]

        avg_color_per_row = np.average(roi_mur_clim, axis=0)
        avg_color_mur = np.average(avg_color_per_row, axis=0)
        
        calque_ambiance = np.full(visible_clim_rgb.shape, avg_color_mur, dtype=np.uint8)
        influence_mur = 0.15
        visible_clim_rgb = cv2.addWeighted(visible_clim_rgb, 1.0 - influence_mur, calque_ambiance, influence_mur, 0)
        
        roi_mur_grise = cv2.cvtColor(roi_mur_clim, cv2.COLOR_BGR2GRAY)
        lum_mur = np.mean(roi_mur_grise)
        
        clim_grise = cv2.cvtColor(visible_clim_rgb, cv2.COLOR_BGR2GRAY)
        mask_binaire = (visible_alpha * 255).astype(np.uint8)
        
        # Sécurité : si la seule partie visible de la clim est 100% transparente (ex: le coin transparent du PNG sort de l'écran)
        if cv2.countNonZero(mask_binaire) > 0:
            lum_clim = cv2.mean(clim_grise, mask=mask_binaire)[0]
        else:
            lum_clim = lum_mur

        ratio_lum = lum_mur / (lum_clim + 1e-5)
        ratio_adouci = 1.0 - ((1.0 - ratio_lum) * 0.40) 
        ratio_adouci = np.clip(ratio_adouci, 0.7, 1.1)
        
        print(f"[Traitement] Lumière Mur: {lum_mur:.1f}, Clim: {lum_clim:.1f} -> Ratio adouci: {ratio_adouci:.2f}")
        
        clim_hsv = cv2.cvtColor(visible_clim_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
        clim_hsv[:, :, 2] = np.clip(clim_hsv[:, :, 2] * ratio_adouci, 0, 255)
        visible_clim_rgb = cv2.cvtColor(clim_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        for c in range(0, 3):
            roi_mur_clim[:, :, c] = (visible_alpha * visible_clim_rgb[:, :, c] + (1.0 - visible_alpha) * roi_mur_clim[:, :, c])

        result_img[y1:y2, x1:x2] = roi_mur_clim

    return result_img