import cv2
import numpy as np

def incruster_climatisation(mur_img, clim_img, pts_autocollant, dim_clim_mm, dim_autocollant_mm, position_cible=None):
    # On fait une copie de l'image du mur pour ne pas modifier l'originale
    result_img = mur_img.copy()
    h_mur, w_mur = result_img.shape[:2]
    print(f"\n[Traitement] Dimensions de l'image du mur : {w_mur}x{h_mur} pixels")

    # ==========================================
    # PHASE 1 : EFFACEMENT DE L'AUTOCOLLANT
    # ==========================================
    print("\n[Traitement] === PHASE 1 : EFFACEMENT DE L'AUTOCOLLANT ===")
    mask_geo = np.zeros((h_mur, w_mur), dtype=np.uint8)
    pts_int = np.int32(pts_autocollant).reshape((-1, 1, 2))
    cv2.fillPoly(mask_geo, [pts_int], 255)

    hsv_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2HSV)
    vert_bas = np.array([40, 50, 20])
    vert_haut = np.array([85, 255, 255])
    mask_couleur = cv2.inRange(hsv_img, vert_bas, vert_haut)

    mask_final = cv2.bitwise_and(mask_couleur, mask_geo)
    mask_final = cv2.dilate(mask_final, np.ones((5, 5), np.uint8), iterations=2)
    
    pixels_a_effacer = cv2.countNonZero(mask_final)
    print(f"[Traitement] Nombre de pixels verts ciblés (Inpainting) : {pixels_a_effacer} pixels")
    
    result_img = cv2.inpaint(result_img, mask_final, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # ==========================================
    # PHASE 2 : CALCUL DE LA PERSPECTIVE STABILISÉE
    # ==========================================
    print("\n[Traitement] === PHASE 2 : CALCUL DE LA PERSPECTIVE STABILISEE ===")
    
    # Extraction des points d'origine
    pt_hg, pt_hd, pt_bd, pt_bg = np.float32(pts_autocollant)
    print(f"[Traitement] Points d'origine : HG:{pt_hg}, HD:{pt_hd}, BG:{pt_bg}, BD:{pt_bd}")
    
    # Affichage du problème dans le terminal
    w_haut = np.linalg.norm(pt_hd - pt_hg)
    w_bas = np.linalg.norm(pt_bd - pt_bg)
    print(f"[Traitement] Largeur détectée -> Haut: {w_haut:.1f}px | Bas: {w_bas:.1f}px")
    if abs(w_haut - w_bas) > 1:
        print("[Traitement] Attention : Légère distorsion détectée. Lissage en cours...")

    # Lissage mathématique pour stabiliser la perspective
    dx = pt_hd[0] - pt_hg[0]
    dy = pt_hd[1] - pt_hg[1]
    largeur_px = np.sqrt(dx**2 + dy**2)
    angle_rad = np.arctan2(dy, dx)
    print(f"[Traitement] Écarts (dx, dy) du bord haut : dx={dx:.2f}, dy={dy:.2f}")
    
    # On force la hauteur en utilisant le vrai ratio physique de l'autocollant
    ratio_physique = dim_autocollant_mm[0] / dim_autocollant_mm[1]
    hauteur_px = largeur_px * ratio_physique
    
    print(f"[Traitement] Ratio physique de l'autocollant (H/L) : {ratio_physique:.2f}")
    print(f"[Traitement] Rectification -> Angle: {np.degrees(angle_rad):.2f}° | Taille 2D lissée: {largeur_px:.1f}x{hauteur_px:.1f}px")

    # Calcul des vecteurs de direction
    u = np.array([largeur_px * np.cos(angle_rad), largeur_px * np.sin(angle_rad)])
    v = np.array([-hauteur_px * np.sin(angle_rad), hauteur_px * np.cos(angle_rad)])
    print(f"[Traitement] Vecteur directeur Largeur (u) : [{u[0]:.2f}, {u[1]:.2f}]")
    print(f"[Traitement] Vecteur directeur Hauteur (v) : [{v[0]:.2f}, {v[1]:.2f}]")

    # Construction des 4 nouveaux points lissés
    pts_dst_lisses = np.float32([
        pt_hg,           # Haut-Gauche (Point d'ancrage)
        pt_hg + u,       # Haut-Droit
        pt_hg + u + v,   # Bas-Droit
        pt_hg + v        # Bas-Gauche
    ])
    print(f"[Traitement] Nouveaux points cibles (lissés) :\n{pts_dst_lisses}")

    # Gestion du déplacement manuel de la cible
    if position_cible is not None:
        dx_souris = position_cible[0] - pts_dst_lisses[0][0]
        dy_souris = position_cible[1] - pts_dst_lisses[0][1]
        pts_dst_lisses += [dx_souris, dy_souris]
        print(f"[Traitement] Déplacement manuel cible -> X:{int(position_cible[0])}, Y:{int(position_cible[1])}")

    # Préparation de l'image source
    h_clim_mm, w_clim_mm, _ = dim_clim_mm
    h_auto_mm, w_auto_mm = dim_autocollant_mm
    h_img_clim, w_img_clim = clim_img.shape[:2]
    print(f"[Traitement] Dimensions de l'image source clim : {w_img_clim}x{h_img_clim}px")

    # Rapport d'échelle virtuel dans l'image PNG
    w_auto_px = (w_auto_mm / w_clim_mm) * w_img_clim
    h_auto_px = (h_auto_mm / h_clim_mm) * h_img_clim
    print(f"[Traitement] Taille virtuelle de l'autocollant sur la clim : {w_auto_px:.1f}x{h_auto_px:.1f}px")

    pts_src = np.float32([
        [0, 0],
        [w_auto_px, 0],
        [w_auto_px, h_auto_px],
        [0, h_auto_px]
    ])

    # Matrice final de déformation
    H_matrix, _ = cv2.findHomography(pts_src, pts_dst_lisses)
    print(f"[Traitement] Matrice d'Homographie générée :\n{H_matrix}")
    print("[Traitement] Application de la matrice (WarpPerspective)...")
    
    # On applique la déformation sur tout le mur
    clim_warped = cv2.warpPerspective(clim_img, H_matrix, (w_mur, h_mur), flags=cv2.INTER_LINEAR)

    # Séparation RGB et Alpha
    clim_rgb = clim_warped[:, :, 0:3]
    alpha_mask = clim_warped[:, :, 3] / 255.0

    # ==========================================
    # PHASE 3 : OMBRE PORTÉE EN PERSPECTIVE
    # ==========================================
    print("\n[Traitement] === PHASE 3 : OMBRE PORTEE ===")
    # On décale le masque alpha de la clim tordue pour faire l'ombre (dx=10, dy=20)
    M_trans = np.float32([[1, 0, 10], [0, 1, 20]])
    alpha_decale = cv2.warpAffine(alpha_mask, M_trans, (w_mur, h_mur))
    
    # On floute l'ombre
    ombre_floue = cv2.GaussianBlur(alpha_decale, (61, 61), 0)

    # Intensité de l'ombre basée sur la luminosité globale du mur
    lum_mur = np.mean(cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY))
    intensite = 0.05 + (lum_mur / 255.0) * 0.20
    print(f"[Traitement] Luminosité globale du mur : {lum_mur:.1f}/255")
    print(f"[Traitement] Intensité de l'ombre calculée : {intensite:.3f} (soit {(intensite*100):.1f}%)")
    
    ombre_alpha = ombre_floue * intensite

    # Application de l'ombre
    for c in range(3):
        result_img[:, :, c] = result_img[:, :, c] * (1.0 - ombre_alpha)

    # ==========================================
    # PHASE 4 : LUMIÈRE ET INCRUSTATION FINALE
    # ==========================================
    print("\n[Traitement] === PHASE 4 : LUMIERE ET INCRUSTATION FINALE ===")
    mask_binaire = (alpha_mask * 255).astype(np.uint8)

    # Si la clim est au moins partiellement visible
    pixels_clim_visibles = cv2.countNonZero(mask_binaire)
    print(f"[Traitement] Pixels visibles de la clim à incruster : {pixels_clim_visibles}px")
    
    if pixels_clim_visibles > 0:
        
        # On calcule les lumières pile sous la clim déformée
        moyenne_couleur_mur = cv2.mean(result_img, mask=mask_binaire)[:3]
        lum_mur_sous_clim = cv2.mean(cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY), mask=mask_binaire)[0]

        clim_grise = cv2.cvtColor(clim_rgb, cv2.COLOR_BGR2GRAY)
        lum_clim = cv2.mean(clim_grise, mask=mask_binaire)[0]

        print(f"[Traitement] Teinte moyenne du mur sous la clim (BGR) : {[round(val, 1) for val in moyenne_couleur_mur]}")
        print(f"[Traitement] Luminosité du mur sous la clim : {lum_mur_sous_clim:.1f}/255")
        print(f"[Traitement] Luminosité d'origine de la clim : {lum_clim:.1f}/255")

        # Color Cast (Teinte de la pièce)
        calque_ambiance = np.full(clim_rgb.shape, moyenne_couleur_mur, dtype=np.uint8)
        clim_rgb = cv2.addWeighted(clim_rgb, 0.85, calque_ambiance, 0.15, 0)
        print("[Traitement] Filtre de transfert de couleur ambiante (Color Cast) appliqué (15%).")

        # Adaptation de l'éclairage en HSV
        ratio_lum = lum_mur_sous_clim / (lum_clim + 1e-5)
        ratio_adouci = np.clip(1.0 - ((1.0 - ratio_lum) * 0.40), 0.7, 1.1)
        print(f"[Traitement] Ratio brut d'éclairage : {ratio_lum:.2f} -> Ratio adouci appliqué : {ratio_adouci:.2f}")

        clim_hsv = cv2.cvtColor(clim_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
        clim_hsv[:, :, 2] = np.clip(clim_hsv[:, :, 2] * ratio_adouci, 0, 255)
        clim_rgb = cv2.cvtColor(clim_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Fusion finale mathématique pixel par pixel !
    print("[Traitement] Incrustation finale pixel par pixel en cours...")
    for c in range(3):
        result_img[:, :, c] = (alpha_mask * clim_rgb[:, :, c] + (1.0 - alpha_mask) * result_img[:, :, c])

    print("[Traitement] === TERMINE AVEC SUCCES ===\n")
    return result_img