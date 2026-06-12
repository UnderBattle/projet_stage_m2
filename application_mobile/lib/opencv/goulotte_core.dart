import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;

class GoulotteCore {
  /// Isolate pour incruster la goulotte sur l'image.
  static Future<Uint8List?> incrusterGoulotteIsolate(Map<String, dynamic> params) async {
    return await incrusterGoulotte(
      imageDeFondBytes: params['imageDeFondBytes'] as Uint8List,
      ptDepartX: params['ptDepartX'] as double,
      ptDepartY: params['ptDepartY'] as double,
      ptArriveeX: params['ptArriveeX'] as double,
      ptArriveeY: params['ptArriveeY'] as double,
      largeurPx: params['largeurPx'] as double,
    );
  }

  /// Dessine une ligne épaisse avec des extrémités plates en traçant un polygone rectangulaire.
  /// Remplace `cv.line` qui produit des bouts arrondis avec une épaisseur élevée.
  static void tracerLigneRectangulaire(cv.Mat mat, cv.Point p1, cv.Point p2, cv.Scalar color, double thickness) {
    double dx = (p2.x - p1.x).toDouble();
    double dy = (p2.y - p1.y).toDouble();
    double len = math.sqrt(dx * dx + dy * dy);
    
    if (len < 1.0) return; // Sécurité si les points sont identiques
    
    double ux = dx / len;
    double uy = dy / len;
    
    // Vecteur normal (perpendiculaire)
    double nx = -uy;
    double ny = ux;
    
    // Demi-épaisseur
    double ht = thickness / 2.0;
    
    // Calcul des 4 coins du rectangle pour avoir des bouts parfaitement plats
    cv.Point ptA = cv.Point((p1.x + nx * ht).toInt(), (p1.y + ny * ht).toInt());
    cv.Point ptB = cv.Point((p1.x - nx * ht).toInt(), (p1.y - ny * ht).toInt());
    cv.Point ptC = cv.Point((p2.x - nx * ht).toInt(), (p2.y - ny * ht).toInt());
    cv.Point ptD = cv.Point((p2.x + nx * ht).toInt(), (p2.y + ny * ht).toInt());
    
    cv.fillPoly(mat, cv.VecVecPoint.fromList([[ptA, ptB, ptC, ptD]]), color, lineType: cv.LINE_AA);
  }

  // =========================================================================
  // === INCRUSTATION REALISTE DE LA GOULOTTE ===
  // =========================================================================
  static Future<Uint8List?> incrusterGoulotte({
    required Uint8List imageDeFondBytes, // Sera toujours le Mur Propre désormais
    required double ptDepartX,
    required double ptDepartY,
    required double ptArriveeX,
    required double ptArriveeY,
    required double largeurPx,
  }) async {
    try {
      cv.Mat fondMat = cv.imdecode(imageDeFondBytes, cv.IMREAD_COLOR);
      int w = fondMat.cols;
      int h = fondMat.rows;

      cv.Point p1 = cv.Point(ptDepartX.toInt(), ptDepartY.toInt());
      cv.Point p2 = cv.Point(ptArriveeX.toInt(), ptArriveeY.toInt());

      // Identifie le point le plus bas pour dessiner l'effet de perspective du "bouchon" 3D.
      cv.Point bottomPt = p1.y > p2.y ? p1 : p2;
      cv.Point topPt = p1.y > p2.y ? p2 : p1;

      double dxCap = (bottomPt.x - topPt.x).toDouble();
      double dyCap = (bottomPt.y - topPt.y).toDouble();
      double lenCap = math.sqrt(dxCap * dxCap + dyCap * dyCap);

      // Variables géométriques pour le bouchon 3D
      cv.Point ptLeft = bottomPt;
      cv.Point ptRight = bottomPt;
      cv.Point ptLeftMur = bottomPt;
      cv.Point ptRightMur = bottomPt;

      if (lenCap >= 1.0) {
        double uxCap = dxCap / lenCap;
        double uyCap = dyCap / lenCap;
        double nxCap = -uyCap;
        double nyCap = uxCap;
        double htCap = largeurPx / 2.0;

        // Base plate inférieure du cylindre de plastique
        ptLeft = cv.Point((bottomPt.x - nxCap * htCap).toInt(), (bottomPt.y - nyCap * htCap).toInt());
        ptRight = cv.Point((bottomPt.x + nxCap * htCap).toInt(), (bottomPt.y + nyCap * htCap).toInt());

        // Calcule la perspective du bouchon qui semble s'enfoncer dans le mur.
        double depthExtrusion = largeurPx * 0.20; 
        double ex = 0.0; 
        double ey = depthExtrusion; 

        // Effet de rétrécissement pour simuler la perspective lointaine (le mur)
        double shrinkFactor = 0.85; 
        ptLeftMur = cv.Point((bottomPt.x - nxCap * htCap * shrinkFactor + ex).toInt(), 
                             (bottomPt.y - nyCap * htCap * shrinkFactor + ey).toInt());
        ptRightMur = cv.Point((bottomPt.x + nxCap * htCap * shrinkFactor + ex).toInt(), 
                              (bottomPt.y + nyCap * htCap * shrinkFactor + ey).toInt());
      }

      // 1. Création du masque binaire global (Goulotte + Bouchon)
      cv.Mat maskBinaire = cv.Mat.zeros(h, w, cv.MatType.CV_8UC1);
      tracerLigneRectangulaire(maskBinaire, p1, p2, cv.Scalar.all(255), largeurPx);
      if (lenCap >= 1.0) {
        cv.fillPoly(maskBinaire, cv.VecVecPoint.fromList([[ptLeft, ptRight, ptRightMur, ptLeftMur]]), cv.Scalar.all(255), lineType: cv.LINE_AA);
      }

      // 2. Construit l'apparence visuelle de la goulotte
      cv.Mat goulotteBgr = cv.Mat.zeros(h, w, cv.MatType.CV_8UC3);
      tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(210, 210, 210, 0), largeurPx);
      tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(240, 245, 245, 0), largeurPx * 0.85);
      tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(245, 250, 250, 0), largeurPx * 0.50);
      tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(255, 255, 255, 0), largeurPx * 0.15);
      
      if (lenCap >= 1.0) {
        // Face inférieure (le bouchon du bas) encore plus sombre pour bien marquer le volume (130 au lieu de 160)
        cv.fillPoly(goulotteBgr, cv.VecVecPoint.fromList([[ptLeft, ptRight, ptRightMur, ptLeftMur]]), cv.Scalar(130, 135, 135, 0), lineType: cv.LINE_AA);
        int edgeThickness = math.max(1, (largeurPx * 0.05).toInt());
        cv.line(goulotteBgr, ptLeft, ptRight, cv.Scalar(160, 165, 165, 0), thickness: edgeThickness, lineType: cv.LINE_AA);
      }

      // Le flou est appliqué à la toute fin pour que la ligne sombre du bas
      // se fonde de manière naturelle avec le reste de la goulotte (dégradé doux)
      cv.Mat goulotteBgrSmooth = cv.gaussianBlur(goulotteBgr, (7, 7), 0.0);

      // 3. Création de l'ombre portée de la goulotte (Drop Shadow)
      // Ombre adaptative calculée sur le fond (comme pour l'équipement)
      cv.Mat grayFond = cv.cvtColor(fondMat, cv.COLOR_BGR2GRAY);
      int downscaleSobelG = 32;
      cv.Mat grayFondSmall = cv.resize(grayFond, (w ~/ downscaleSobelG, h ~/ downscaleSobelG));
      cv.Mat maskBinaireSmallG = cv.resize(maskBinaire, (w ~/ downscaleSobelG, h ~/ downscaleSobelG));

      var meanStdDevGoulotte = cv.meanStdDev(grayFondSmall);
      double ecartTypeMurGoulotte = meanStdDevGoulotte.$2.val[0];
      double ratioContrasteGoulotte = (ecartTypeMurGoulotte / 50.0).clamp(0.2, 1.2);

      cv.Mat grayFondFlou = cv.gaussianBlur(grayFondSmall, (7, 7), 0.0);
      cv.Mat sobelXG = cv.sobel(grayFondFlou, cv.MatType.CV_32F, 1, 0, ksize: 3);
      cv.Mat sobelYG = cv.sobel(grayFondFlou, cv.MatType.CV_32F, 0, 1, ksize: 3);

      cv.Scalar meanSobelXG = cv.mean(sobelXG, mask: maskBinaireSmallG);
      cv.Scalar meanSobelYG = cv.mean(sobelYG, mask: maskBinaireSmallG);
      double gradXG = meanSobelXG.val[0];
      double gradYG = meanSobelYG.val[0];
      double normeG = math.sqrt(gradXG * gradXG + gradYG * gradYG) + 0.0001;

      double dirLumiereXG = gradXG / normeG;
      double dirLumiereYG = gradYG / normeG;

      double ratioVolumeGoulotte = 40.0 / 100.0; // Goulotte moins profonde
      double forceOmbreGoulotte = 12.0 * ratioVolumeGoulotte;
      double longueurOmbreGoulotte = forceOmbreGoulotte * ratioContrasteGoulotte;

      double shiftXG = normeG < 1.0 ? (3.0 * ratioVolumeGoulotte) : -dirLumiereXG * longueurOmbreGoulotte;
      double shiftYG = normeG < 1.0 ? (8.0 * ratioVolumeGoulotte) : -dirLumiereYG * longueurOmbreGoulotte;

      cv.Mat affineMatDir = cv.getAffineTransform(
        cv.VecPoint.fromList([cv.Point(0, 0), cv.Point(10, 0), cv.Point(0, 10)]),
        cv.VecPoint.fromList([
          cv.Point(shiftXG.toInt(), shiftYG.toInt()), 
          cv.Point(10 + shiftXG.toInt(), shiftYG.toInt()), 
          cv.Point(shiftXG.toInt(), 10 + shiftYG.toInt())
        ]) // Décalage adaptatif de l'ombre
      );
      cv.Mat shadowWarped = cv.warpAffine(maskBinaire, affineMatDir, (w, h));
      
      int baseBlurGoulotte = (5 + (ratioVolumeGoulotte * 4) + ((1.0 - ratioContrasteGoulotte) * 8)).toInt();
      if (baseBlurGoulotte % 2 == 0) baseBlurGoulotte += 1;
      
      cv.Mat shadowBlurred = cv.gaussianBlur(shadowWarped, (baseBlurGoulotte, baseBlurGoulotte), 0.0);
      
      // Assombrissement progressif de l'image de fond
      double opaciteOmbre = 0.35 * ratioContrasteGoulotte;
      cv.Mat ombre8u = shadowBlurred.convertTo(cv.MatType.CV_8UC1, alpha: opaciteOmbre); // Opacité de l'ombre adaptative
      cv.Mat invOmbre8u = cv.bitwiseNOT(ombre8u);
      cv.Mat invOmbre3c = cv.cvtColor(invOmbre8u, cv.COLOR_GRAY2BGR);
      
      // OPTIMISATION RAM : Application directe de l'ombre sans matrice 32F via l'argument scale
      cv.Mat murOmbre = cv.multiply(fondMat, invOmbre3c, scale: 1.0 / 255.0);

      // 4. Teinte et Lumière dynamiques (Même traitement que pour l'équipement)
      // CORRECTION MAJEURE : On utilise 'fondMat' (le mur vierge) et non 'murOmbre' !
      // Si on utilise 'murOmbre', la goulotte analyse sa PROPRE ombre et s'assombrit elle-même.
      cv.Mat murUltraSmall = cv.resize(fondMat, (w ~/ 32, h ~/ 32), interpolation: cv.INTER_AREA);
      cv.Mat murUltraFlou = cv.gaussianBlur(murUltraSmall, (15, 15), 0.0);
      cv.Mat murLisse = cv.resize(murUltraFlou, (w, h), interpolation: cv.INTER_CUBIC);

      cv.Mat grayLisse = cv.cvtColor(murLisse, cv.COLOR_BGR2GRAY);
      
      cv.Scalar meanMurSousGoulotte = cv.mean(murLisse, mask: maskBinaire);
      double bMur = meanMurSousGoulotte.val[0];
      double gMur = meanMurSousGoulotte.val[1];
      double rMur = meanMurSousGoulotte.val[2];
      double lumMurLocal = (0.114 * bMur) + (0.587 * gMur) + (0.299 * rMur);
      lumMurLocal = math.max(lumMurLocal, 1.0);

      // Création du filtre de Teinte
      double tintB = bMur / lumMurLocal;
      double tintG = gMur / lumMurLocal;
      double tintR = rMur / lumMurLocal;

      // La goulotte blanche absorbe la teinte de la pièce
      double forceTeinteGoulotte = 0.20;
      tintB = 1.0 + (tintB - 1.0) * forceTeinteGoulotte;
      tintG = 1.0 + (tintG - 1.0) * forceTeinteGoulotte;
      tintR = 1.0 + (tintR - 1.0) * forceTeinteGoulotte;

      // OPTIMISATION TEINTE : Split des canaux et ajustement 8-bit ultra-rapide (Zéro Float Matrix)
      var goulotteChannels = cv.split(goulotteBgrSmooth); // On utilise la version smooth !
      cv.Mat bTinted = goulotteChannels[0].convertTo(cv.MatType.CV_8UC1, alpha: tintB);
      cv.Mat gTinted = goulotteChannels[1].convertTo(cv.MatType.CV_8UC1, alpha: tintG);
      cv.Mat rTinted = goulotteChannels[2].convertTo(cv.MatType.CV_8UC1, alpha: tintR);
      cv.Mat goulotteTinted = cv.merge(cv.VecMat.fromList([bTinted, gTinted, rTinted]));

      // Ajustement de l'exposition (HSV)
      cv.Mat goulotteHsv = cv.cvtColor(goulotteTinted, cv.COLOR_BGR2HSV);
      var hsvChannels = cv.split(goulotteHsv);
      
      cv.Scalar meanGoulotteV = cv.mean(hsvChannels[2], mask: maskBinaire);
      double lumaGoulotteNative = math.max(meanGoulotteV.val[0], 1.0);

      // POUR REGLER LA LUMINOSITE (Influence du mur)
      // Si la goulotte est trop sombre sur un mur foncé, baisse cette variable vers 0.30 ou 0.20
      // pour que la goulotte "ignore" l'obscurité du mur en dessous d'elle.
      double influenceMurGoulotte = 0.43; 
      
      // OPTIMISATION LUMINOSITÉ : On combine la conversion 32F, l'alpha et le beta en 1 seule ligne !
      cv.Mat ratioSecurise = grayLisse.convertTo(cv.MatType.CV_32FC1, alpha: influenceMurGoulotte / lumaGoulotteNative, beta: 1.0 - influenceMurGoulotte);
      
      cv.Mat vChannelF = hsvChannels[2].convertTo(cv.MatType.CV_32FC1);
      cv.Mat vShadowedF = cv.multiply(vChannelF, ratioSecurise);

      // LUMINOSITE ADAPTATIVE POUR LA GOULOTTE
      // Comme pour l'équipement, on adapte l'exposition de la goulotte selon la lumière de la pièce.
      // Si le mur derrière la goulotte est sombre (ex: à l'ombre, 80/255), on baisse l'intensité lumineuse de la goulotte.
      double ratioLuminosite = (lumMurLocal / 128.0).clamp(0.65, 1.0);
      double luminositeGoulotteAdaptive = 0.95 * ratioLuminosite;

      // Contre-jour (Voile atmosphérique) calqué sur le contraste environnant
      cv.Rect rectGoulotte = cv.boundingRect(cv.VecPoint.fromList([p1, p2]));
      int rx = math.max(0, rectGoulotte.x - 20);
      int ry = math.max(0, rectGoulotte.y - 20);
      int rw = math.min(w - rx, rectGoulotte.width + 40);
      int rh = math.min(h - ry, rectGoulotte.height + 40);

      cv.Mat grayMurTextureBrute = cv.cvtColor(fondMat, cv.COLOR_BGR2GRAY); // Ici on prend la vraie texture (sans ombre) pour le contraste
      double voileAtmospherique = 0.0;
      if (rw > 0 && rh > 0) {
        cv.Mat roiMurGray = grayMurTextureBrute.region(cv.Rect(rx, ry, rw, rh));
        var minMaxRoi = cv.minMaxLoc(roiMurGray);
        double contrasteLocal = math.max(0.0, minMaxRoi.$2 - minMaxRoi.$1);
        voileAtmospherique = math.min(contrasteLocal * 0.15, 25.0);
      }
      
      double ratioLift = (255.0 - voileAtmospherique) / 255.0;
      
      // OPTIMISATION VOILE : Opération combinée (Luminosité globale + Contraste + Voile) en une passe
      cv.Mat vLiftedF = vShadowedF.convertTo(cv.MatType.CV_32FC1, alpha: luminositeGoulotteAdaptive * ratioLift, beta: voileAtmospherique);
      cv.Mat vCappedF = cv.threshold(vLiftedF, 245.0, 245.0, cv.THRESH_TRUNC).$2;
      hsvChannels[2] = vCappedF.convertTo(cv.MatType.CV_8UC1);

      cv.Mat goulotteHsvFinal = cv.merge(hsvChannels);
      cv.Mat goulotteRgbFinalPropre = cv.cvtColor(goulotteHsvFinal, cv.COLOR_HSV2BGR);

      // 5. Dégradation photographique
      // Flou pour enlever l'aspect "image de synthèse"
      cv.Mat goulotteBrouillee = cv.gaussianBlur(goulotteRgbFinalPropre, (3, 3), 0.6);
      
      // Simulation du grain ISO de la caméra
      // OPTIMISATION RAM : Génération du grain natif en 8 bits (Zéro 32F)
      cv.Mat noise = cv.Mat.zeros(h, w, cv.MatType.CV_8UC3);
      cv.randn(noise, cv.Scalar.all(128.0), cv.Scalar.all(5.0)); 
      
      // L'offset -128.0 restaure l'équilibre colorimétrique instantanément
      cv.Mat goulotteRgbFinal = cv.addWeighted(goulotteBrouillee, 1.0, noise, 1.0, -128.0);

      // 6. Fusion Finale (Alpha Blending)
      // Léger flou sur le masque binaire pour un anti-aliasing parfait des bords
      cv.Mat alphaMask = cv.gaussianBlur(maskBinaire, (3, 3), 0.0);
      cv.Mat alpha3_8u = cv.cvtColor(alphaMask, cv.COLOR_GRAY2BGR);
      cv.Mat invAlpha3_8u = cv.bitwiseNOT(alpha3_8u);

      // Fusion Alpha entièrement en 8 bits avec paramètre d'échelle
      cv.Mat fgBlended = cv.multiply(goulotteRgbFinal, alpha3_8u, scale: 1.0 / 255.0);
      cv.Mat bgBlended = cv.multiply(murOmbre, invAlpha3_8u, scale: 1.0 / 255.0);
      cv.Mat resultatFinal = cv.add(fgBlended, bgBlended);

      var encodeResult = cv.imencode('.jpg', resultatFinal);
      return encodeResult.$2;
      
    } catch (e) {
      print("[OpenCV] Erreur lors de l'incrustation de la goulotte : $e");
      return null;
    }
  }
}