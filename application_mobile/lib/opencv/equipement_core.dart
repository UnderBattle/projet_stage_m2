import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;

class EquipementCore {
  /// Isolate pour générer l'équipement sur fond transparent (PNG).
  static Future<Uint8List?> genererCalqueEquipementIsolate(Map<String, dynamic> params) async {
    return await genererCalqueEquipement(
      fondPropreBytes: params['fondPropreBytes'] as Uint8List,
      equipementBytes: params['equipementBytes'] as Uint8List,
      pointsIA: (params['pointsIA'] as List).map((e) => Map<String, double>.from(e)).toList(),
      decalageX: params['decalageX'] as double,
      decalageY: params['decalageY'] as double,
      equipementAssetPath: params['equipementAssetPath'] as String,
      profondeurMm: params['profondeurMm'] as double,
      hauteurMm: params['hauteurMm'] as double,
      largeurMm: params['largeurMm'] as double,
    );
  }

  /// Calcule l'équipement, ses couleurs et ses ombres portées et renvoie le tout 
  /// sous forme d'image PNG TRANSPARENTE prête à être superposée n'importe où !
  static Future<Uint8List?> genererCalqueEquipement({
    required Uint8List fondPropreBytes, 
    required Uint8List equipementBytes,
    required List<Map<String, double>> pointsIA,
    double decalageX = 0.0,
    double decalageY = 0.0, 
    required String equipementAssetPath,
    required double profondeurMm, 
    required double hauteurMm,
    required double largeurMm,
  }) async {
    try {
      // Décode l'image du mur nettoyé pour récupérer la texture et la couleur de la pièce
      cv.Mat murMat = cv.imdecode(fondPropreBytes, cv.IMREAD_COLOR);
      int wMur = murMat.cols;
      int hMur = murMat.rows;

      // =========================================================================
      // === SOLUTION DE ROGNAGE (PADDING VIRTUEL) ===
      // On agrandit l'espace de calcul de la 3D pour que l'équipement puisse 
      // déborder hors de l'écran sans que ses contours 3D ne soient coupés net.
      // =========================================================================
      int pad = 300;
      int wPad = wMur + pad * 2;
      int hPad = hMur + pad * 2;

      double ratioX = wMur / 1024.0;
      double ratioY = hMur / 1024.0;

      // Les points d'origine sont calculés pour correspondre à l'image non-padée
      List<cv.Point> ptsOri = pointsIA.map((pt) {
        return cv.Point((pt['x']! * ratioX).toInt(), (pt['y']! * ratioY).toInt());
      }).toList();

      cv.Mat resultImg = murMat.clone(); 

      // =========================================================================
      // === PHASE PRE-CALCUL : OPTIMISATION DES ASSETS AVANT TRAITEMENT ===
      // =========================================================================
      cv.Mat equipementMat = cv.imdecode(equipementBytes, cv.IMREAD_UNCHANGED);
      
      var rawChannels = cv.split(equipementMat);
      cv.Mat rawAlpha = cv.threshold(rawChannels[3], 10, 255, cv.THRESH_BINARY).$2;
      cv.Mat rawBgr = cv.cvtColor(equipementMat, cv.COLOR_BGRA2BGR);
      cv.Scalar rawMeanColor = cv.mean(rawBgr, mask: rawAlpha);
      
      double lumaNativeEquipement = (0.114 * rawMeanColor.val[0]) + (0.587 * rawMeanColor.val[1]) + (0.299 * rawMeanColor.val[2]);
      bool estEquipementNoir = lumaNativeEquipement < 80.0;

      // =========================================================================
      // === PHASE 2 : CALCUL DE LA PERSPECTIVE STABILISEE ===
      // =========================================================================
      cv.Point ptHg = ptsOri[0];
      cv.Point ptHd = ptsOri[1];
      double dx = (ptHd.x - ptHg.x).toDouble();
      double dy = (ptHd.y - ptHg.y).toDouble();
      double largeurPx = math.sqrt(dx * dx + dy * dy);
      double angleRad = math.atan2(dy, dx);

      double hAutoMm = 100.0;
      double wAutoMm = 50.0; 
      double ratioPhysique = hAutoMm / wAutoMm; 
      double hauteurPx = largeurPx * ratioPhysique;

      double ux = largeurPx * math.cos(angleRad);
      double uy = largeurPx * math.sin(angleRad);
      double vx = -hauteurPx * math.sin(angleRad);
      double vy = hauteurPx * math.cos(angleRad);

      List<cv.Point> ptsDstLisses = [
        cv.Point((ptHg.x + decalageX).toInt(), (ptHg.y + decalageY).toInt()),
        cv.Point((ptHg.x + ux + decalageX).toInt(), (ptHg.y + uy + decalageY).toInt()),
        cv.Point((ptHg.x + ux + vx + decalageX).toInt(), (ptHg.y + uy + vy + decalageY).toInt()),
        cv.Point((ptHg.x + vx + decalageX).toInt(), (ptHg.y + vy + decalageY).toInt())
      ];

      // Translation des points vers le "Canvas Virtuel" (Padded)
      List<cv.Point> ptsDstLissesPad = ptsDstLisses.map((p) => cv.Point(p.x + pad, p.y + pad)).toList();

      // =========================================================================
      // === PHASE 3 : TAILLE REELLE ET DEFORMATION 3D SUR CANVAS PADDED ===
      // =========================================================================
      double hEquipementMm = hauteurMm;
      double wEquipementMm = largeurMm; 
      int wImgEquipement = equipementMat.cols;
      int hImgEquipement = equipementMat.rows;

      double wAutoPx = (wAutoMm / wEquipementMm) * wImgEquipement;
      double hAutoPx = (hAutoMm / hEquipementMm) * hImgEquipement;

      List<cv.Point> ptsSrc = [
        cv.Point(0, 0),
        cv.Point(wAutoPx.toInt(), 0),
        cv.Point(wAutoPx.toInt(), hAutoPx.toInt()),
        cv.Point(0, hAutoPx.toInt())
      ];

      var vecPtsSrc = cv.VecPoint.fromList(ptsSrc);
      var vecPtsDstPad = cv.VecPoint.fromList(ptsDstLissesPad);
      cv.Mat hMatrixPad = cv.getPerspectiveTransform(vecPtsSrc, vecPtsDstPad);
      
      // La projection se fait sur le GRAND canvas (wPad, hPad) pour ne rien couper
      cv.Mat equipementWarpedPad = cv.warpPerspective(equipementMat, hMatrixPad, (wPad, hPad));

      var channelsPad = cv.split(equipementWarpedPad);
      cv.Mat alphaMaskOriginalePad = channelsPad[3]; 

      cv.Mat alphaBinairePad = cv.threshold(alphaMaskOriginalePad, 127, 255, cv.THRESH_BINARY).$2;
      cv.Mat kernelErode = cv.Mat.ones(3, 3, cv.MatType.CV_8UC1);
      cv.Mat alphaErodePad = cv.erode(alphaBinairePad, kernelErode);
      cv.Mat alphaMaskPad = cv.gaussianBlur(alphaErodePad, (3, 3), 0.0);
      
      cv.Mat equipementBgrPad = cv.cvtColor(equipementWarpedPad, cv.COLOR_BGRA2BGR);
      cv.Mat maskBinairePad = cv.threshold(alphaMaskPad, 5, 255, cv.THRESH_BINARY).$2;

      // =========================================================================
      // === PHASE 3.5 : GÉNÉRATION DU VOLUME 3D (EXTRUSION EXACTE DU PNG) ===
      // =========================================================================
      var momentsPad = cv.moments(maskBinairePad);
      double eqCXPad = momentsPad.m10 / (momentsPad.m00 + 0.0001);
      double eqCYPad = momentsPad.m01 / (momentsPad.m00 + 0.0001);
      
      // Le vecteur de fuite part du VRAI centre optique de la caméra 
      // (décalé du padding pour correspondre au centre physique de la photo)
      double imgCXPad = (wMur / 2.0) + pad;
      double imgCYPad = (hMur / 2.0) + pad;
      
      double vecX = imgCXPad - eqCXPad;
      double vecY = imgCYPad - eqCYPad;
      
      // L'extrusion dépend de la profondeur physique
      double baseExtrusion = (profondeurMm / 1000.0) * 0.14; 
      
      double ratioForme = hauteurMm / largeurMm;
      if (ratioForme < 0.45) {
          // Si c'est une clim murale (ratio très écrasé), on diminue l'extrusion 3D 
          baseExtrusion *= 0.45;
      }

      int dxBack = (vecX * baseExtrusion).toInt();
      int dyBack = (vecY * baseExtrusion).toInt();
      
      cv.Mat sidesBgrPad = cv.Mat.zeros(hPad, wPad, cv.MatType.CV_8UC3);
      cv.Mat sidesAlphaPad = cv.Mat.zeros(hPad, wPad, cv.MatType.CV_8UC1);
      
      cv.Scalar baseColorEq = cv.mean(equipementBgrPad, mask: maskBinairePad);
      double mB = baseColorEq.val[0];
      double mG = baseColorEq.val[1];
      double mR = baseColorEq.val[2];
      
      cv.Scalar colorTop = cv.Scalar(mB, mG, mR, 0);
      cv.Scalar colorBottom = cv.Scalar(mB * 0.60, mG * 0.60, mR * 0.60, 0); 
      cv.Scalar colorLeft = cv.Scalar(mB * 0.89, mG * 0.89, mR * 0.89, 0);
      cv.Scalar colorRight = cv.Scalar(mB * 0.89, mG * 0.89, mR * 0.89, 0);

      // Dessin du "fond" du volume 3D
      cv.Mat affineBackPad = cv.getAffineTransform(
        cv.VecPoint.fromList([cv.Point(0, 0), cv.Point(10, 0), cv.Point(0, 10)]),
        cv.VecPoint.fromList([cv.Point(dxBack, dyBack), cv.Point(10 + dxBack, dyBack), cv.Point(dxBack, 10 + dyBack)])
      );
      cv.Mat backMaskAlphaPad = cv.warpAffine(maskBinairePad, affineBackPad, (wPad, hPad));
      cv.Mat backCapBgrPad = cv.Mat.zeros(hPad, wPad, cv.MatType.CV_8UC3)..setTo(colorBottom); 
      backCapBgrPad.copyTo(sidesBgrPad, mask: backMaskAlphaPad);
      cv.bitwiseOR(sidesAlphaPad, backMaskAlphaPad, dst: sidesAlphaPad);

      // Murs latéraux en retrait (érosion)
      cv.Mat kernelShrink = cv.Mat.ones(5, 5, cv.MatType.CV_8UC1);
      cv.Mat maskBinaireShrunkPad = cv.erode(maskBinairePad, kernelShrink);

      var contoursResult = cv.findContours(maskBinaireShrunkPad, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      var contours = contoursResult.$1;

      for (int c = 0; c < contours.length; c++) {
        var contour = contours[c];
        
        List<cv.Point> ptList = contour.toList();
        int numPts = ptList.length;

        for (int i = 0; i < numPts; i++) {
          cv.Point p1 = ptList[i];
          cv.Point p2 = ptList[(i + 1) % numPts];

          cv.Point p1Back = cv.Point(p1.x + dxBack, p1.y + dyBack);
          cv.Point p2Back = cv.Point(p2.x + dxBack, p2.y + dyBack);

          double midX = (p1.x + p2.x) / 2.0;
          double midY = (p1.y + p2.y) / 2.0;

          double radX = midX - eqCXPad;
          double radY = midY - eqCYPad;
          
          double segDx = (p2.x - p1.x).toDouble();
          double segDy = (p2.y - p1.y).toDouble();
          
          double nx = -segDy;
          double ny = segDx;
          
          if ((nx * radX + ny * radY) < 0) {
            nx = -nx;
            ny = -ny;
          }

          double len = math.sqrt(nx * nx + ny * ny) + 0.0001;
          nx /= len;
          ny /= len;

          cv.Scalar faceColor;
          if (ny.abs() > nx.abs()) {
            if (ny < 0) {
              faceColor = colorTop;
            } else {
              faceColor = colorBottom;
            }
          } else {
            if (nx < 0) {
              faceColor = colorLeft;
            } else {
              faceColor = colorRight;
            }
          }

          var quad = [p1, p2, p2Back, p1Back];
          cv.fillPoly(sidesBgrPad, cv.VecVecPoint.fromList([quad]), faceColor, lineType: cv.LINE_AA);
          cv.fillPoly(sidesAlphaPad, cv.VecVecPoint.fromList([quad]), cv.Scalar.all(255), lineType: cv.LINE_AA);
        }
      }

      cv.Mat sidesBgrSmoothPad = cv.gaussianBlur(sidesBgrPad, (15, 15), 0.0);
      cv.Mat sidesAlphaSmoothPad = cv.gaussianBlur(sidesAlphaPad, (3, 3), 0.0);
      
      cv.Mat maskTotalVolumePad = cv.add(maskBinairePad, sidesAlphaPad);
      cv.Mat alphaMaskTotalPad = cv.add(alphaMaskPad, sidesAlphaSmoothPad);

      cv.Mat equipement3DBgrPad = sidesBgrSmoothPad.clone();
      equipementBgrPad.copyTo(equipement3DBgrPad, mask: maskBinairePad);

      // =========================================================================
      // === PHASE 4 : CALCUL DE LA DIRECTION DE LA LUMIERE ET OMBRE PROGRESSIVE ===
      // =========================================================================
      // Le calcul de la lumière de la pièce se fait sur l'image ORIGINALE (non paddée)
      cv.Mat grayMur = cv.cvtColor(resultImg, cv.COLOR_BGR2GRAY);
      int downscaleSobel = 32;
      cv.Mat grayMurSmall = cv.resize(grayMur, (wMur ~/ downscaleSobel, hMur ~/ downscaleSobel));
      
      // On découpe temporairement le masque pour qu'il matche la taille du mur
      cv.Mat maskTotalVolumeOrig = maskTotalVolumePad.region(cv.Rect(pad, pad, wMur, hMur));
      cv.Mat maskBinaireSmall = cv.resize(maskTotalVolumeOrig, (wMur ~/ downscaleSobel, hMur ~/ downscaleSobel));

      var meanStdDev = cv.meanStdDev(grayMurSmall);
      double lumiereMoyenneMur = meanStdDev.$1.val[0];
      double ecartTypeMur = meanStdDev.$2.val[0]; 

      double ratioContraste = (ecartTypeMur / 50.0).clamp(0.2, 1.2);

      final double reglageOmbreDirBase = 0.12 * ratioContraste;
      final double reglageOmbreContact = 0.25 * ratioContraste;

      cv.Mat grayMurFlou = cv.gaussianBlur(grayMurSmall, (7, 7), 0.0);
      cv.Mat sobelX = cv.sobel(grayMurFlou, cv.MatType.CV_32F, 1, 0, ksize: 3);
      cv.Mat sobelY = cv.sobel(grayMurFlou, cv.MatType.CV_32F, 0, 1, ksize: 3);

      cv.Scalar meanSobelX = cv.mean(sobelX, mask: maskBinaireSmall);
      cv.Scalar meanSobelY = cv.mean(sobelY, mask: maskBinaireSmall);
      double gradX = meanSobelX.val[0];
      double gradY = meanSobelY.val[0];
      double norme = math.sqrt(gradX * gradX + gradY * gradY) + 0.0001; 

      double dirLumiereX = gradX / norme;
      double dirLumiereY = gradY / norme;

      double ratioVolume = profondeurMm / 100.0; 
      double forceOmbre = 12.0 * ratioVolume; 
      double longueurOmbre = forceOmbre * ratioContraste;

      double shiftX = norme < 1.0 ? (3.0 * ratioVolume) : -dirLumiereX * longueurOmbre;
      double shiftY = norme < 1.0 ? (8.0 * ratioVolume) : -dirLumiereY * longueurOmbre;

      // Création de l'ombre dans l'espace "Padded" pour ne pas la couper si elle sort de l'écran
      var srcPts = cv.VecPoint.fromList([cv.Point(0, 0), cv.Point(10, 0), cv.Point(0, 10)]);
      var dstPtsDir = cv.VecPoint.fromList([cv.Point(shiftX.toInt(), shiftY.toInt()), cv.Point(10 + shiftX.toInt(), shiftY.toInt()), cv.Point(shiftX.toInt(), 10 + shiftY.toInt())]);
      cv.Mat affineMatDir = cv.getAffineTransform(srcPts, dstPtsDir);
      
      cv.Mat alphaOmbrePad = cv.warpAffine(alphaMaskTotalPad, affineMatDir, (wPad, hPad));
      cv.Mat smallAlphaPad = cv.resize(alphaOmbrePad, (wPad ~/ 4, hPad ~/ 4));
      
      int baseBlur = (5 + (ratioVolume * 4) + ((1.0 - ratioContraste) * 8)).toInt();
      if (baseBlur % 2 == 0) baseBlur += 1; 
      
      cv.Mat smallOmbreFlouePad = cv.gaussianBlur(smallAlphaPad, (baseBlur, baseBlur), 0.0);
      cv.Mat ombreFloueDirectionnellePad = cv.resize(smallOmbreFlouePad, (wPad, hPad), interpolation: cv.INTER_CUBIC);

      var dstPtsContact = cv.VecPoint.fromList([cv.Point(0, 3), cv.Point(10, 3), cv.Point(0, 13)]);
      cv.Mat affineMatContact = cv.getAffineTransform(srcPts, dstPtsContact);
      cv.Mat alphaContactPad = cv.warpAffine(alphaMaskTotalPad, affineMatContact, (wPad, hPad));
      
      cv.Mat smallContactPad = cv.resize(alphaContactPad, (wPad ~/ 4, hPad ~/ 4));
      cv.Mat smallContactFlouPad = cv.gaussianBlur(smallContactPad, (3, 3), 0.0);
      cv.Mat ombreFloueContactPad = cv.resize(smallContactFlouPad, (wPad, hPad), interpolation: cv.INTER_CUBIC);

      cv.Mat ombreDir8uPad = ombreFloueDirectionnellePad.convertTo(cv.MatType.CV_8UC1, alpha: reglageOmbreDirBase);
      cv.Mat ombreContact8uPad = ombreFloueContactPad.convertTo(cv.MatType.CV_8UC1, alpha: reglageOmbreContact);
      
      cv.Mat ombreTotalePad = cv.add(ombreDir8uPad, ombreContact8uPad);

      // =========================================================================
      // === LE GRAND DÉCOUPAGE (RETOUR À LA TAILLE DE L'ÉCRAN) ===
      // On retire la marge de 300px. L'image a pu déborder sans être "écrasée" 
      // et elle est maintenant recadrée parfaitement.
      // =========================================================================
      cv.Rect cropRect = cv.Rect(pad, pad, wMur, hMur);
      
      cv.Mat equipementBgr = equipement3DBgrPad.region(cropRect).clone();
      cv.Mat maskTotalVolume = maskTotalVolumePad.region(cropRect).clone();
      cv.Mat alphaMaskTotal = alphaMaskTotalPad.region(cropRect).clone();
      cv.Mat ombreTotale = ombreTotalePad.region(cropRect).clone();
      maskBinairePad.region(cropRect).clone();

      // =========================================================================
      // === PHASE 5 : LUMIERE ET TEMPERATURE DE COULEUR INTELLIGENTE ===
      // =========================================================================
      final double reglageMixAmbiancePiece = 0.65;
      final double reglageMixCouleurMur = 0.35;    
      final double reglageTeinteEquipementBlanc = 0.30;
      final double reglageTeinteEquipementNoir = 0.08;   
      
      final double reglageInfluenceOmbreSurBlanc = 0.65;
      final double reglageInfluenceOmbreSurNoir = 0.21;  
      
      double ratioLuminositeAmbiante = (lumiereMoyenneMur / 128.0).clamp(0.7, 1.0);
      final double reglageLuminositeEquipementBlanc = 0.95 * ratioLuminositeAmbiante; 
      final double reglageLuminositeEquipementNoir = 0.78;   

      cv.Mat murUltraSmall = cv.resize(resultImg, (wMur ~/ 32, hMur ~/ 32), interpolation: cv.INTER_AREA);
      cv.Mat murUltraFlou = cv.gaussianBlur(murUltraSmall, (15, 15), 0.0);
      cv.Mat murLisse = cv.resize(murUltraFlou, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      cv.Scalar meanMurGlobal = cv.mean(murLisse);
      cv.Scalar meanMurSousEquipement = cv.mean(murLisse, mask: maskTotalVolume);

      double bMur = (meanMurGlobal.val[0] * reglageMixAmbiancePiece) + (meanMurSousEquipement.val[0] * reglageMixCouleurMur);
      double gMur = (meanMurGlobal.val[1] * reglageMixAmbiancePiece) + (meanMurSousEquipement.val[1] * reglageMixCouleurMur);
      double rMur = (meanMurGlobal.val[2] * reglageMixAmbiancePiece) + (meanMurSousEquipement.val[2] * reglageMixCouleurMur);

      double lumMurLocal = (0.114 * bMur) + (0.587 * gMur) + (0.299 * rMur);
      lumMurLocal = math.max(lumMurLocal, 1.0); 

      double tintB = bMur / lumMurLocal;
      double tintG = gMur / lumMurLocal;
      double tintR = rMur / lumMurLocal;

      double forceTeinte = estEquipementNoir ? reglageTeinteEquipementNoir : reglageTeinteEquipementBlanc; 
      tintB = 1.0 + (tintB - 1.0) * forceTeinte;
      tintG = 1.0 + (tintG - 1.0) * forceTeinte;
      tintR = 1.0 + (tintR - 1.0) * forceTeinte;

      tintB = math.max(0.80, math.min(1.20, tintB));
      tintG = math.max(0.80, math.min(1.20, tintG));
      tintR = math.max(0.80, math.min(1.20, tintR));

      cv.Mat tintMat = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC3)..setTo(cv.Scalar(tintB, tintG, tintR, 0));
      cv.Mat equipementF = equipementBgr.convertTo(cv.MatType.CV_32FC3);
      cv.Mat equipementTintedF = cv.multiply(equipementF, tintMat);
      cv.Mat equipementTinted = equipementTintedF.convertTo(cv.MatType.CV_8UC3);

      cv.Mat equipementHsv = cv.cvtColor(equipementTinted, cv.COLOR_BGR2HSV);
      var hsvChannels = cv.split(equipementHsv);
      
      cv.Mat grayLisse = cv.cvtColor(murLisse, cv.COLOR_BGR2GRAY);
      cv.Mat grayLisseF = grayLisse.convertTo(cv.MatType.CV_32FC1);

      cv.Scalar meanEquipementV = cv.mean(hsvChannels[2], mask: maskTotalVolume);
      double lumaEquipementNativeHSV = math.max(meanEquipementV.val[0], 1.0);

      double denominateurLuma = estEquipementNoir ? math.max(lumaEquipementNativeHSV, 130.0) : lumaEquipementNativeHSV;

      cv.Mat ratioMap = grayLisseF.convertTo(cv.MatType.CV_32FC1, alpha: 1.0 / denominateurLuma);
      cv.Mat matriceUn = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC1)..setTo(cv.Scalar.all(1.0));
      
      double influenceMur = estEquipementNoir ? reglageInfluenceOmbreSurNoir : reglageInfluenceOmbreSurBlanc; 
      double influenceEquipement = 1.0 - influenceMur;
      
      cv.Mat ratioSecurise = cv.addWeighted(ratioMap, influenceMur, matriceUn, influenceEquipement, 0.0);
      cv.Mat vChannelF = hsvChannels[2].convertTo(cv.MatType.CV_32FC1);
      cv.Mat vShadowedF = cv.multiply(vChannelF, ratioSecurise);

      if (estEquipementNoir) {
         vShadowedF = vShadowedF.convertTo(cv.MatType.CV_32FC1, alpha: reglageLuminositeEquipementNoir);
      } else {
         vShadowedF = vShadowedF.convertTo(cv.MatType.CV_32FC1, alpha: reglageLuminositeEquipementBlanc); 
      }

      cv.Rect rectEquipement = cv.boundingRect(cv.VecPoint.fromList(ptsDstLisses));
      int rx = math.max(0, rectEquipement.x - 20);
      int ry = math.max(0, rectEquipement.y - 20);
      int rw = math.min(wMur - rx, rectEquipement.width + 40);
      int rh = math.min(hMur - ry, rectEquipement.height + 40);
      
      double voileAtmospherique = 0.0;
      
      if (rw > 0 && rh > 0) {
        cv.Mat roiMurGray = grayMur.region(cv.Rect(rx, ry, rw, rh));
        var minMaxRoi = cv.minMaxLoc(roiMurGray);
        double contrasteLocal = math.max(0.0, minMaxRoi.$2 - minMaxRoi.$1);
        voileAtmospherique = math.min(contrasteLocal * 0.15, 25.0);
      }
      
      double ratioLift = (255.0 - voileAtmospherique) / 255.0;
      cv.Mat vLiftedF = cv.addWeighted(vShadowedF, ratioLift, vShadowedF, 0.0, voileAtmospherique);
      cv.Mat vCappedF = cv.threshold(vLiftedF, 245.0, 245.0, cv.THRESH_TRUNC).$2;
      hsvChannels[2] = vCappedF.convertTo(cv.MatType.CV_8UC1);

      cv.Mat equipementHsvFinal = cv.merge(hsvChannels);
      cv.Mat equipementRgbFinalPropre = cv.cvtColor(equipementHsvFinal, cv.COLOR_HSV2BGR);

      // =========================================================================
      // === PHASE 5.5 : DEGRADATION REALISTE (CAPTEUR PHOTO) ===
      // =========================================================================
      cv.Mat equipementBrouillee = cv.gaussianBlur(equipementRgbFinalPropre, (3, 3), 0.6);

      cv.Mat equipementFloat = equipementBrouillee.convertTo(cv.MatType.CV_32FC3);
      cv.Mat noise = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC3);
      cv.randn(noise, cv.Scalar.all(0.0), cv.Scalar.all(8.0)); 
      cv.Mat equipementNoisyFloat = cv.add(equipementFloat, noise);
      
      cv.Mat equipementRgbFinal = equipementNoisyFloat.convertTo(cv.MatType.CV_8UC3);

      // =========================================================================
      // === PHASE 6 : CREATION DU CALQUE TRANSPARENT ===
      // =========================================================================
      cv.Mat alphaMaskF = alphaMaskTotal.convertTo(cv.MatType.CV_32FC1, alpha: 1.0 / 255.0);
      cv.Mat ombreF = ombreTotale.convertTo(cv.MatType.CV_32FC1, alpha: 1.0 / 255.0);
      
      cv.Mat matriceUnAlpha = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC1)..setTo(cv.Scalar.all(1.0));
      cv.Mat invAlphaMaskF = cv.subtract(matriceUnAlpha, alphaMaskF);
      cv.Mat shadowAlphaF = cv.multiply(invAlphaMaskF, ombreF);
      cv.Mat finalAlphaF = cv.add(alphaMaskF, shadowAlphaF);
      
      cv.Mat finalAlpha8u = finalAlphaF.convertTo(cv.MatType.CV_8UC1, alpha: 255.0);

      cv.Mat bgrFinal = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC3);
      equipementRgbFinal.copyTo(bgrFinal, mask: alphaMaskTotal);

      cv.Mat bgraFinal = cv.cvtColor(bgrFinal, cv.COLOR_BGR2BGRA);
      var bgraChannels = cv.split(bgraFinal);
      bgraChannels[3] = finalAlpha8u; 
      cv.Mat finalImage = cv.merge(bgraChannels);

      var encodeResult = cv.imencode('.png', finalImage, params: cv.VecI32.fromList([16, 0]));
      return encodeResult.$2;
      
    } catch (e) {
      print("[OpenCV] ERREUR FATALE : $e");
      return null;
    }
  }
}