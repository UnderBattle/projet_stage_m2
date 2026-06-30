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
      cv.Mat murMat = cv.imdecode(fondPropreBytes, cv.IMREAD_COLOR);
      int wMur = murMat.cols;
      int hMur = murMat.rows;

      // =========================================================================
      // === SOLUTION DE ROGNAGE (PADDING VIRTUEL) ===
      // =========================================================================
      int pad = 300;
      int wPad = wMur + pad * 2;
      int hPad = hMur + pad * 2;

      double ratioX = wMur / 1024.0;
      double ratioY = hMur / 1024.0;

      List<cv.Point> ptsOri = pointsIA.map((pt) {
        return cv.Point((pt['x']! * ratioX).toInt(), (pt['y']! * ratioY).toInt());
      }).toList();

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
      // === PHASE 2 : CALCUL DE LA PERSPECTIVE STABILISEE ET ZOOM OPTIQUE ===
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
      double hauteurPxBase = largeurPx * ratioPhysique;

      // Calcul des vecteurs pour trouver le centre exact de l'autocollant
      double uxBase = largeurPx * math.cos(angleRad);
      double uyBase = largeurPx * math.sin(angleRad);
      double vxBase = -hauteurPxBase * math.sin(angleRad);
      double vyBase = hauteurPxBase * math.cos(angleRad);

      double centreX = ptHg.x + (uxBase + vxBase) / 2.0;
      double centreY = ptHg.y + (uyBase + vyBase) / 2.0;

      // NOUVEAU : Effet d'avancée optique (Zoom depuis le centre)
      double effetZoomProfondeur = 1.0 + (profondeurMm / 1000.0) * 0.60; 

      double largeurPxZoom = largeurPx * effetZoomProfondeur;
      double hauteurPxZoom = hauteurPxBase * effetZoomProfondeur;

      double uxZoom = largeurPxZoom * math.cos(angleRad);
      double uyZoom = largeurPxZoom * math.sin(angleRad);
      double vxZoom = -hauteurPxZoom * math.sin(angleRad);
      double vyZoom = hauteurPxZoom * math.cos(angleRad);

      // CORRECTION DU DÉCALAGE : Décalage correctif vers la gauche
      double ratioDecalageGauche = 0.50;
      double compensationMurX = -(uxBase * ratioDecalageGauche);
      double compensationMurY = -(uyBase * ratioDecalageGauche);

      // Nouveau point Haut-Gauche calculé depuis le centre, avec le décalage correctif inclus
      double nouveauPtHgX = centreX - (uxZoom + vxZoom) / 2.0 + compensationMurX;
      double nouveauPtHgY = centreY - (uyZoom + vyZoom) / 2.0 + compensationMurY;

      List<cv.Point> ptsDstLisses = [
        cv.Point((nouveauPtHgX + decalageX).toInt(), (nouveauPtHgY + decalageY).toInt()),
        cv.Point((nouveauPtHgX + uxZoom + decalageX).toInt(), (nouveauPtHgY + uyZoom + decalageY).toInt()),
        cv.Point((nouveauPtHgX + uxZoom + vxZoom + decalageX).toInt(), (nouveauPtHgY + uyZoom + vyZoom + decalageY).toInt()),
        cv.Point((nouveauPtHgX + vxZoom + decalageX).toInt(), (nouveauPtHgY + vyZoom + decalageY).toInt())
      ];

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
      
      cv.Mat equipementWarpedPad = cv.warpPerspective(equipementMat, hMatrixPad, (wPad, hPad));

      var channelsPad = cv.split(equipementWarpedPad);
      cv.Mat alphaMaskOriginalePad = channelsPad[3]; 
      cv.Mat alphaBinairePad = cv.threshold(alphaMaskOriginalePad, 127, 255, cv.THRESH_BINARY).$2;
      cv.Mat alphaErodePad = cv.erode(alphaBinairePad, cv.Mat.ones(3, 3, cv.MatType.CV_8UC1));
      cv.Mat alphaMaskPad = cv.gaussianBlur(alphaErodePad, (3, 3), 0.0); 
      cv.Mat equipementBgrPad = cv.cvtColor(equipementWarpedPad, cv.COLOR_BGRA2BGR);
      cv.Mat maskBinairePad = cv.threshold(alphaMaskPad, 5, 255, cv.THRESH_BINARY).$2;

      // =========================================================================
      // === PHASE 3.5 : GÉNÉRATION DU VOLUME (PERSPECTIVE DYNAMIQUE POINT PAR POINT) =
      // =========================================================================
      var momentsPad = cv.moments(maskBinairePad);
      double eqCXPad = momentsPad.m10 / (momentsPad.m00 + 0.0001);
      double eqCYPad = momentsPad.m01 / (momentsPad.m00 + 0.0001);
      
      double centrePhotoX = (wMur / 2.0) + pad;
      double centrePhotoY = (hMur / 2.0) + pad;

      double distanceX = (eqCXPad - centrePhotoX).abs() / (wMur / 2.0);
      double distanceY = (eqCYPad - centrePhotoY).abs() / (hMur / 2.0);

      double attenuationX = (1.0 - distanceX).clamp(0.0, 1.0);
      double attenuationY = (1.0 - distanceY).clamp(0.0, 1.0);

      double ratioForme = hauteurMm / largeurMm;
      bool estClimMurale = ratioForme < 0.45;

      double forcePerspectiveBasse = estClimMurale ? 0.08 : 0.25; 
      double forcePerspectiveLaterale = estClimMurale ? 0.04 : 0.05; 

      double dynamicShiftX = (wMur * forcePerspectiveLaterale) * attenuationX;
      double dynamicShiftY = (hMur * forcePerspectiveBasse) * attenuationY;

      double imgCXPad = centrePhotoX + dynamicShiftX;
      double imgCYPad = centrePhotoY + dynamicShiftY;
      
      double baseExtrusion = estClimMurale 
          ? (profondeurMm / 1000.0) * 0.18  
          : (profondeurMm / 1000.0) * 0.25; 

      cv.Mat sidesBgrPad = cv.Mat.zeros(hPad, wPad, cv.MatType.CV_8UC3);
      cv.Mat sidesAlphaPad = cv.Mat.zeros(hPad, wPad, cv.MatType.CV_8UC1);
      
      cv.Scalar baseColorEq = cv.mean(equipementBgrPad, mask: maskBinairePad);
      double mB = baseColorEq.val[0];
      double mG = baseColorEq.val[1];
      double mR = baseColorEq.val[2];
      
      // AMÉLIORATION RELIEF : Assombrissement asymétrique des faces latérales
      cv.Scalar colorTop = cv.Scalar(mB, mG, mR, 0);
      cv.Scalar colorBottom = estEquipementNoir 
          ? cv.Scalar(math.min(255.0, mB + 50), math.min(255.0, mG + 50), math.min(255.0, mR + 50), 0) 
          : cv.Scalar(mB * 0.67, mG * 0.67, mR * 0.67, 0); 
          
      cv.Scalar colorLeft = estEquipementNoir 
          ? cv.Scalar(math.min(255.0, mB + 35), math.min(255.0, mG + 35), math.min(255.0, mR + 35), 0)
          : cv.Scalar(mB * 0.85, mG * 0.85, mR * 0.85, 0);   
          
      cv.Scalar colorRight = estEquipementNoir 
          ? cv.Scalar(math.min(255.0, mB + 20), math.min(255.0, mG + 20), math.min(255.0, mR + 20), 0)
          : cv.Scalar(mB * 0.85, mG * 0.85, mR * 0.85, 0);

      double scaleS = 1.0 - baseExtrusion;
      double tx = imgCXPad * baseExtrusion;
      double ty = imgCYPad * baseExtrusion;

      cv.Mat affineBackPad = cv.getAffineTransform(
        cv.VecPoint.fromList([cv.Point(0, 0), cv.Point(10000, 0), cv.Point(0, 10000)]),
        cv.VecPoint.fromList([
          cv.Point(tx.toInt(), ty.toInt()), 
          cv.Point((10000 * scaleS + tx).toInt(), ty.toInt()), 
          cv.Point(tx.toInt(), (10000 * scaleS + ty).toInt())
        ])
      );
      
      cv.Mat backMaskAlphaPad = cv.warpAffine(maskBinairePad, affineBackPad, (wPad, hPad));
      cv.Mat backCapBgrPad = cv.Mat.zeros(hPad, wPad, cv.MatType.CV_8UC3)..setTo(colorBottom); 
      backCapBgrPad.copyTo(sidesBgrPad, mask: backMaskAlphaPad);
      cv.bitwiseOR(sidesAlphaPad, backMaskAlphaPad, dst: sidesAlphaPad);
      cv.Mat maskBinaireShrunkPad = cv.erode(maskBinairePad, cv.Mat.ones(5, 5, cv.MatType.CV_8UC1));

      var contoursResult = cv.findContours(maskBinaireShrunkPad, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
      var contours = contoursResult.$1;

      List<List<cv.Point>> quadsTop = [];
      List<List<cv.Point>> quadsBottom = [];
      List<List<cv.Point>> quadsLeft = [];
      List<List<cv.Point>> quadsRight = [];
      List<List<cv.Point>> quadsAll = [];

      for (int c = 0; c < contours.length; c++) {
        var contour = contours[c];
        
        List<cv.Point> ptList = contour.toList();
        int numPts = ptList.length;

        for (int i = 0; i < numPts; i++) {
          cv.Point p1 = ptList[i];
          cv.Point p2 = ptList[(i + 1) % numPts];

          cv.Point p1Back = cv.Point(
            (p1.x + (imgCXPad - p1.x) * baseExtrusion).toInt(),
            (p1.y + (imgCYPad - p1.y) * baseExtrusion).toInt(),
          );
          cv.Point p2Back = cv.Point(
            (p2.x + (imgCXPad - p2.x) * baseExtrusion).toInt(),
            (p2.y + (imgCYPad - p2.y) * baseExtrusion).toInt(),
          );

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

          var quad = [p1, p2, p2Back, p1Back];
          quadsAll.add(quad); 

          if (ny.abs() > nx.abs()) {
            if (ny < 0) {
              quadsTop.add(quad);
            } else {
              quadsBottom.add(quad);
            }
          } else {
            if (nx < 0) {
              quadsLeft.add(quad);
            } else {
              quadsRight.add(quad);
            }
          }
        }
      }

      if (quadsTop.isNotEmpty) cv.fillPoly(sidesBgrPad, cv.VecVecPoint.fromList(quadsTop), colorTop, lineType: cv.LINE_AA);
      if (quadsBottom.isNotEmpty) cv.fillPoly(sidesBgrPad, cv.VecVecPoint.fromList(quadsBottom), colorBottom, lineType: cv.LINE_AA);
      if (quadsLeft.isNotEmpty) cv.fillPoly(sidesBgrPad, cv.VecVecPoint.fromList(quadsLeft), colorLeft, lineType: cv.LINE_AA);
      if (quadsRight.isNotEmpty) cv.fillPoly(sidesBgrPad, cv.VecVecPoint.fromList(quadsRight), colorRight, lineType: cv.LINE_AA);
      
      if (quadsAll.isNotEmpty) cv.fillPoly(sidesAlphaPad, cv.VecVecPoint.fromList(quadsAll), cv.Scalar.all(255), lineType: cv.LINE_AA);

      cv.Mat sidesBgrSmoothPad = cv.gaussianBlur(sidesBgrPad, (15, 15), 0.0);
      cv.Mat sidesAlphaSmoothPad = cv.gaussianBlur(sidesAlphaPad, (3, 3), 0.0);
      
      cv.Mat maskTotalVolumePad = cv.add(maskBinairePad, sidesAlphaPad);
      cv.Mat alphaMaskTotalPad = cv.add(alphaMaskPad, sidesAlphaSmoothPad);

      equipementBgrPad.copyTo(sidesBgrSmoothPad, mask: maskBinairePad);

      // =========================================================================
      // === PHASE 4 : CALCUL DE LA DIRECTION DE LA LUMIERE ET OMBRE PROGRESSIVE ===
      // =========================================================================
      cv.Mat grayMur = cv.cvtColor(murMat, cv.COLOR_BGR2GRAY);
      int downscaleSobel = 32;
      cv.Mat grayMurSmall = cv.resize(grayMur, (wMur ~/ downscaleSobel, hMur ~/ downscaleSobel));
      
      cv.Mat maskTotalVolumeOrig = maskTotalVolumePad.region(cv.Rect(pad, pad, wMur, hMur));
      cv.Mat maskBinaireSmall = cv.resize(maskTotalVolumeOrig, (wMur ~/ downscaleSobel, hMur ~/ downscaleSobel));

      var meanStdDev = cv.meanStdDev(grayMurSmall);
      double ecartTypeMur = meanStdDev.$2.val[0]; 

      double ratioContraste = (ecartTypeMur / 50.0).clamp(0.2, 1.2);

      final double reglageOmbreDirBase = 0.12 * ratioContraste;
      final double reglageOmbreContact = estEquipementNoir ? 0.05 * ratioContraste : 0.25 * ratioContraste;

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
      // =========================================================================
      cv.Rect cropRect = cv.Rect(pad, pad, wMur, hMur);
      
      cv.Mat equipementBgr = sidesBgrSmoothPad.region(cropRect).clone();
      cv.Mat maskTotalVolume = maskTotalVolumePad.region(cropRect).clone();
      cv.Mat alphaMaskTotal = alphaMaskTotalPad.region(cropRect).clone();
      cv.Mat ombreTotale = ombreTotalePad.region(cropRect).clone();

      // =========================================================================
      // === PHASE 5 : LUMIERE ET TEMPERATURE DE COULEUR INTELLIGENTE ===
      // =========================================================================
      final double reglageMixAmbiancePiece = 0.65;
      final double reglageMixCouleurMur = 0.35;    
      final double reglageTeinteEquipementBlanc = 0.30;
      final double reglageTeinteEquipementNoir = 0.08;   
      
      final double reglageInfluenceOmbreSurBlanc = 0.65;
      final double reglageInfluenceOmbreSurNoir = 0.21;  

      cv.Scalar meanMurGlobal = cv.mean(murMat);
      cv.Scalar meanMurSousEquipement = cv.mean(murMat, mask: maskTotalVolume);

      double bMur = (meanMurGlobal.val[0] * reglageMixAmbiancePiece) + (meanMurSousEquipement.val[0] * reglageMixCouleurMur);
      double gMur = (meanMurGlobal.val[1] * reglageMixAmbiancePiece) + (meanMurSousEquipement.val[1] * reglageMixCouleurMur);
      double rMur = (meanMurGlobal.val[2] * reglageMixAmbiancePiece) + (meanMurSousEquipement.val[2] * reglageMixCouleurMur);

      double lumMurLocal = (0.114 * bMur) + (0.587 * gMur) + (0.299 * rMur);
      lumMurLocal = math.max(lumMurLocal, 1.0); 

      double ratioLuminositeAmbiante = (lumMurLocal / 128.0).clamp(0.7, 1.0);
      final double reglageLuminositeEquipementBlanc = 0.95 * ratioLuminositeAmbiante; 
      final double reglageLuminositeEquipementNoir = 0.78;

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

      var equipementChannels = cv.split(equipementBgr);

      cv.Mat bTinted = equipementChannels[0].convertTo(cv.MatType.CV_8UC1, alpha: tintB);
      cv.Mat gTinted = equipementChannels[1].convertTo(cv.MatType.CV_8UC1, alpha: tintG);
      cv.Mat rTinted = equipementChannels[2].convertTo(cv.MatType.CV_8UC1, alpha: tintR);
      cv.Mat equipementTinted = cv.merge(cv.VecMat.fromList([bTinted, gTinted, rTinted]));
      cv.Mat equipementHsv = cv.cvtColor(equipementTinted, cv.COLOR_BGR2HSV);

      var hsvChannels = cv.split(equipementHsv);

      cv.Mat grayLisse = cv.resize(grayMurFlou, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      cv.Scalar meanEquipementV = cv.mean(hsvChannels[2], mask: maskTotalVolume);

      double lumaEquipementNativeHSV = math.max(meanEquipementV.val[0], 1.0);
      double denominateurLuma = estEquipementNoir ? math.max(lumaEquipementNativeHSV, 130.0) : lumaEquipementNativeHSV;
      double influenceMur = estEquipementNoir ? reglageInfluenceOmbreSurNoir : reglageInfluenceOmbreSurBlanc; 
      double influenceEquipement = 1.0 - influenceMur;

      cv.Mat ratioSecurise = grayLisse.convertTo(cv.MatType.CV_32FC1, alpha: influenceMur / denominateurLuma, beta: influenceEquipement);
      cv.Mat vChannelF = hsvChannels[2].convertTo(cv.MatType.CV_32FC1);
      cv.Mat vShadowedF = cv.multiply(vChannelF, ratioSecurise);

      double alphaLum = estEquipementNoir ? reglageLuminositeEquipementNoir : reglageLuminositeEquipementBlanc;

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
      
      cv.Mat vLiftedF = vShadowedF.convertTo(cv.MatType.CV_32FC1, alpha: alphaLum * ratioLift, beta: voileAtmospherique);
      cv.Mat vCappedF = cv.threshold(vLiftedF, 245.0, 245.0, cv.THRESH_TRUNC).$2;
      hsvChannels[2] = vCappedF.convertTo(cv.MatType.CV_8UC1);

      cv.Mat equipementHsvFinal = cv.merge(hsvChannels);
      cv.Mat equipementRgbFinalPropre = cv.cvtColor(equipementHsvFinal, cv.COLOR_HSV2BGR);

      // =========================================================================
      // === PHASE 5.5 : DEGRADATION REALISTE (CAPTEUR PHOTO) ===
      // =========================================================================
      cv.Mat equipementBrouillee = cv.gaussianBlur(equipementRgbFinalPropre, (3, 3), 0.6);
      cv.Mat noise = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC3);
      cv.randn(noise, cv.Scalar.all(128.0), cv.Scalar.all(8.0)); 
      
      // La magie des mathématiques : l'offset de -128.0 équilibre la luminosité instantanément
      cv.Mat equipementRgbFinal = cv.addWeighted(equipementBrouillee, 1.0, noise, 1.0, -128.0);

      // =========================================================================
      // === PHASE 6 : CREATION DU CALQUE TRANSPARENT ===
      // =========================================================================
      cv.Mat invAlphaMask = cv.bitwiseNOT(alphaMaskTotal);
      
      // Le paramètre "scale" de multiply remplace la conversion matricielle de division
      cv.Mat shadowAlpha = cv.multiply(invAlphaMask, ombreTotale, scale: 1.0 / 255.0);
      cv.Mat finalAlpha8u = cv.add(alphaMaskTotal, shadowAlpha);

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