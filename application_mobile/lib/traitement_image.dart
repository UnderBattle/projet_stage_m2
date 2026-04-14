import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;

class TraitementImage {

  // ==========================================
  // PASSERELLES POUR LES ISOLATES (CALCUL EN ARRIÈRE-PLAN)
  // ==========================================
  static Future<Uint8List?> effacerAutocollantIsolate(Map<String, dynamic> params) async {
    return await effacerAutocollant(
      photoPath: params['photoPath'] as String,
      pointsIA: (params['pointsIA'] as List).map((e) => Map<String, double>.from(e)).toList(),
    );
  }

  static Future<Uint8List?> incrusterClimatisationIsolate(Map<String, dynamic> params) async {
    return await incrusterClimatisation(
      photoPath: params['photoPath'] as String,
      climBytes: params['climBytes'] as Uint8List,
      pointsIA: (params['pointsIA'] as List).map((e) => Map<String, double>.from(e)).toList(),
      decalageX: params['decalageX'] as double,
      decalageY: params['decalageY'] as double,
      climAssetPath: params['climAssetPath'] as String,
    );
  }


  // ==========================================
  // 1. TRI MATHÉMATIQUE DES POINTS ET FOND PROPRE
  // ==========================================
  static List<Map<String, double>> trierPoints(List<Map<String, double>> points) {
    List<Map<String, double>> pts = List.from(points);
    pts.sort((a, b) => (a['x']! + a['y']!).compareTo(b['x']! + b['y']!));
    var hg = pts.first;
    var bd = pts.last;
    pts.remove(hg);
    pts.remove(bd);
    pts.sort((a, b) => (a['y']! - a['x']!).compareTo(b['y']! - b['x']!));
    var hd = pts.first;
    var bg = pts.last;
    return [hg, hd, bd, bg];
  }

  static Future<Uint8List?> effacerAutocollant({
    required String photoPath,
    required List<Map<String, double>> pointsIA,
  }) async {
    try {
      cv.Mat murMat = cv.imread(photoPath, flags: cv.IMREAD_COLOR);
      int wMur = murMat.cols;
      int hMur = murMat.rows;

      double ratioX = wMur / 1024.0;
      double ratioY = hMur / 1024.0;

      List<cv.Point> ptsOri = pointsIA.map((pt) {
        return cv.Point((pt['x']! * ratioX).toInt(), (pt['y']! * ratioY).toInt());
      }).toList();

      cv.Mat maskGeo = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC1);
      cv.fillPoly(maskGeo, cv.VecVecPoint.fromList([ptsOri]), cv.Scalar.all(255));

      cv.Mat kernelDilate = cv.Mat.ones(15, 15, cv.MatType.CV_8UC1);
      cv.Mat maskMurExt = cv.dilate(maskGeo, kernelDilate, iterations: 1);
      cv.Mat maskBagueMur = cv.subtract(maskMurExt, maskGeo);

      cv.Scalar couleurMoyenneMur = cv.mean(murMat, mask: maskBagueMur);
      
      cv.Mat murBase = murMat.clone();
      cv.fillPoly(murBase, cv.VecVecPoint.fromList([ptsOri]), couleurMoyenneMur);

      cv.Mat murFlou = cv.gaussianBlur(murBase, (81, 81), 0.0);

      cv.Mat maskTransition = cv.dilate(maskGeo, kernelDilate, iterations: 2); 
      cv.Mat maskFeather8u = cv.gaussianBlur(maskTransition, (51, 51), 0.0);

      cv.Mat maskFeather3c = cv.cvtColor(maskFeather8u, cv.COLOR_GRAY2BGR);
      cv.Mat maskFeatherF = maskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat invMaskFeather8u = cv.bitwiseNOT(maskFeather8u);
      cv.Mat invMaskFeather3c = cv.cvtColor(invMaskFeather8u, cv.COLOR_GRAY2BGR);
      cv.Mat invMaskFeatherF = invMaskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat murBaseF = murBase.convertTo(cv.MatType.CV_32FC3);
      cv.Mat murFlouF = murFlou.convertTo(cv.MatType.CV_32FC3);

      cv.Mat fgInpaint = cv.multiply(murFlouF, maskFeatherF);
      cv.Mat bgInpaint = cv.multiply(murBaseF, invMaskFeatherF);
      
      cv.Mat resultImgF = cv.add(fgInpaint, bgInpaint);
      cv.Mat resultImg = resultImgF.convertTo(cv.MatType.CV_8UC3);

      var encodeResult = cv.imencode('.jpg', resultImg);
      return encodeResult.$2;
    } catch (e) {
      print("[OpenCV] Erreur lors de la génération du fond propre : $e");
      return null;
    }
  }

  // ==========================================
  // 2. LE TRAITEMENT D'IMAGE COMPLET
  // ==========================================
  static Future<Uint8List?> incrusterClimatisation({
    required String photoPath,
    required Uint8List climBytes,
    required List<Map<String, double>> pointsIA,
    double decalageX = 0.0,
    double decalageY = 0.0, required String climAssetPath,
  }) async {
    try {
      cv.Mat murMat = cv.imread(photoPath, flags: cv.IMREAD_COLOR);
      int wMur = murMat.cols;
      int hMur = murMat.rows;

      double ratioX = wMur / 1024.0;
      double ratioY = hMur / 1024.0;

      List<cv.Point> ptsOri = pointsIA.map((pt) {
        return cv.Point((pt['x']! * ratioX).toInt(), (pt['y']! * ratioY).toInt());
      }).toList();

      // === PHASE 1 : INPAINTING MAISON ===
      cv.Mat maskGeo = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC1);
      cv.fillPoly(maskGeo, cv.VecVecPoint.fromList([ptsOri]), cv.Scalar.all(255));

      cv.Mat kernelDilate = cv.Mat.ones(15, 15, cv.MatType.CV_8UC1);
      cv.Mat maskMurExt = cv.dilate(maskGeo, kernelDilate, iterations: 1);
      cv.Mat maskBagueMur = cv.subtract(maskMurExt, maskGeo);

      cv.Scalar couleurMoyenneMur = cv.mean(murMat, mask: maskBagueMur);
      
      cv.Mat murBase = murMat.clone();
      cv.fillPoly(murBase, cv.VecVecPoint.fromList([ptsOri]), couleurMoyenneMur);

      cv.Mat murFlou = cv.gaussianBlur(murBase, (81, 81), 0.0);

      cv.Mat maskTransition = cv.dilate(maskGeo, kernelDilate, iterations: 2); 
      cv.Mat maskFeather8u = cv.gaussianBlur(maskTransition, (51, 51), 0.0);

      cv.Mat maskFeather3c = cv.cvtColor(maskFeather8u, cv.COLOR_GRAY2BGR);
      cv.Mat maskFeatherF = maskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat invMaskFeather8u = cv.bitwiseNOT(maskFeather8u);
      cv.Mat invMaskFeather3c = cv.cvtColor(invMaskFeather8u, cv.COLOR_GRAY2BGR);
      cv.Mat invMaskFeatherF = invMaskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat murBaseF = murBase.convertTo(cv.MatType.CV_32FC3);
      cv.Mat murFlouF = murFlou.convertTo(cv.MatType.CV_32FC3);

      cv.Mat fgInpaint = cv.multiply(murFlouF, maskFeatherF);
      cv.Mat bgInpaint = cv.multiply(murBaseF, invMaskFeatherF);
      
      cv.Mat resultImgF = cv.add(fgInpaint, bgInpaint);
      cv.Mat resultImg = resultImgF.convertTo(cv.MatType.CV_8UC3);

      // === PHASE 2 : CALCUL DE LA PERSPECTIVE STABILISEE ===
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

      // === PHASE 3 : TAILLE RÉELLE ET DÉFORMATION 3D ===
      cv.Mat climMat = cv.imdecode(climBytes, cv.IMREAD_UNCHANGED);

      double hClimMm = 270.0; 
      double wClimMm = 798.0; 
      int wImgClim = climMat.cols;
      int hImgClim = climMat.rows;

      double wAutoPx = (wAutoMm / wClimMm) * wImgClim;
      double hAutoPx = (hAutoMm / hClimMm) * hImgClim;

      List<cv.Point> ptsSrc = [
        cv.Point(0, 0),
        cv.Point(wAutoPx.toInt(), 0),
        cv.Point(wAutoPx.toInt(), hAutoPx.toInt()),
        cv.Point(0, hAutoPx.toInt())
      ];

      var vecPtsSrc = cv.VecPoint.fromList(ptsSrc);
      var vecPtsDst = cv.VecPoint.fromList(ptsDstLisses);
      cv.Mat hMatrix = cv.getPerspectiveTransform(vecPtsSrc, vecPtsDst);

      cv.Mat climWarped = cv.warpPerspective(climMat, hMatrix, (wMur, hMur));
      var channels = cv.split(climWarped);
      cv.Mat alphaMask = channels[3]; 
      cv.Mat climBgr = cv.cvtColor(climWarped, cv.COLOR_BGRA2BGR);
      cv.Mat maskBinaire = cv.threshold(alphaMask, 5, 255, cv.THRESH_BINARY).$2;

      // === PHASE 4 : L'OMBRE PORTÉE RÉALISTE ===
      List<cv.Point> ptsDstOmbre = ptsDstLisses.map((pt) => cv.Point(pt.x + 10, pt.y + 20)).toList();
      cv.Mat hMatrixOmbre = cv.getPerspectiveTransform(vecPtsSrc, cv.VecPoint.fromList(ptsDstOmbre));
      cv.Mat climWarpedOmbre = cv.warpPerspective(climMat, hMatrixOmbre, (wMur, hMur));
      cv.Mat alphaOmbre = cv.split(climWarpedOmbre)[3];

      cv.Mat ombreFloue = cv.gaussianBlur(alphaOmbre, (61, 61), 0.0);

      cv.Mat grayMur = cv.cvtColor(resultImg, cv.COLOR_BGR2GRAY);
      cv.Scalar lumMurGlobal = cv.mean(grayMur);
      double intensite = 0.05 + (lumMurGlobal.val[0] / 255.0) * 0.20;

      cv.Mat ombre8u = ombreFloue.convertTo(cv.MatType.CV_8UC1, alpha: intensite);
      cv.Mat invOmbre8u = cv.bitwiseNOT(ombre8u);
      cv.Mat invOmbre3c = cv.cvtColor(invOmbre8u, cv.COLOR_GRAY2BGR);
      
      cv.Mat invOmbreF = invOmbre3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      cv.Mat resultImgForShadow = resultImg.convertTo(cv.MatType.CV_32FC3);
      cv.Mat murOmbreF = cv.multiply(resultImgForShadow, invOmbreF);
      
      cv.Mat murOmbre = murOmbreF.convertTo(cv.MatType.CV_8UC3);

      // === PHASE 5 : LUMIÈRE ET COLOR CAST (Ambiance) ===
      cv.Scalar meanMurSousClim = cv.mean(murOmbre, mask: maskBinaire);
      cv.Scalar meanGrayMurSousClim = cv.mean(grayMur, mask: maskBinaire);
      double lumMurSousClim = meanGrayMurSousClim.val[0];

      cv.Mat grayClim = cv.cvtColor(climBgr, cv.COLOR_BGR2GRAY);
      cv.Scalar meanGrayClim = cv.mean(grayClim, mask: maskBinaire);
      double lumClim = meanGrayClim.val[0];

      cv.Mat calqueAmbiance = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC3)..setTo(meanMurSousClim);
      cv.Mat climCast = cv.addWeighted(climBgr, 0.85, calqueAmbiance, 0.15, 0.0);

      double ratioLum = lumMurSousClim / (lumClim + 0.0001);
      double ratioAdouci = 1.0 - ((1.0 - ratioLum) * 0.40);
      ratioAdouci = math.max(0.7, math.min(1.1, ratioAdouci));

      cv.Mat climHsv = cv.cvtColor(climCast, cv.COLOR_BGR2HSV);
      var hsvChannels = cv.split(climHsv);
      cv.Mat vScaled = cv.addWeighted(hsvChannels[2], ratioAdouci, hsvChannels[2], 0.0, 0.0);
      hsvChannels[2] = vScaled;
      cv.Mat climHsvFinal = cv.merge(hsvChannels);
      cv.Mat climRgbFinal = cv.cvtColor(climHsvFinal, cv.COLOR_HSV2BGR);

      // === PHASE 6 : FUSION ALPHA BLENDING ===
      cv.Mat alpha3_8u = cv.cvtColor(alphaMask, cv.COLOR_GRAY2BGR);
      cv.Mat invAlpha3_8u = cv.bitwiseNOT(alpha3_8u);

      cv.Mat alphaF = alpha3_8u.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      cv.Mat invAlphaF = invAlpha3_8u.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat fgF = climRgbFinal.convertTo(cv.MatType.CV_32FC3);
      cv.Mat bgF = murOmbre.convertTo(cv.MatType.CV_32FC3);

      cv.Mat fgBlended = cv.multiply(fgF, alphaF);
      cv.Mat bgBlended = cv.multiply(bgF, invAlphaF);

      cv.Mat resultF = cv.add(fgBlended, bgBlended);
      cv.Mat resultatFinal = resultF.convertTo(cv.MatType.CV_8UC3);

      // EXPORT
      var encodeResult = cv.imencode('.jpg', resultatFinal);
      return encodeResult.$2;

    } catch (e) {
      print("[OpenCV] ERREUR FATALE : $e");
      return null;
    }
  }
}