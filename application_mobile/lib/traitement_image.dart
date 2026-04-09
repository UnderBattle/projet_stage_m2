import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;
import 'package:opencv_dart/opencv_dart.dart' as cv;

class TraitementImage {

  // ==========================================
  // 1. TRI MATHÉMATIQUE DES POINTS
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

  // ==========================================
  // 2. LE TRAITEMENT D'IMAGE (PORTAGE PYTHON COMPLET)
  // ==========================================
  static Future<Uint8List?> incrusterClimatisation({
    required String photoPath,
    required String climAssetPath,
    required List<Map<String, double>> pointsIA,
  }) async {
    try {
      print("\n[OpenCV] === DÉBUT DU TRAITEMENT D'IMAGE AVANCÉ ===");
      cv.Mat murMat = cv.imread(photoPath, flags: cv.IMREAD_COLOR);
      int wMur = murMat.cols;
      int hMur = murMat.rows;

      // Échelle des points de l'IA (1024x1024 vers la taille réelle de la photo)
      double ratioX = wMur / 1024.0;
      double ratioY = hMur / 1024.0;

      List<cv.Point> ptsOri = pointsIA.map((pt) {
        return cv.Point((pt['x']! * ratioX).toInt(), (pt['y']! * ratioY).toInt());
      }).toList();

      // === PHASE 1 : EFFACEMENT DE L'AUTOCOLLANT ===
      print("[OpenCV] Phase 1 : Effacement du bout de papier...");
      cv.Mat maskGeo = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC1);
      cv.fillPoly(maskGeo, cv.VecVecPoint.fromList([ptsOri]), cv.Scalar.all(255));

      cv.Mat kernelDilate = cv.Mat.ones(15, 15, cv.MatType.CV_8UC1);
      cv.Mat maskMurExt = cv.dilate(maskGeo, kernelDilate, iterations: 1);
      cv.Mat maskBagueMur = cv.subtract(maskMurExt, maskGeo);

      cv.Scalar couleurMoyenneMur = cv.mean(murMat, mask: maskBagueMur);
      cv.Mat resultImg = murMat.clone();
      cv.fillPoly(resultImg, cv.VecVecPoint.fromList([ptsOri]), couleurMoyenneMur);

      // === PHASE 2 : CALCUL DE LA PERSPECTIVE STABILISEE ===
      print("[OpenCV] Phase 2 : Trigonométrie et Lissage...");
      cv.Point ptHg = ptsOri[0];
      cv.Point ptHd = ptsOri[1];

      double dx = (ptHd.x - ptHg.x).toDouble();
      double dy = (ptHd.y - ptHg.y).toDouble();
      double largeurPx = math.sqrt(dx * dx + dy * dy);
      double angleRad = math.atan2(dy, dx);

      // Vraies dimensions (H=100, W=50)
      double hAutoMm = 100.0;
      double wAutoMm = 50.0; 
      double ratioPhysique = hAutoMm / wAutoMm; // Ratio 2.0
      double hauteurPx = largeurPx * ratioPhysique;

      double ux = largeurPx * math.cos(angleRad);
      double uy = largeurPx * math.sin(angleRad);
      double vx = -hauteurPx * math.sin(angleRad);
      double vy = hauteurPx * math.cos(angleRad);

      List<cv.Point> ptsDstLisses = [
        cv.Point(ptHg.x, ptHg.y),
        cv.Point((ptHg.x + ux).toInt(), (ptHg.y + uy).toInt()),
        cv.Point((ptHg.x + ux + vx).toInt(), (ptHg.y + uy + vy).toInt()),
        cv.Point((ptHg.x + vx).toInt(), (ptHg.y + vy).toInt())
      ];

      // === PHASE 3 : TAILLE RÉELLE ET DÉFORMATION 3D ===
      print("[OpenCV] Phase 3 : Calcul Homographie à taille réelle...");
      ByteData climData = await rootBundle.load(climAssetPath);
      Uint8List climBytes = climData.buffer.asUint8List();
      cv.Mat climMat = cv.imdecode(climBytes, cv.IMREAD_UNCHANGED);

      // Dimensions Takao Plus
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
      print("[OpenCV] Phase 4 : Génération de l'ombre portée multiplicative...");
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
      cv.Mat resultImgF = resultImg.convertTo(cv.MatType.CV_32FC3);
      cv.Mat murOmbreF = cv.multiply(resultImgF, invOmbreF);
      
      cv.Mat murOmbre = murOmbreF.convertTo(cv.MatType.CV_8UC3);

      // === PHASE 5 : LUMIÈRE ET COLOR CAST ===
      print("[OpenCV] Phase 5 : Traitement de la lumière et de l'ambiance...");
      cv.Scalar meanMurSousClim = cv.mean(murOmbre, mask: maskBinaire);
      cv.Scalar meanGrayMurSousClim = cv.mean(grayMur, mask: maskBinaire);
      double lumMurSousClim = meanGrayMurSousClim.val[0];

      cv.Mat grayClim = cv.cvtColor(climBgr, cv.COLOR_BGR2GRAY);
      cv.Scalar meanGrayClim = cv.mean(grayClim, mask: maskBinaire);
      double lumClim = meanGrayClim.val[0];

      // Color Cast
      cv.Mat calqueAmbiance = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC3)..setTo(meanMurSousClim);
      cv.Mat climCast = cv.addWeighted(climBgr, 0.85, calqueAmbiance, 0.15, 0.0);

      // Éclairage
      double ratioLum = lumMurSousClim / (lumClim + 0.0001);
      double ratioAdouci = 1.0 - ((1.0 - ratioLum) * 0.40);
      ratioAdouci = math.max(0.7, math.min(1.1, ratioAdouci));

      cv.Mat climHsv = cv.cvtColor(climCast, cv.COLOR_BGR2HSV);
      var hsvChannels = cv.split(climHsv);
      cv.Mat vScaled = cv.addWeighted(hsvChannels[2], ratioAdouci, hsvChannels[2], 0.0, 0.0);
      hsvChannels[2] = vScaled;
      cv.Mat climHsvFinal = cv.merge(hsvChannels);
      cv.Mat climRgbFinal = cv.cvtColor(climHsvFinal, cv.COLOR_HSV2BGR);

      // === PHASE 6 : FUSION ALPHA BLENDING (ANTI-ALIASING PARFAIT) ===
      print("[OpenCV] Phase 6 : Alpha Blending (Contours ultra-lisses)...");
      
      // Préparation de l'Alpha
      cv.Mat alpha3_8u = cv.cvtColor(alphaMask, cv.COLOR_GRAY2BGR);
      cv.Mat invAlpha3_8u = cv.bitwiseNOT(alpha3_8u);

      cv.Mat alphaF = alpha3_8u.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      cv.Mat invAlphaF = invAlpha3_8u.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat fgF = climRgbFinal.convertTo(cv.MatType.CV_32FC3);
      cv.Mat bgF = murOmbre.convertTo(cv.MatType.CV_32FC3);

      // dst = src1 * alpha + src2 * (1-alpha)
      cv.Mat fgBlended = cv.multiply(fgF, alphaF);
      cv.Mat bgBlended = cv.multiply(bgF, invAlphaF);

      cv.Mat resultF = cv.add(fgBlended, bgBlended);
      
      // DERNIER CONVERT TO :
      cv.Mat resultatFinal = resultF.convertTo(cv.MatType.CV_8UC3);

      // EXPORT
      var encodeResult = cv.imencode('.jpg', resultatFinal);
      print("[OpenCV] === TERMINÉ AVEC SUCCÈS ===");
      return encodeResult.$2;

    } catch (e) {
      print("[OpenCV] ERREUR FATALE : $e");
      return null;
    }
  }
}