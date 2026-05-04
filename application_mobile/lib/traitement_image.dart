import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class TraitementImage {
  // Fonctions "passerelles" pour exécuter les traitements lourds dans un Isolate.
  // Cela évite de geler l'interface utilisateur.
  static Future<Uint8List?> effacerAutocollantIsolate(Map<String, dynamic> params) async {
    return await effacerAutocollant(
      photoPath: params['photoPath'] as String,
      pointsIA: (params['pointsIA'] as List).map((e) => Map<String, double>.from(e)).toList(),
      lamaBytes: params['lamaBytes'] as Uint8List?,
    );
  }

  static Future<Uint8List?> incrusterClimatisationIsolate(Map<String, dynamic> params) async {
    return await incrusterClimatisation(
      fondPropreBytes: params['fondPropreBytes'] as Uint8List, // NOUVEAU : On récupère l'image déjà nettoyée
      climBytes: params['climBytes'] as Uint8List,
      pointsIA: (params['pointsIA'] as List).map((e) => Map<String, double>.from(e)).toList(),
      decalageX: params['decalageX'] as double,
      decalageY: params['decalageY'] as double,
      climAssetPath: params['climAssetPath'] as String,
      profondeurMm: params['profondeurMm'] as double,
      hauteurMm: params['hauteurMm'] as double, // AJOUT : Passage de la hauteur dynamique
      largeurMm: params['largeurMm'] as double, // AJOUT : Passage de la largeur dynamique
    );
  }

  /// Trie les 4 points reçus de l'IA pour les ordonner : Haut-Gauche, Haut-Droit, Bas-Droit, Bas-Gauche.
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

  /// Efface l'autocollant de l'image du mur en utilisant LaMa Inpainting (ou clonage en secours).
  static Future<Uint8List?> effacerAutocollant({
    required String photoPath,
    required List<Map<String, double>> pointsIA,
    required Uint8List? lamaBytes,
  }) async {
    try {
      cv.Mat murMat = cv.imread(photoPath, flags: cv.IMREAD_COLOR);
      int wMur = murMat.cols;
      int hMur = murMat.rows;

      // Calcule les ratios pour redimensionner les points de l'IA (qui sont basés sur une image 1024x1024).
      double ratioX = wMur / 1024.0;
      double ratioY = hMur / 1024.0;

      // Convertit les points de l'IA en points OpenCV.
      List<cv.Point> ptsOri = pointsIA.map((pt) {
        return cv.Point((pt['x']! * ratioX).toInt(), (pt['y']! * ratioY).toInt());
      }).toList();

      // Crée un masque noir de la forme de l'autocollant.
      cv.Mat maskGeo = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC1);
      cv.fillPoly(maskGeo, cv.VecVecPoint.fromList([ptsOri]), cv.Scalar.all(255));

      // DILATATION POUR L'IA (On gonfle le masque avec une marge de 25px pour avaler les bords verts)
      cv.Mat kernelLama = cv.Mat.ones(25, 25, cv.MatType.CV_8UC1);
      cv.Mat maskLama = cv.dilate(maskGeo, kernelLama, iterations: 1); 

      // Calcule le rectangle qui entoure l'autocollant.
      cv.Rect rect = cv.boundingRect(cv.VecPoint.fromList(ptsOri));
      
      cv.Mat murRepare = murMat.clone();
      bool inpaintingReussi = false;
      
      // On sauvegarde ce rectangle pour l'optimisation plus bas
      cv.Rect cropRect = cv.Rect(0, 0, 0, 0);

      // =========================================================================
      // === PHASE 1 : EFFACEMENT DU DEFAUT (LAMA INPAINTING ou CLONE STAMP) =====
      // =========================================================================

      // TENTATIVE AVEC IA (LaMa)
      if (lamaBytes != null) {
        try {
          print("[IA Inpainting] Démarrage de l'analyse LaMa...");
          
          // AJOUT : Élargissement du contexte. En passant de 1.8 à 2.5, LaMa voit beaucoup plus 
          // du mur autour et a beaucoup plus de facilité à reproduire les motifs complexes.
          int cropS = (math.max(rect.width, rect.height) * 2.5).toInt();
          int cropX = (rect.x + rect.width / 2 - cropS / 2).toInt();
          int cropY = (rect.y + rect.height / 2 - cropS / 2).toInt();
          
          // Sécurisation des bords
          if (cropX < 0) cropX = 0;
          if (cropY < 0) cropY = 0;
          if (cropX + cropS > wMur) cropX = wMur - cropS;
          if (cropY + cropS > hMur) cropY = hMur - cropS;
          if (cropS > wMur) cropS = wMur;
          if (cropS > hMur) cropS = hMur;

          cropRect = cv.Rect(cropX, cropY, cropS, cropS);
          cv.Mat cropImg = murMat.region(cropRect).clone(); 
          cv.Mat cropMaskLama = maskLama.region(cropRect);

          // =================================================================================
          // L'ASTUCE OPENCV + LAMA : On remplit l'autocollant avec la couleur moyenne.
          // Cela permet à l'IA d'avoir une coupure nette pour continuer les motifs (briques, rayures)
          // sans baver, contrairement à l'algorithme Telea qui détruit les raccords.
          // =================================================================================
          cv.Mat invCropMask = cv.bitwiseNOT(cropMaskLama);
          cv.Scalar couleurMoyenne = cv.mean(cropImg, mask: invCropMask);
          cropImg.setTo(couleurMoyenne, mask: cropMaskLama);

          // Redimensionnement au standard LaMa (512x512)
          cv.Mat img512 = cv.resize(cropImg, (512, 512));
          cv.Mat mask512 = cv.resize(cropMaskLama, (512, 512));
          
          // LaMa exige du RGB, OpenCV utilise du BGR
          cv.Mat imgRGB = cv.cvtColor(img512, cv.COLOR_BGR2RGB);
          Uint8List rgbBytes = imgRGB.data;
          Uint8List maskBytes = mask512.data;

          // Préparation des tenseurs pour TFLite
          var inputImg = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => List.generate(3, (l) => 0.0))));
          var inputMask = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => List.generate(1, (l) => 0.0))));
          
          int idx = 0;
          for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
              inputImg[0][y][x][0] = rgbBytes[idx] / 255.0;     // R
              inputImg[0][y][x][1] = rgbBytes[idx+1] / 255.0;   // G
              inputImg[0][y][x][2] = rgbBytes[idx+2] / 255.0;   // B
              
              inputMask[0][y][x][0] = maskBytes[y * 512 + x] > 127 ? 1.0 : 0.0;
              idx += 3;
            }
          }

          // Inférence LaMa
          Interpreter interpreter = Interpreter.fromBuffer(lamaBytes);
          
          var tensor0 = interpreter.getInputTensor(0);
          List<Object> inputs = (tensor0.shape.last == 3) ? [inputImg, inputMask] : [inputMask, inputImg];
          var outputImg = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => List.generate(3, (l) => 0.0))));
          
          interpreter.runForMultipleInputs(inputs, {0: outputImg});

          // Reconstruction de l'image (Retour en BGR pour OpenCV)
          Uint8List outBytes = Uint8List(512 * 512 * 3);
          int outIdx = 0;
          for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
              outBytes[outIdx] = (outputImg[0][y][x][2] * 255).clamp(0, 255).toInt();   // B
              outBytes[outIdx+1] = (outputImg[0][y][x][1] * 255).clamp(0, 255).toInt(); // G
              outBytes[outIdx+2] = (outputImg[0][y][x][0] * 255).clamp(0, 255).toInt(); // R
              outIdx += 3;
            }
          }

          img.Image repairedImg = img.Image.fromBytes(width: 512, height: 512, bytes: outBytes.buffer, order: img.ChannelOrder.bgr);
          Uint8List jpgBytes = img.encodeJpg(repairedImg, quality: 100);
          cv.Mat patch512 = cv.imdecode(jpgBytes, cv.IMREAD_COLOR);
          
          // =================================================================================
          // UPSCALING PRO ET FILTRE DE NETTETÉ (Pour tuer le flou d'agrandissement)
          // =================================================================================
          // AJOUT : On utilise INTER_LANCZOS4 au lieu de INTER_CUBIC. C'est l'algorithme le plus 
          // performant pour garder la netteté de la texture quand on agrandit l'image de l'IA.
          cv.Mat patchFinal = cv.resize(patch512, (cropS, cropS), interpolation: cv.INTER_LANCZOS4);
          
          // Filtre Unsharp Mask (Netteté de la texture) ajusté pour ne pas brûler les couleurs
          cv.Mat blurredPatch = cv.gaussianBlur(patchFinal, (0, 0), 2.0); 
          cv.Mat patchNet = cv.addWeighted(patchFinal, 1.5, blurredPatch, -0.5, 0.0);
          
          // AJOUT : Injection de grain photographique (Noise Matching). 
          // L'IA génère un rendu "plastique". On ajoute du bruit iso pour faire correspondre 
          // la zone réparée à la texture granuleuse de la vraie photo du téléphone.
          cv.Mat patchFloat = patchNet.convertTo(cv.MatType.CV_32FC3);
          cv.Mat noisePatch = cv.Mat.zeros(cropS, cropS, cv.MatType.CV_32FC3);
          cv.randn(noisePatch, cv.Scalar.all(0.0), cv.Scalar.all(4.5)); // Bruit léger
          cv.Mat patchNoisyFloat = cv.add(patchFloat, noisePatch);
          patchNet = patchNoisyFloat.convertTo(cv.MatType.CV_8UC3);

          // On remet la zone à sa taille d'origine et on la colle
          patchNet.copyTo(murRepare.region(cropRect));
          
          interpreter.close();
          inpaintingReussi = true;
          print("[IA Inpainting] LaMa a rebouché le trou avec succès !");
          
        } catch (e) {
          print("[IA Inpainting] Échec de LaMa, passage au Tampon OpenCV : $e");
        }
      }

      // Fallback si l'IA plante
      if (!inpaintingReussi) {
        int padding = 20;
        int rectX = math.max(0, rect.x - padding);
        int rectY = math.max(0, rect.y - padding);
        int rectW = math.min(wMur - rectX, rect.width + padding * 2);
        int rectH = math.min(hMur - rectY, rect.height + padding * 2);

        int srcX = rectX;
        int srcY = rectY;
        if (rectY - rectH > 0) {
          srcY = rectY - rectH; 
        } else if (rectY + rectH * 2 < hMur) {
          srcY = rectY + rectH; 
        } else if (rectX - rectW > 0) {
          srcX = rectX - rectW; 
        } else if (rectX + rectW * 2 < wMur) {
          srcX = rectX + rectW; 
        }

        cropRect = cv.Rect(rectX, rectY, rectW, rectH); // On sauvegarde le rect utilisé pour le fallback
        cv.Mat patch = murMat.region(cv.Rect(srcX, srcY, rectW, rectH));
        patch.copyTo(murRepare.region(cropRect));
      }

      // =================================================================================
      // FUSION PAR FEATHERING (10x plus rapide et plus fidèle que SeamlessClone)
      // =================================================================================
      cv.Mat resultImg = murRepare.clone(); // Par défaut, on retourne le mur réparé
      
      try {
        // OPTIMISATION DU FEATHERING
        // Au lieu de calculer le feathering sur l'image entière de 12 Mégapixels,
        // on ne le calcule QUE sur le petit carré (cropRect) qu'on vient de réparer !
        if (cropRect.width > 0 && cropRect.height > 0) {
          cv.Mat petitMaskLama = maskLama.region(cropRect);
          cv.Mat petitMurRepare = murRepare.region(cropRect);
          cv.Mat petitMurOriginal = murMat.region(cropRect);

          cv.Mat maskFeather8u = cv.gaussianBlur(petitMaskLama, (31, 31), 0.0);
          cv.Mat maskFeather3c = cv.cvtColor(maskFeather8u, cv.COLOR_GRAY2BGR);
          cv.Mat maskFeatherF = maskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

          cv.Mat invMaskFeather8u = cv.bitwiseNOT(maskFeather8u);
          cv.Mat invMaskFeather3c = cv.cvtColor(invMaskFeather8u, cv.COLOR_GRAY2BGR);
          cv.Mat invMaskFeatherF = invMaskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

          cv.Mat murRepareF = petitMurRepare.convertTo(cv.MatType.CV_32FC3);
          cv.Mat murOriginalF = petitMurOriginal.convertTo(cv.MatType.CV_32FC3);

          cv.Mat fgInpaint = cv.multiply(murRepareF, maskFeatherF);
          cv.Mat bgInpaint = cv.multiply(murOriginalF, invMaskFeatherF);
          
          cv.Mat petitResultImgF = cv.add(fgInpaint, bgInpaint);
          cv.Mat petitResultImg = petitResultImgF.convertTo(cv.MatType.CV_8UC3);

          // On colle le petit carré fusionné à sa place sur l'image finale
          petitResultImg.copyTo(resultImg.region(cropRect));
        }
      } catch (e) {
        print("[OpenCV] Erreur lors du blending : $e");
        resultImg = murRepare; // Sécurité
      }

      // Encode l'image finale en JPEG.
      var encodeResult = cv.imencode('.jpg', resultImg);
      return encodeResult.$2;
      
    } catch (e) {
      print("[OpenCV] Erreur lors de la génération du fond propre : $e");
      return null;
    }
  }

  /// Incruste la climatisation sur le mur en gérant la perspective, l'ombre et la lumière.
  static Future<Uint8List?> incrusterClimatisation({
    required Uint8List fondPropreBytes, 
    required Uint8List climBytes,
    required List<Map<String, double>> pointsIA,
    double decalageX = 0.0,
    double decalageY = 0.0, 
    required String climAssetPath,
    required double profondeurMm, 
    required double hauteurMm, // Passage de la hauteur dynamique
    required double largeurMm, // Passage de la largeur dynamique
  }) async {
    try {
      // Décode directement le mur nettoyé (fini de recalculer l'inpainting à chaque mouvement !)
      cv.Mat murMat = cv.imdecode(fondPropreBytes, cv.IMREAD_COLOR);
      int wMur = murMat.cols;
      int hMur = murMat.rows;

      // Calcule les ratios pour redimensionner les points de l'IA.
      double ratioX = wMur / 1024.0;
      double ratioY = hMur / 1024.0;

      // Convertit les points de l'IA en points OpenCV.
      List<cv.Point> ptsOri = pointsIA.map((pt) {
        return cv.Point((pt['x']! * ratioX).toInt(), (pt['y']! * ratioY).toInt());
      }).toList();

      // On utilise directement le mur réparé comme base de travail
      cv.Mat resultImg = murMat.clone(); 

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

      // =========================================================================
      // === PHASE 3 : TAILLE RÉELLE ET DÉFORMATION 3D ===
      // =========================================================================
      cv.Mat climMat = cv.imdecode(climBytes, cv.IMREAD_UNCHANGED);
      
      // Remplacement des variables en dur par les paramètres dynamiques
      double hClimMm = hauteurMm; 
      double wClimMm = largeurMm; 
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
      cv.Mat alphaMaskOriginale = channels[3]; 

      // FIX TRANSPARENCE : Bétonnage du masque Alpha !
      // On force tous les pixels semi-transparents du centre de l'image (ex: l'ombre interne des plastiques)
      // à devenir 100% opaques (255) pour que les lattes de bois ne passent plus à travers.
      cv.Mat alphaBinaire = cv.threshold(alphaMaskOriginale, 127, 255, cv.THRESH_BINARY).$2;

      // Anti-Halo (érosion douce du masque)
      cv.Mat kernelErode = cv.Mat.ones(3, 3, cv.MatType.CV_8UC1);
      cv.Mat alphaErode = cv.erode(alphaBinaire, kernelErode);
      cv.Mat alphaMask = cv.gaussianBlur(alphaErode, (3, 3), 0.0);
      
      cv.Mat climBgr = cv.cvtColor(climWarped, cv.COLOR_BGRA2BGR);
      cv.Mat maskBinaire = cv.threshold(alphaMask, 5, 255, cv.THRESH_BINARY).$2;

      // =========================================================================
      // === PHASE 4 : CALCUL DE LA DIRECTION DE LA LUMIÈRE & OMBRE PROGRESSIVE ===
      // =========================================================================
      // VARIABLES DE RÉGLAGES DES OMBRES
      final double reglageOmbreDirBase = 0.12; // Opacité de base de l'ombre longue/directionnelle
      final double reglageOmbreContact = 0.25;  // Opacité de l'ombre de contact juste sous la clim

      cv.Mat grayMur = cv.cvtColor(resultImg, cv.COLOR_BGR2GRAY);
      
      int downscaleSobel = 32;
      cv.Mat grayMurSmall = cv.resize(grayMur, (wMur ~/ downscaleSobel, hMur ~/ downscaleSobel));
      cv.Mat maskBinaireSmall = cv.resize(maskBinaire, (wMur ~/ downscaleSobel, hMur ~/ downscaleSobel));

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
      
      double shiftX = norme < 1.0 ? (3.0 * ratioVolume) : -dirLumiereX * forceOmbre;
      double shiftY = norme < 1.0 ? (8.0 * ratioVolume) : -dirLumiereY * forceOmbre;

      var srcPts = cv.VecPoint.fromList([cv.Point(0, 0), cv.Point(10, 0), cv.Point(0, 10)]);
      
      // Ombre Directionnelle
      var dstPtsDir = cv.VecPoint.fromList([cv.Point(shiftX.toInt(), shiftY.toInt()), cv.Point(10 + shiftX.toInt(), shiftY.toInt()), cv.Point(shiftX.toInt(), 10 + shiftY.toInt())]);
      cv.Mat affineMatDir = cv.getAffineTransform(srcPts, dstPtsDir);
      cv.Mat alphaOmbre = cv.warpAffine(alphaMask, affineMatDir, (wMur, hMur));
      
      cv.Mat smallAlpha = cv.resize(alphaOmbre, (wMur ~/ 4, hMur ~/ 4));
      int baseBlur = (5 + (ratioVolume * 4)).toInt();
      if (baseBlur % 2 == 0) baseBlur += 1; 
      
      cv.Mat smallOmbreFloue = cv.gaussianBlur(smallAlpha, (baseBlur, baseBlur), 0.0);
      cv.Mat ombreFloueDirectionnelle = cv.resize(smallOmbreFloue, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      // Ombre de Contact (Ambient Occlusion)
      var dstPtsContact = cv.VecPoint.fromList([cv.Point(0, 3), cv.Point(10, 3), cv.Point(0, 13)]);
      cv.Mat affineMatContact = cv.getAffineTransform(srcPts, dstPtsContact);
      cv.Mat alphaContact = cv.warpAffine(alphaMask, affineMatContact, (wMur, hMur));
      
      cv.Mat smallContact = cv.resize(alphaContact, (wMur ~/ 4, hMur ~/ 4));
      cv.Mat smallContactFlou = cv.gaussianBlur(smallContact, (3, 3), 0.0);
      cv.Mat ombreFloueContact = cv.resize(smallContactFlou, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      cv.Scalar lumMurGlobal = cv.mean(grayMurSmall);
      
      // On utilise nos nouvelles variables de réglages
      double intensiteDir = reglageOmbreDirBase + (lumMurGlobal.val[0] / 255.0) * 0.30;
      double intensiteContact = reglageOmbreContact; 

      cv.Mat ombreDir8u = ombreFloueDirectionnelle.convertTo(cv.MatType.CV_8UC1, alpha: intensiteDir);
      cv.Mat ombreContact8u = ombreFloueContact.convertTo(cv.MatType.CV_8UC1, alpha: intensiteContact);
      
      cv.Mat ombreTotale = cv.add(ombreDir8u, ombreContact8u);
      cv.Mat invOmbre8u = cv.bitwiseNOT(ombreTotale);
      cv.Mat invOmbre3c = cv.cvtColor(invOmbre8u, cv.COLOR_GRAY2BGR);
      
      cv.Mat invOmbreF = invOmbre3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      cv.Mat resultImgForShadow = resultImg.convertTo(cv.MatType.CV_32FC3);
      cv.Mat murOmbreF = cv.multiply(resultImgForShadow, invOmbreF);
      
      cv.Mat murOmbre = murOmbreF.convertTo(cv.MatType.CV_8UC3);

      // =========================================================================
      // === PHASE 5 : LUMIÈRE ET TEMPÉRATURE DE COULEUR INTELLIGENTE ===
      // =========================================================================
      
      // VARIABLES DE RÉGLAGES DE LA COULEUR ET LUMINOSITÉ
      final double reglageMixAmbiancePiece = 0.65; // Teinte prise sur l'ambiance de la pièce
      final double reglageMixCouleurMur = 0.35;    // Teinte prise sur la couleur du mur juste derrière
      
      final double reglageTeinteClimBlanche = 0.30; // Capacité du plastique blanc à absorber la teinte (miroir)
      final double reglageTeinteClimNoire = 0.08;   // Capacité du plastique noir à absorber la teinte
      
      final double reglageInfluenceOmbreSurBlanc = 0.65;
      
      // POUR RÉGLER LA NOIRCEUR : Joue avec ces deux variables
      // 1. Influence du mur : Plus c'est haut (ex: 0.25), plus un mur clair va "laver" ou éclaircir la clim noire.
      final double reglageInfluenceOmbreSurNoir = 0.20;  // Modifiable (Était 0.15)
      
      final double reglageLuminositeClimBlanche = 1.0; 
      
      // Luminosité de base : C'est le réglage principal pour la noirceur !
      // 0.70 = très sombre, 0.80 = moyen, 0.90 = gris foncé.
      final double reglageLuminositeClimNoire = 0.78;   // Modifiable (Était 0.70)

      cv.Scalar meanClimColorO = cv.mean(climBgr, mask: maskBinaire);
      double lumaNativeClim = (0.114 * meanClimColorO.val[0]) + (0.587 * meanClimColorO.val[1]) + (0.299 * meanClimColorO.val[2]);
      bool estClimNoire = lumaNativeClim < 80.0;

      cv.Mat murUltraSmall = cv.resize(resultImg, (wMur ~/ 32, hMur ~/ 32), interpolation: cv.INTER_AREA);
      cv.Mat murUltraFlou = cv.gaussianBlur(murUltraSmall, (15, 15), 0.0);
      cv.Mat murLisse = cv.resize(murUltraFlou, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      // --- Teinte de la climatisation ---
      cv.Scalar meanMurGlobal = cv.mean(murLisse);
      cv.Scalar meanMurSousClim = cv.mean(murLisse, mask: maskBinaire);

      // Mixage grâce aux variables définies en haut
      double bMur = (meanMurGlobal.val[0] * reglageMixAmbiancePiece) + (meanMurSousClim.val[0] * reglageMixCouleurMur);
      double gMur = (meanMurGlobal.val[1] * reglageMixAmbiancePiece) + (meanMurSousClim.val[1] * reglageMixCouleurMur);
      double rMur = (meanMurGlobal.val[2] * reglageMixAmbiancePiece) + (meanMurSousClim.val[2] * reglageMixCouleurMur);

      double lumMurLocal = (0.114 * bMur) + (0.587 * gMur) + (0.299 * rMur);
      lumMurLocal = math.max(lumMurLocal, 1.0); 

      double tintB = bMur / lumMurLocal;
      double tintG = gMur / lumMurLocal;
      double tintR = rMur / lumMurLocal;

      double forceTeinte = estClimNoire ? reglageTeinteClimNoire : reglageTeinteClimBlanche; 
      tintB = 1.0 + (tintB - 1.0) * forceTeinte;
      tintG = 1.0 + (tintG - 1.0) * forceTeinte;
      tintR = 1.0 + (tintR - 1.0) * forceTeinte;

      tintB = math.max(0.80, math.min(1.20, tintB));
      tintG = math.max(0.80, math.min(1.20, tintG));
      tintR = math.max(0.80, math.min(1.20, tintR));

      cv.Mat tintMat = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC3)..setTo(cv.Scalar(tintB, tintG, tintR, 0));
      cv.Mat climF = climBgr.convertTo(cv.MatType.CV_32FC3);
      cv.Mat climTintedF = cv.multiply(climF, tintMat);
      cv.Mat climTinted = climTintedF.convertTo(cv.MatType.CV_8UC3);

      // --- Simulation de l'exposition (Ombre et Soleil sur la clim) ---
      cv.Mat climHsv = cv.cvtColor(climTinted, cv.COLOR_BGR2HSV);
      var hsvChannels = cv.split(climHsv);
      
      cv.Mat grayLisse = cv.cvtColor(murLisse, cv.COLOR_BGR2GRAY);
      cv.Mat grayLisseF = grayLisse.convertTo(cv.MatType.CV_32FC1);

      cv.Scalar meanClimV = cv.mean(hsvChannels[2], mask: maskBinaire);
      double lumaClimNativeHSV = math.max(meanClimV.val[0], 1.0);

      // FIX CLIM NOIRE GRISE : On empêche le ratio d'exploser si la clim est très sombre et le mur très clair.
      // Un mur à 255 / une clim à 20 = multiplicateur énorme de 12.75 ! On force un dénominateur plus élevé pour le noir.
      double denominateurLuma = estClimNoire ? math.max(lumaClimNativeHSV, 130.0) : lumaClimNativeHSV;

      // Le Ratio magique : Lumière du Mur / Lumière de la Clim
      cv.Mat ratioMap = grayLisseF.convertTo(cv.MatType.CV_32FC1, alpha: 1.0 / denominateurLuma);
      cv.Mat matriceUn = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC1)..setTo(cv.Scalar.all(1.0));
      
      // Influence dynamique basée sur nos réglages
      double influenceMur = estClimNoire ? reglageInfluenceOmbreSurNoir : reglageInfluenceOmbreSurBlanc; 
      double influenceClim = 1.0 - influenceMur; // C'est la balance opposée mathématique
      
      cv.Mat ratioSecurise = cv.addWeighted(ratioMap, influenceMur, matriceUn, influenceClim, 0.0);
      cv.Mat vChannelF = hsvChannels[2].convertTo(cv.MatType.CV_32FC1);
      cv.Mat vShadowedF = cv.multiply(vChannelF, ratioSecurise);

      // Application de l'alpha (luminosité) selon nos nouveaux réglages
      if (estClimNoire) {
         vShadowedF = vShadowedF.convertTo(cv.MatType.CV_32FC1, alpha: reglageLuminositeClimNoire);
      } else {
         vShadowedF = vShadowedF.convertTo(cv.MatType.CV_32FC1, alpha: reglageLuminositeClimBlanche); 
      }

      // FIX CONTRE-JOUR REVISITÉ : Adaptabilité au contraste local
      // L'ancienne méthode avec minMax prenait la valeur brute, ce qui ajoutait "+100" à une clim noire
      // sur un mur blanc pur (la rendant grise fluo). 
      // La nouvelle méthode analyse la "dureté" de la lumière via l'amplitude du contraste local.
      cv.Rect rectClim = cv.boundingRect(cv.VecPoint.fromList(ptsDstLisses));
      int rx = math.max(0, rectClim.x - 20);
      int ry = math.max(0, rectClim.y - 20);
      int rw = math.min(wMur - rx, rectClim.width + 40);
      int rh = math.min(hMur - ry, rectClim.height + 40);
      
      double voileAtmospherique = 0.0;
      
      if (rw > 0 && rh > 0) {
        cv.Mat roiMurGray = grayMur.region(cv.Rect(rx, ry, rw, rh));
        var minMaxRoi = cv.minMaxLoc(roiMurGray);
        double minLocal = minMaxRoi.$1;
        double maxLocal = minMaxRoi.$2;
        
        // Calcul d'un proxy pour l'écart-type (contraste local)
        double contrasteLocal = math.max(0.0, maxLocal - minLocal);
        
        // Plus le contraste est élevé, plus le risque de contre-jour vif est fort.
        voileAtmospherique = (contrasteLocal * 0.15); 
        
        // Sécurité stricte : On bride sévèrement le voile sur les machines noires
        // pour qu'elles restent noires, même sur un mur très contrasté.
        if (estClimNoire) {
           voileAtmospherique = math.min(voileAtmospherique, 8.0); // Plafond très bas pour le noir
        } else {
           voileAtmospherique = math.min(voileAtmospherique, 25.0); // Plafond plus permissif pour le blanc
        }
      }

      // On compense le contre-jour (Level Lift) de façon sécurisée
      double ratioLift = (255.0 - voileAtmospherique) / 255.0;
      cv.Mat vLiftedF = cv.addWeighted(vShadowedF, ratioLift, vShadowedF, 0.0, voileAtmospherique);

      // SÉCURITÉ ANTI-SUREXPOSITION : On empêche les pixels de dépasser 245/255.
      cv.Mat vCappedF = cv.threshold(vLiftedF, 245.0, 245.0, cv.THRESH_TRUNC).$2;
      hsvChannels[2] = vCappedF.convertTo(cv.MatType.CV_8UC1);

      cv.Mat climHsvFinal = cv.merge(hsvChannels);
      cv.Mat climRgbFinalPropre = cv.cvtColor(climHsvFinal, cv.COLOR_HSV2BGR);

      // =========================================================================
      // === PHASE 5.5 : DÉGRADATION RÉALISTE (CAPTEUR PHOTO) ===
      // =========================================================================
      
      // Léger flou pour tuer la netteté parfaite (CGI) de l'image de synthèse 3D
      cv.Mat climBrouillee = cv.gaussianBlur(climRgbFinalPropre, (3, 3), 0.6);

      // Ajout de grain (bruit numérique) pour "ancrer" la clim dans la texture de la vraie photo
      cv.Mat climFloat = climBrouillee.convertTo(cv.MatType.CV_32FC3);
      cv.Mat noise = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC3);
      
      // Génère un bruit gaussien aléatoire simulant l'ISO du téléphone
      cv.randn(noise, cv.Scalar.all(0.0), cv.Scalar.all(8.0)); 
      cv.Mat climNoisyFloat = cv.add(climFloat, noise);
      
      cv.Mat climRgbFinal = climNoisyFloat.convertTo(cv.MatType.CV_8UC3);

      // =========================================================================
      // === PHASE 6 : FUSION ALPHA BLENDING ===
      // =========================================================================
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

      var encodeResult = cv.imencode('.jpg', resultatFinal);
      return encodeResult.$2;
      
    } catch (e) {
      print("[OpenCV] ERREUR FATALE : $e");
      return null;
    }
  }
}