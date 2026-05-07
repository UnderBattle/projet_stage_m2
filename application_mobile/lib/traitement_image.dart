import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

/// Fonctions "passerelles" conçues pour être exécutées dans des Isolates.
/// Elles permettent d'effectuer les traitements lourds (OpenCV, TFLite) sans bloquer l'interface utilisateur.
class TraitementImage {
  /// Isolate pour effacer l'autocollant de l'image.
  static Future<Uint8List?> effacerAutocollantIsolate(Map<String, dynamic> params) async {
    return await effacerAutocollant(
      photoPath: params['photoPath'] as String,
      pointsIA: (params['pointsIA'] as List).map((e) => Map<String, double>.from(e)).toList(),
      lamaBytes: params['lamaBytes'] as Uint8List?,
    );
  }

  /// Isolate pour incruster la climatisation sur le mur.
  static Future<Uint8List?> incrusterClimatisationIsolate(Map<String, dynamic> params) async {
    return await incrusterClimatisation(
      fondPropreBytes: params['fondPropreBytes'] as Uint8List,
      climBytes: params['climBytes'] as Uint8List,
      pointsIA: (params['pointsIA'] as List).map((e) => Map<String, double>.from(e)).toList(),
      decalageX: params['decalageX'] as double,
      decalageY: params['decalageY'] as double,
      climAssetPath: params['climAssetPath'] as String,
      profondeurMm: params['profondeurMm'] as double,
      hauteurMm: params['hauteurMm'] as double,
      largeurMm: params['largeurMm'] as double,
    );
  }

  /// Isolate pour incruster la goulotte sur l'image.
  static Future<Uint8List?> incrusterGoulotteIsolate(Map<String, dynamic> params) async {
    return await incrusterGoulotte(
      imageAvecClimBytes: params['imageAvecClimBytes'] as Uint8List,
      ptDepartX: params['ptDepartX'] as double,
      ptDepartY: params['ptDepartY'] as double,
      ptArriveeX: params['ptArriveeX'] as double,
      ptArriveeY: params['ptArriveeY'] as double,
      largeurPx: params['largeurPx'] as double,
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

      // Dilate le masque pour s'assurer que les bords de l'autocollant sont bien inclus pour l'inpainting.
      cv.Mat kernelLama = cv.Mat.ones(25, 25, cv.MatType.CV_8UC1);
      cv.Mat maskLama = cv.dilate(maskGeo, kernelLama, iterations: 1); 

      // Calcule le rectangle qui entoure l'autocollant.
      cv.Rect rect = cv.boundingRect(cv.VecPoint.fromList(ptsOri));
      
      cv.Mat murRepare = murMat.clone();
      bool inpaintingReussi = false;
      
      // Ce rectangle sera utilisé pour optimiser le feathering plus tard.
      cv.Rect cropRect = cv.Rect(0, 0, 0, 0);

      // =========================================================================
      // === PHASE 1 : EFFACEMENT DU DEFAUT (LAMA INPAINTING ou CLONE STAMP) =====
      // =========================================================================

      // TENTATIVE AVEC IA (LaMa)
      if (lamaBytes != null) {
        try {
          print("[IA Inpainting] Démarrage de l'analyse LaMa...");
          
          // Élargit la zone de recadrage autour de l'autocollant pour donner plus de contexte à l'IA LaMa,
          // ce qui améliore la reconstruction des motifs complexes (briques, etc.).
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

          // PRÉ-TRAITEMENT POUR LAMA : Remplit la zone de l'autocollant avec la couleur moyenne du mur environnant.
          // Cette astuce aide l'IA à mieux raccorder les motifs (briques, rayures) en évitant les bavures
          // que pourrait causer un remplissage avec l'algorithme Telea.
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
          var inputImg = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => Float32List(3))));
          var inputMask = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => Float32List(1))));
          
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
          var outputImg = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => Float32List(3))));
          
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
          
          // Agrandit le patch réparé à sa taille d'origine en utilisant l'interpolation Lanczos4,
          // qui préserve mieux la netteté que les méthodes plus simples comme CUBIC.
          cv.Mat patchFinal = cv.resize(patch512, (cropS, cropS), interpolation: cv.INTER_LANCZOS4);
          
          // Applique un filtre de netteté (Unsharp Mask) pour compenser le léger flou de l'upscaling.
          cv.Mat blurredPatch = cv.gaussianBlur(patchFinal, (0, 0), 2.0); 
          cv.Mat patchNet = cv.addWeighted(patchFinal, 1.5, blurredPatch, -0.5, 0.0);
          
          // INJECTION DE BRUIT (NOISE MATCHING) : Ajoute un léger bruit gaussien au patch réparé.
          // L'IA produit un résultat souvent trop "lisse". Ce bruit permet de marier la texture du patch
          // avec le grain naturel de la photo d'origine prise par le téléphone.
          cv.Mat patchFloat = patchNet.convertTo(cv.MatType.CV_32FC3);
          cv.Mat noisePatch = cv.Mat.zeros(cropS, cropS, cv.MatType.CV_32FC3);
          cv.randn(noisePatch, cv.Scalar.all(0.0), cv.Scalar.all(4.5)); // Bruit léger
          cv.Mat patchNoisyFloat = cv.add(patchFloat, noisePatch);
          patchNet = patchNoisyFloat.convertTo(cv.MatType.CV_8UC3);

          // Copie le patch final réparé sur l'image du mur.
          patchNet.copyTo(murRepare.region(cropRect));
          
          interpreter.close();
          inpaintingReussi = true;
          print("[IA Inpainting] LaMa a rebouché le trou avec succès !");
          
        } catch (e) {
          print("[IA Inpainting] Échec de LaMa, passage au Tampon OpenCV : $e");
        }
      }

      // MÉTHODE DE SECOURS (CLONAGE SIMPLE) : Si l'inpainting avec LaMa échoue,
      // on utilise une méthode de clonage simple (copier/coller d'une zone voisine).
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

      // FUSION DES BORDS (FEATHERING) : Fusionne en douceur les bords du patch réparé avec le mur d'origine.
      // Cette méthode est plus rapide et donne de meilleurs résultats que cv.seamlessClone.
      cv.Mat resultImg = murRepare.clone(); // Par défaut, on retourne le mur réparé
      
      try {
        // OPTIMISATION : Le feathering est appliqué uniquement sur la petite zone recadrée (cropRect)
        // au lieu de l'image entière, pour des performances grandement améliorées.
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

          // Colle le petit carré fusionné à sa place sur l'image finale.
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
    required double hauteurMm,
    required double largeurMm,
  }) async {
    try {
      // Décode l'image du mur déjà nettoyée. On évite de refaire l'inpainting à chaque frame.
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
      // === PHASE PRÉ-CALCUL : OPTIMISATION DES ASSETS AVANT TRAITEMENT =======
      // =========================================================================
      cv.Mat climMat = cv.imdecode(climBytes, cv.IMREAD_UNCHANGED);
      
      // OPTIMISATION : Détermine si la climatisation est de couleur sombre en analysant l'image source (petite taille)
      // plutôt que l'image déformée (grande taille), ce qui est beaucoup plus rapide.
      var rawChannels = cv.split(climMat);
      cv.Mat rawAlpha = cv.threshold(rawChannels[3], 10, 255, cv.THRESH_BINARY).$2;
      cv.Mat rawBgr = cv.cvtColor(climMat, cv.COLOR_BGRA2BGR);
      cv.Scalar rawMeanColor = cv.mean(rawBgr, mask: rawAlpha);
      
      double lumaNativeClim = (0.114 * rawMeanColor.val[0]) + (0.587 * rawMeanColor.val[1]) + (0.299 * rawMeanColor.val[2]);
      bool estClimNoire = lumaNativeClim < 80.0;

      // =========================================================================
      // === PHASE 2 : CALCUL DE LA PERSPECTIVE STABILISEE ===
      // =========================================================================
      // Calcule une perspective 2D stable à partir des points de l'autocollant,
      // en ignorant les légères distorsions pour créer un rectangle parfait.
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
      
      // Calcule la taille en pixels de l'autocollant sur l'image de la climatisation
      // en se basant sur les dimensions réelles (en mm) fournies.
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

      // CORRECTION DE LA TRANSPARENCE : Le masque alpha est rendu binaire pour s'assurer que le centre de la clim
      // est 100% opaque et que la texture du mur ne "transparaisse" pas à travers.
      cv.Mat alphaBinaire = cv.threshold(alphaMaskOriginale, 127, 255, cv.THRESH_BINARY).$2;

      // Applique une légère érosion et un flou sur le masque pour éviter les halos blancs sur les bords.
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

      // Détecte la direction de la lumière principale sur le mur en analysant les gradients (Sobel).
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
      
      // Crée une ombre portée directionnelle, décalée dans la direction opposée à la lumière.
      var dstPtsDir = cv.VecPoint.fromList([cv.Point(shiftX.toInt(), shiftY.toInt()), cv.Point(10 + shiftX.toInt(), shiftY.toInt()), cv.Point(shiftX.toInt(), 10 + shiftY.toInt())]);
      cv.Mat affineMatDir = cv.getAffineTransform(srcPts, dstPtsDir);
      cv.Mat alphaOmbre = cv.warpAffine(alphaMask, affineMatDir, (wMur, hMur));
      
      cv.Mat smallAlpha = cv.resize(alphaOmbre, (wMur ~/ 4, hMur ~/ 4));
      int baseBlur = (5 + (ratioVolume * 4)).toInt();
      if (baseBlur % 2 == 0) baseBlur += 1; 
      
      cv.Mat smallOmbreFloue = cv.gaussianBlur(smallAlpha, (baseBlur, baseBlur), 0.0);
      cv.Mat ombreFloueDirectionnelle = cv.resize(smallOmbreFloue, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      // Crée une ombre de contact plus douce et plus proche pour simuler l'occlusion ambiante.
      var dstPtsContact = cv.VecPoint.fromList([cv.Point(0, 3), cv.Point(10, 3), cv.Point(0, 13)]);
      cv.Mat affineMatContact = cv.getAffineTransform(srcPts, dstPtsContact);
      cv.Mat alphaContact = cv.warpAffine(alphaMask, affineMatContact, (wMur, hMur));
      
      cv.Mat smallContact = cv.resize(alphaContact, (wMur ~/ 4, hMur ~/ 4));
      cv.Mat smallContactFlou = cv.gaussianBlur(smallContact, (3, 3), 0.0);
      cv.Mat ombreFloueContact = cv.resize(smallContactFlou, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      cv.Scalar lumMurGlobal = cv.mean(grayMurSmall);
      
      // Combine les deux ombres et les applique sur le mur.
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
      final double reglageInfluenceOmbreSurNoir = 0.21;  // Modifiable (Était 0.15)
      
      final double reglageLuminositeClimBlanche = 1.0; 
      
      // Luminosité de base : C'est le réglage principal pour la noirceur !
      // 0.70 = très sombre, 0.80 = moyen, 0.90 = gris foncé.
      final double reglageLuminositeClimNoire = 0.78;   // Modifiable (Était 0.70)

      cv.Mat murUltraSmall = cv.resize(resultImg, (wMur ~/ 32, hMur ~/ 32), interpolation: cv.INTER_AREA);
      cv.Mat murUltraFlou = cv.gaussianBlur(murUltraSmall, (15, 15), 0.0);
      cv.Mat murLisse = cv.resize(murUltraFlou, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      // Calcule la couleur et la luminosité ambiante de la pièce et du mur sous la clim.
      cv.Scalar meanMurGlobal = cv.mean(murLisse);
      cv.Scalar meanMurSousClim = cv.mean(murLisse, mask: maskBinaire);

      // Applique une teinte à la climatisation pour qu'elle s'intègre mieux à l'ambiance colorée de la pièce.
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

      // Ajuste la luminosité de la climatisation en fonction de la luminosité du mur derrière elle.
      cv.Mat climHsv = cv.cvtColor(climTinted, cv.COLOR_BGR2HSV);
      var hsvChannels = cv.split(climHsv);
      
      cv.Mat grayLisse = cv.cvtColor(murLisse, cv.COLOR_BGR2GRAY);
      cv.Mat grayLisseF = grayLisse.convertTo(cv.MatType.CV_32FC1);

      cv.Scalar meanClimV = cv.mean(hsvChannels[2], mask: maskBinaire);
      double lumaClimNativeHSV = math.max(meanClimV.val[0], 1.0);

      // CORRECTION POUR CLIM NOIRE : Empêche une clim noire de devenir grise sur un mur très clair
      // en limitant l'influence de la luminosité du mur.
      double denominateurLuma = estClimNoire ? math.max(lumaClimNativeHSV, 130.0) : lumaClimNativeHSV;

      // Le "Ratio Magique" : compare la luminosité du mur à celle de la clim pour ajuster l'exposition.
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

      // CORRECTION DU CONTRE-JOUR : Simule un "voile atmosphérique" en se basant sur le contraste local du mur.
      // Cela évite que la clim ne devienne trop sombre ou "brûlée" dans des conditions de forte luminosité
      // (ex: mur blanc à côté d'une fenêtre).
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

      // Empêche les pixels de la clim de devenir blancs purs (surexposition).
      cv.Mat vCappedF = cv.threshold(vLiftedF, 245.0, 245.0, cv.THRESH_TRUNC).$2;
      hsvChannels[2] = vCappedF.convertTo(cv.MatType.CV_8UC1);

      cv.Mat climHsvFinal = cv.merge(hsvChannels);
      cv.Mat climRgbFinalPropre = cv.cvtColor(climHsvFinal, cv.COLOR_HSV2BGR);

      // =========================================================================
      // === PHASE 5.5 : DÉGRADATION RÉALISTE (CAPTEUR PHOTO) ===
      // =========================================================================
      
      // Applique un très léger flou pour casser l'aspect "image de synthèse" trop net.
      cv.Mat climBrouillee = cv.gaussianBlur(climRgbFinalPropre, (3, 3), 0.6);

      // Ajoute un bruit numérique (grain) pour que la clim corresponde à la texture de la photo d'origine.
      cv.Mat climFloat = climBrouillee.convertTo(cv.MatType.CV_32FC3);
      cv.Mat noise = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC3);
      
      // Génère un bruit gaussien aléatoire simulant l'ISO du téléphone
      cv.randn(noise, cv.Scalar.all(0.0), cv.Scalar.all(8.0)); 
      cv.Mat climNoisyFloat = cv.add(climFloat, noise);
      
      cv.Mat climRgbFinal = climNoisyFloat.convertTo(cv.MatType.CV_8UC3);

      // =========================================================================
      // === PHASE 6 : FUSION ALPHA BLENDING ===
      // =========================================================================
      // Fusionne la climatisation (avec ses effets de lumière) et le mur (avec son ombre) en utilisant le masque alpha.
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

  /// Dessine une ligne épaisse avec des extrémités plates en traçant un polygone rectangulaire.
  /// Remplace `cv.line` qui produit des bouts arrondis avec une épaisseur élevée.
  static void _tracerLigneRectangulaire(cv.Mat mat, cv.Point p1, cv.Point p2, cv.Scalar color, double thickness) {
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
  // === INCRUSTATION RÉALISTE DE LA GOULOTTE ===
  // =========================================================================
  static Future<Uint8List?> incrusterGoulotte({
    required Uint8List imageAvecClimBytes,
    required double ptDepartX,
    required double ptDepartY,
    required double ptArriveeX,
    required double ptArriveeY,
    required double largeurPx,
  }) async {
    try {
      // Décode l'image de fond (celle où la clim est déjà incrustée)
      cv.Mat fondMat = cv.imdecode(imageAvecClimBytes, cv.IMREAD_COLOR);
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
        // L'extrusion se fait vers le bas pour simuler un angle de vue de face.
        double depthExtrusion = largeurPx * 0.20; // Le bouchon recule de 20% de la largeur de la goulotte
        double ex = 0.0; // Pas de décalage à droite/gauche
        double ey = depthExtrusion; // Décalage uniquement vers le bas

        // Effet de rétrécissement pour simuler la perspective lointaine (le mur)
        double shrinkFactor = 0.85; 
        ptLeftMur = cv.Point((bottomPt.x - nxCap * htCap * shrinkFactor + ex).toInt(), 
                             (bottomPt.y - nyCap * htCap * shrinkFactor + ey).toInt());
        ptRightMur = cv.Point((bottomPt.x + nxCap * htCap * shrinkFactor + ex).toInt(), 
                              (bottomPt.y + nyCap * htCap * shrinkFactor + ey).toInt());
      }

      // 1. Création du masque binaire global (Goulotte + Bouchon)
      cv.Mat maskBinaire = cv.Mat.zeros(h, w, cv.MatType.CV_8UC1);
      _tracerLigneRectangulaire(maskBinaire, p1, p2, cv.Scalar.all(255), largeurPx);
      if (lenCap >= 1.0) {
        cv.fillPoly(maskBinaire, cv.VecVecPoint.fromList([[ptLeft, ptRight, ptRightMur, ptLeftMur]]), cv.Scalar.all(255), lineType: cv.LINE_AA);
      }

      // 2. Construit l'apparence visuelle de la goulotte en superposant plusieurs lignes de couleurs différentes
      // pour simuler un effet de cylindre 3D avec des reflets.
      cv.Mat goulotteBgr = cv.Mat.zeros(h, w, cv.MatType.CV_8UC3);
      
      // Couche de fond (Les bords sombres du plastique qui fuient vers le mur)
      _tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(210, 210, 210, 0), largeurPx);
      // Couche principale (Gris clair plastique)
      _tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(240, 245, 245, 0), largeurPx * 0.85);
      // Lumière diffuse sur le dessus
      _tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(245, 250, 250, 0), largeurPx * 0.50);
      // Reflet spéculaire (Ligne de lumière très fine pour le côté lisse/brillant)
      _tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(255, 255, 255, 0), largeurPx * 0.15);
      
      // Un léger flou fond les couches pour créer un dégradé cylindrique réaliste.
      cv.Mat goulotteBgrSmooth = cv.gaussianBlur(goulotteBgr, (5, 5), 0.0);

      // Dessine la face inférieure du "bouchon" pour donner un effet de volume.
      if (lenCap >= 1.0) {
        // On dessine la face inférieure (qui s'enfonce vers le mur) APRES le flou pour garder des arêtes nettes type moulage plastique
        // La couleur est plus sombre car c'est une zone d'ombre propre (face orientée vers le bas)
        cv.fillPoly(goulotteBgrSmooth, cv.VecVecPoint.fromList([[ptLeft, ptRight, ptRightMur, ptLeftMur]]), cv.Scalar(160, 165, 165, 0), lineType: cv.LINE_AA);
        
        // Ligne d'arête pour marquer nettement le volume et la cassure 3D
        int edgeThickness = math.max(1, (largeurPx * 0.05).toInt());
        cv.line(goulotteBgrSmooth, ptLeft, ptRight, cv.Scalar(190, 195, 195, 0), thickness: edgeThickness, lineType: cv.LINE_AA);
      }

      // 3. Création de l'ombre portée de la goulotte (Drop Shadow)
      cv.Mat affineMatDir = cv.getAffineTransform(
        cv.VecPoint.fromList([cv.Point(0, 0), cv.Point(10, 0), cv.Point(0, 10)]),
        cv.VecPoint.fromList([cv.Point(5, 8), cv.Point(15, 8), cv.Point(5, 18)]) // Force du décalage de l'ombre
      );
      cv.Mat shadowWarped = cv.warpAffine(maskBinaire, affineMatDir, (w, h));
      cv.Mat shadowBlurred = cv.gaussianBlur(shadowWarped, (15, 15), 0.0);
      
      // Assombrissement progressif de l'image de fond
      cv.Mat ombre8u = shadowBlurred.convertTo(cv.MatType.CV_8UC1, alpha: 0.35); // Opacité de l'ombre
      cv.Mat invOmbre8u = cv.bitwiseNOT(ombre8u);
      cv.Mat invOmbre3c = cv.cvtColor(invOmbre8u, cv.COLOR_GRAY2BGR);
      cv.Mat invOmbreF = invOmbre3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      
      cv.Mat fondF = fondMat.convertTo(cv.MatType.CV_32FC3);
      cv.Mat fondOmbreF = cv.multiply(fondF, invOmbreF);
      cv.Mat murOmbre = fondOmbreF.convertTo(cv.MatType.CV_8UC3);

      // 4. Teinte et Lumière dynamiques (Même traitement que pour la Clim)
      // CORRECTION DE LA TRANSPARENCE : Utilise une version très floutée du mur pour l'analyse de la lumière.
      // Cela évite que la texture du mur (ex: briques) n'apparaisse "à travers" la goulotte.
      cv.Mat murUltraSmall = cv.resize(murOmbre, (w ~/ 32, h ~/ 32), interpolation: cv.INTER_AREA);
      cv.Mat murUltraFlou = cv.gaussianBlur(murUltraSmall, (15, 15), 0.0);
      cv.Mat murLisse = cv.resize(murUltraFlou, (w, h), interpolation: cv.INTER_CUBIC);

      cv.Mat grayLisse = cv.cvtColor(murLisse, cv.COLOR_BGR2GRAY);
      cv.Mat grayLisseF = grayLisse.convertTo(cv.MatType.CV_32FC1);
      
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

      // La goulotte blanche absorbe la teinte de la pièce à hauteur de 35%
      double forceTeinteGoulotte = 0.35;
      tintB = 1.0 + (tintB - 1.0) * forceTeinteGoulotte;
      tintG = 1.0 + (tintG - 1.0) * forceTeinteGoulotte;
      tintR = 1.0 + (tintR - 1.0) * forceTeinteGoulotte;

      cv.Mat tintMat = cv.Mat.zeros(h, w, cv.MatType.CV_32FC3)..setTo(cv.Scalar(tintB, tintG, tintR, 0));
      cv.Mat goulotteF = goulotteBgrSmooth.convertTo(cv.MatType.CV_32FC3); // On utilise la version smooth !
      cv.Mat goulotteTintedF = cv.multiply(goulotteF, tintMat);
      cv.Mat goulotteTinted = goulotteTintedF.convertTo(cv.MatType.CV_8UC3);

      // Ajustement de l'exposition (HSV)
      cv.Mat goulotteHsv = cv.cvtColor(goulotteTinted, cv.COLOR_BGR2HSV);
      var hsvChannels = cv.split(goulotteHsv);
      
      cv.Scalar meanGoulotteV = cv.mean(hsvChannels[2], mask: maskBinaire);
      double lumaGoulotteNative = math.max(meanGoulotteV.val[0], 1.0);

      cv.Mat ratioMap = grayLisseF.convertTo(cv.MatType.CV_32FC1, alpha: 1.0 / lumaGoulotteNative);
      cv.Mat matriceUn = cv.Mat.zeros(h, w, cv.MatType.CV_32FC1)..setTo(cv.Scalar.all(1.0));
      
      // POUR RÉGLER LA LUMINOSITÉ (Influence du mur)
      // Si la goulotte est trop sombre sur un mur foncé, baisse cette variable vers 0.30 ou 0.20
      // pour que la goulotte "ignore" l'obscurité du mur en dessous d'elle.
      double influenceMurGoulotte = 0.43; 
      cv.Mat ratioSecurise = cv.addWeighted(ratioMap, influenceMurGoulotte, matriceUn, 1.0 - influenceMurGoulotte, 0.0);
      
      cv.Mat vChannelF = hsvChannels[2].convertTo(cv.MatType.CV_32FC1);
      cv.Mat vShadowedF = cv.multiply(vChannelF, ratioSecurise);

      // Contre-jour (Voile atmosphérique) calqué sur le contraste environnant
      cv.Rect rectGoulotte = cv.boundingRect(cv.VecPoint.fromList([p1, p2]));
      int rx = math.max(0, rectGoulotte.x - 20);
      int ry = math.max(0, rectGoulotte.y - 20);
      int rw = math.min(w - rx, rectGoulotte.width + 40);
      int rh = math.min(h - ry, rectGoulotte.height + 40);

      cv.Mat grayMurTextureBrute = cv.cvtColor(murOmbre, cv.COLOR_BGR2GRAY); // Ici on prend la vraie texture pour le contraste
      double voileAtmospherique = 0.0;
      if (rw > 0 && rh > 0) {
        cv.Mat roiMurGray = grayMurTextureBrute.region(cv.Rect(rx, ry, rw, rh));
        var minMaxRoi = cv.minMaxLoc(roiMurGray);
        double contrasteLocal = math.max(0.0, minMaxRoi.$2 - minMaxRoi.$1);
        voileAtmospherique = math.min(contrasteLocal * 0.15, 25.0);
      }
      
      double ratioLift = (255.0 - voileAtmospherique) / 255.0;
      cv.Mat vLiftedF = cv.addWeighted(vShadowedF, ratioLift, vShadowedF, 0.0, voileAtmospherique);
      cv.Mat vCappedF = cv.threshold(vLiftedF, 245.0, 245.0, cv.THRESH_TRUNC).$2;
      hsvChannels[2] = vCappedF.convertTo(cv.MatType.CV_8UC1);

      cv.Mat goulotteHsvFinal = cv.merge(hsvChannels);
      cv.Mat goulotteRgbFinalPropre = cv.cvtColor(goulotteHsvFinal, cv.COLOR_HSV2BGR);

      // 5. Dégradation photographique
      // Flou pour enlever l'aspect "image de synthèse"
      cv.Mat goulotteBrouillee = cv.gaussianBlur(goulotteRgbFinalPropre, (3, 3), 0.6);
      
      cv.Mat goulotteFloat = goulotteBrouillee.convertTo(cv.MatType.CV_32FC3);
      cv.Mat noise = cv.Mat.zeros(h, w, cv.MatType.CV_32FC3);
      // Simulation du grain ISO de la caméra
      cv.randn(noise, cv.Scalar.all(0.0), cv.Scalar.all(5.0)); 
      cv.Mat goulotteNoisyFloat = cv.add(goulotteFloat, noise);
      
      cv.Mat goulotteRgbFinal = goulotteNoisyFloat.convertTo(cv.MatType.CV_8UC3);

      // 6. Fusion Finale (Alpha Blending)
      // Léger flou sur le masque binaire pour un anti-aliasing parfait des bords
      cv.Mat alphaMask = cv.gaussianBlur(maskBinaire, (3, 3), 0.0);
      cv.Mat alpha3_8u = cv.cvtColor(alphaMask, cv.COLOR_GRAY2BGR);
      cv.Mat invAlpha3_8u = cv.bitwiseNOT(alpha3_8u);

      cv.Mat alphaF = alpha3_8u.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      cv.Mat invAlphaF = invAlpha3_8u.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat fgF = goulotteRgbFinal.convertTo(cv.MatType.CV_32FC3);
      cv.Mat bgF = murOmbre.convertTo(cv.MatType.CV_32FC3);

      cv.Mat fgBlended = cv.multiply(fgF, alphaF);
      cv.Mat bgBlended = cv.multiply(bgF, invAlphaF);

      cv.Mat resultF = cv.add(fgBlended, bgBlended);
      cv.Mat resultatFinal = resultF.convertTo(cv.MatType.CV_8UC3);

      var encodeResult = cv.imencode('.jpg', resultatFinal);
      return encodeResult.$2;
      
    } catch (e) {
      print("[OpenCV] Erreur lors de l'incrustation de la goulotte : $e");
      return null;
    }
  }
}