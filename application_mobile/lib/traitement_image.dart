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

  /// NOUVEAU : Isolate pour générer l'équipement sur fond transparent (PNG).
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

  /// NOUVEAU : Isolate d'assemblage ultra-rapide (Fusionne n'importe quel fond avec le PNG transparent)
  static Future<Uint8List?> fusionnerCalqueIsolate(Map<String, dynamic> params) async {
    try {
      Uint8List fondBytes = params['fondBytes'];
      Uint8List calquePngBytes = params['calquePngBytes'];
      
      cv.Mat bg = cv.imdecode(fondBytes, cv.IMREAD_COLOR);
      cv.Mat overlay = cv.imdecode(calquePngBytes, cv.IMREAD_UNCHANGED); // Conserve le canal Alpha
      
      // Extraction sécurisée du BGR et du canal Alpha
      cv.Mat overlayBgr = cv.cvtColor(overlay, cv.COLOR_BGRA2BGR);
      var channels = cv.split(overlay);
      cv.Mat alpha = channels[3];
      
      // Normalisation de l'Alpha sur les 3 canaux
      cv.Mat alpha3c8u = cv.cvtColor(alpha, cv.COLOR_GRAY2BGR);
      cv.Mat alpha3cF = alpha3c8u.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      
      // Inversion de l'Alpha (pour creuser le trou dans l'image de fond)
      cv.Mat invAlpha3c8u = cv.bitwiseNOT(alpha3c8u);
      cv.Mat invAlpha3cF = invAlpha3c8u.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      
      // Blending mathématique
      cv.Mat bgF = bg.convertTo(cv.MatType.CV_32FC3);
      cv.Mat fgF = overlayBgr.convertTo(cv.MatType.CV_32FC3);
      
      cv.Mat bgBlended = cv.multiply(bgF, invAlpha3cF);
      cv.Mat fgBlended = cv.multiply(fgF, alpha3cF);
      
      cv.Mat resultF = cv.add(bgBlended, fgBlended);
      cv.Mat result8u = resultF.convertTo(cv.MatType.CV_8UC3);
      
      var encodeResult = cv.imencode('.jpg', result8u);
      return encodeResult.$2;
    } catch (e) {
      print("[OpenCV] Erreur lors de la fusion du calque : $e");
      return null;
    }
  }

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
      // === PHASE 1 : EFFACEMENT DU DEFAUT (LAMA INPAINTING OU CLONE STAMP) ===
      // =========================================================================

      // TENTATIVE AVEC IA (LAMA)
      if (lamaBytes != null) {
        try {
          print("[IA Inpainting] Démarrage de l'analyse LaMa...");
          
          // Elargit la zone de recadrage autour de l'autocollant pour donner plus de contexte à l'IA LaMa,
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

          // PRE-TRAITEMENT POUR LAMA : Remplit la zone de l'autocollant avec la couleur moyenne du mur environnant.
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

          // Préparation des tenseurs pour TFLite : Remis en listes standards au lieu de Float32List pour éviter les crashs de mapping
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
          print("[IA Inpainting] Echec de LaMa, passage au Tampon OpenCV : $e");
        }
      }

      // METHODE DE SECOURS (CLONAGE SIMPLE) : Si l'inpainting avec LaMa échoue,
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

  /// NOUVEAU : Calcule l'équipement, ses couleurs et ses ombres portées et renvoie le tout 
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

      double ratioX = wMur / 1024.0;
      double ratioY = hMur / 1024.0;

      List<cv.Point> ptsOri = pointsIA.map((pt) {
        return cv.Point((pt['x']! * ratioX).toInt(), (pt['y']! * ratioY).toInt());
      }).toList();

      cv.Mat resultImg = murMat.clone(); 

      // =========================================================================
      // === PHASE PRE-CALCUL : OPTIMISATION DES ASSETS AVANT TRAITEMENT ===
      // =========================================================================
      cv.Mat equipementMat = cv.imdecode(equipementBytes, cv.IMREAD_UNCHANGED);
      
      // OPTIMISATION : Détermine si l'équipement est de couleur sombre en analysant l'image source (petite taille)
      // plutôt que l'image déformée (grande taille), ce qui est beaucoup plus rapide.
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

      // =========================================================================
      // === PHASE 3 : TAILLE REELLE ET DEFORMATION 3D ===
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
      var vecPtsDst = cv.VecPoint.fromList(ptsDstLisses);
      cv.Mat hMatrix = cv.getPerspectiveTransform(vecPtsSrc, vecPtsDst);
      cv.Mat equipementWarped = cv.warpPerspective(equipementMat, hMatrix, (wMur, hMur));

      var channels = cv.split(equipementWarped);
      cv.Mat alphaMaskOriginale = channels[3]; 

      cv.Mat alphaBinaire = cv.threshold(alphaMaskOriginale, 127, 255, cv.THRESH_BINARY).$2;
      cv.Mat kernelErode = cv.Mat.ones(3, 3, cv.MatType.CV_8UC1);
      cv.Mat alphaErode = cv.erode(alphaBinaire, kernelErode);
      cv.Mat alphaMask = cv.gaussianBlur(alphaErode, (3, 3), 0.0);
      
      cv.Mat equipementBgr = cv.cvtColor(equipementWarped, cv.COLOR_BGRA2BGR);
      cv.Mat maskBinaire = cv.threshold(alphaMask, 5, 255, cv.THRESH_BINARY).$2;

      // =========================================================================
      // === PHASE 3.5 : GÉNÉRATION DU VOLUME 3D (EXTRUSION EXACTE DU PNG) ===
      // =========================================================================
      // Calcul du volume 3D basé sur le CONTOUR REEL du PNG (alpha mask)
      // et non plus sur le rectangle de l'image. Cela permet d'avoir un volume 
      // qui épouse parfaitement la forme de l'installation !

      // On trouve le centre de l'image et le centre géométrique de la machine (via les moments)
      double imgCX = wMur / 2.0;
      double imgCY = hMur / 2.0;
      
      var moments = cv.moments(maskBinaire);
      double eqCX = moments.m10 / (moments.m00 + 0.0001);
      double eqCY = moments.m01 / (moments.m00 + 0.0001);
      
      // Le vecteur de fuite part du centre de l'image (effet de perspective photographique)
      double vecX = imgCX - eqCX;
      double vecY = imgCY - eqCY;
      
      // L'extrusion dépend de la profondeur physique
      // AJOUT : La base d'extrusion est ajustée dynamiquement en fonction du ratio du bloc.
      double baseExtrusion = (profondeurMm / 1000.0) * 0.14; 
      
      double ratioForme = hauteurMm / largeurMm;
      if (ratioForme < 0.45) {
          // Si c'est une clim murale (ratio très écrasé), on diminue l'extrusion 3D 
          // pour éviter que les coques arrondies n'aient l'air d'être des boîtes épaisses.
          baseExtrusion *= 0.45;
      }

      int dxBack = (vecX * baseExtrusion).toInt();
      int dyBack = (vecY * baseExtrusion).toInt();
      
      cv.Mat sidesBgr = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC3);
      cv.Mat sidesAlpha = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC1);
      
      // Couleur moyenne de l'installation pour des ombres 3D réalistes (pas trop sombres)
      cv.Scalar baseColorEq = cv.mean(equipementBgr, mask: maskBinaire);
      double mB = baseColorEq.val[0];
      double mG = baseColorEq.val[1];
      double mR = baseColorEq.val[2];
      
      // Simulation d'éclairage : La couleur s'adapte à la VRAIE couleur de la clim/unité !
      // Les zones orientées vers le bas (colorBottom) sont légèrement plus sombres pour un meilleur effet de volume.
      cv.Scalar colorTop = cv.Scalar(mB * 0.98, mG * 0.98, mR * 0.98, 0);
      cv.Scalar colorBottom = cv.Scalar(mB * 0.55, mG * 0.55, mR * 0.55, 0); 
      cv.Scalar colorLeft = cv.Scalar(mB * 0.88, mG * 0.88, mR * 0.88, 0);
      cv.Scalar colorRight = cv.Scalar(mB * 0.85, mG * 0.85, mR * 0.85, 0);

      // Dessin du "fond" du volume 3D (le capot arrière collé au mur)
      // On translate simplement le masque alpha parfait de la machine
      cv.Mat affineBack = cv.getAffineTransform(
        cv.VecPoint.fromList([cv.Point(0, 0), cv.Point(10, 0), cv.Point(0, 10)]),
        cv.VecPoint.fromList([cv.Point(dxBack, dyBack), cv.Point(10 + dxBack, dyBack), cv.Point(dxBack, 10 + dyBack)])
      );
      cv.Mat backMaskAlpha = cv.warpAffine(maskBinaire, affineBack, (wMur, hMur));
      cv.Mat backCapBgr = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC3)..setTo(colorBottom); 
      backCapBgr.copyTo(sidesBgr, mask: backMaskAlpha);
      cv.bitwiseOR(sidesAlpha, backMaskAlpha, dst: sidesAlpha);

      // Dessin des murs latéraux reliant l'avant et l'arrière en suivant LE CONTOUR DU PNG
      // AJOUT : On rétrécit (érode) le masque pour la construction des murs latéraux.
      // Cela permet aux murs 3D d'être en LÉGER RETRAIT par rapport à la façade lisse de la machine, 
      // réglant ainsi le problème du "contour trop marqué" des climatisations.
      cv.Mat kernelShrink = cv.Mat.ones(5, 5, cv.MatType.CV_8UC1);
      cv.Mat maskBinaireShrunk = cv.erode(maskBinaire, kernelShrink);

      var contoursResult = cv.findContours(maskBinaireShrunk, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
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

          // Vecteur radial depuis le centre de la machine
          double radX = midX - eqCX;
          double radY = midY - eqCY;
          
          // On calcule la VRAIE perpendiculaire du contour !
          double segDx = (p2.x - p1.x).toDouble();
          double segDy = (p2.y - p1.y).toDouble();
          
          double nx = -segDy;
          double ny = segDx;
          
          // On s'assure que la normale pointe vers l'extérieur
          if ((nx * radX + ny * radY) < 0) {
            nx = -nx;
            ny = -ny;
          }

          double len = math.sqrt(nx * nx + ny * ny) + 0.0001;
          nx /= len;
          ny /= len;

          cv.Scalar faceColor;
          // Détermination intelligente de la face pour appliquer la bonne lumière directionnelle
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
          cv.fillPoly(sidesBgr, cv.VecVecPoint.fromList([quad]), faceColor, lineType: cv.LINE_AA);
          cv.fillPoly(sidesAlpha, cv.VecVecPoint.fromList([quad]), cv.Scalar.all(255), lineType: cv.LINE_AA);
        }
      }

      // Un flou plus fort sur les couleurs pour adoucir les transitions 
      // entre les faces (haut clair, bas sombre) et créer un dégradé naturel (effet biseauté).
      cv.Mat sidesBgrSmooth = cv.gaussianBlur(sidesBgr, (15, 15), 0.0);
      cv.Mat sidesAlphaSmooth = cv.gaussianBlur(sidesAlpha, (3, 3), 0.0);
      
      // Masque global englobant le volume 3D + la face avant
      cv.Mat maskTotalVolume = cv.add(maskBinaire, sidesAlpha);
      cv.Mat alphaMaskTotal = cv.add(alphaMask, sidesAlphaSmooth);

      cv.Mat equipement3DBgr = sidesBgrSmooth.clone();
      // On plaque l'image frontale exacte par dessus les murs 3D pour refermer la boîte
      equipementBgr.copyTo(equipement3DBgr, mask: maskBinaire);
      
      equipementBgr = equipement3DBgr; 

      // =========================================================================
      // === PHASE 4 : CALCUL DE LA DIRECTION DE LA LUMIERE ET OMBRE PROGRESSIVE ===
      // =========================================================================
      cv.Mat grayMur = cv.cvtColor(resultImg, cv.COLOR_BGR2GRAY);
      
      int downscaleSobel = 32;
      cv.Mat grayMurSmall = cv.resize(grayMur, (wMur ~/ downscaleSobel, hMur ~/ downscaleSobel));
      
      // L'ombre portée est maintenant calculée par rapport au volume total ! (Plus de boîte qui flotte)
      cv.Mat maskBinaireSmall = cv.resize(maskTotalVolume, (wMur ~/ downscaleSobel, hMur ~/ downscaleSobel));

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

      var srcPts = cv.VecPoint.fromList([cv.Point(0, 0), cv.Point(10, 0), cv.Point(0, 10)]);
      
      var dstPtsDir = cv.VecPoint.fromList([cv.Point(shiftX.toInt(), shiftY.toInt()), cv.Point(10 + shiftX.toInt(), shiftY.toInt()), cv.Point(shiftX.toInt(), 10 + shiftY.toInt())]);
      cv.Mat affineMatDir = cv.getAffineTransform(srcPts, dstPtsDir);
      
      // On projette l'ombre à partir du volume TOTAL pour éviter l'effet "boîte flottante"
      cv.Mat alphaOmbre = cv.warpAffine(alphaMaskTotal, affineMatDir, (wMur, hMur));
      
      cv.Mat smallAlpha = cv.resize(alphaOmbre, (wMur ~/ 4, hMur ~/ 4));
      int baseBlur = (5 + (ratioVolume * 4) + ((1.0 - ratioContraste) * 8)).toInt();
      if (baseBlur % 2 == 0) baseBlur += 1; 
      
      cv.Mat smallOmbreFloue = cv.gaussianBlur(smallAlpha, (baseBlur, baseBlur), 0.0);
      cv.Mat ombreFloueDirectionnelle = cv.resize(smallOmbreFloue, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      var dstPtsContact = cv.VecPoint.fromList([cv.Point(0, 3), cv.Point(10, 3), cv.Point(0, 13)]);
      cv.Mat affineMatContact = cv.getAffineTransform(srcPts, dstPtsContact);
      cv.Mat alphaContact = cv.warpAffine(alphaMaskTotal, affineMatContact, (wMur, hMur));
      
      cv.Mat smallContact = cv.resize(alphaContact, (wMur ~/ 4, hMur ~/ 4));
      cv.Mat smallContactFlou = cv.gaussianBlur(smallContact, (3, 3), 0.0);
      cv.Mat ombreFloueContact = cv.resize(smallContactFlou, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      cv.Mat ombreDir8u = ombreFloueDirectionnelle.convertTo(cv.MatType.CV_8UC1, alpha: reglageOmbreDirBase);
      cv.Mat ombreContact8u = ombreFloueContact.convertTo(cv.MatType.CV_8UC1, alpha: reglageOmbreContact);
      
      // Masque d'ombre pure (0 = pas d'ombre, >0 = ombre)
      cv.Mat ombreTotale = cv.add(ombreDir8u, ombreContact8u);

      // =========================================================================
      // === PHASE 5 : LUMIERE ET TEMPERATURE DE COULEUR INTELLIGENTE ===
      // =========================================================================
      final double reglageMixAmbiancePiece = 0.65;
      final double reglageMixCouleurMur = 0.35;    
      final double reglageTeinteEquipementBlanc = 0.30;
      final double reglageTeinteEquipementNoir = 0.08;   
      
      final double reglageInfluenceOmbreSurBlanc = 0.65;
      final double reglageInfluenceOmbreSurNoir = 0.21;  
      
      // AJUSTEMENT DE L'EBLOUISSEMENT : On baisse le multiplicateur de base et le plafond
      // pour éviter que l'équipement ou la goulotte ne brille trop sur un mur sans ombre.
      double ratioLuminositeAmbiante = (lumiereMoyenneMur / 128.0).clamp(0.7, 1.0);
      final double reglageLuminositeEquipementBlanc = 0.95 * ratioLuminositeAmbiante; 
      final double reglageLuminositeEquipementNoir = 0.78;   

      cv.Mat murUltraSmall = cv.resize(resultImg, (wMur ~/ 32, hMur ~/ 32), interpolation: cv.INTER_AREA);
      cv.Mat murUltraFlou = cv.gaussianBlur(murUltraSmall, (15, 15), 0.0);
      cv.Mat murLisse = cv.resize(murUltraFlou, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      cv.Scalar meanMurGlobal = cv.mean(murLisse);
      // La lumière est analysée sur le mur se trouvant derrière l'ENSEMBLE de la machine
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

      // La correction du contre-jour englobe le volume total, pas seulement la face
      cv.Rect rectEquipement = cv.boundingRect(cv.VecPoint.fromList(ptsDstLisses));
      int rx = math.max(0, rectEquipement.x - 20);
      int ry = math.max(0, rectEquipement.y - 20);
      int rw = math.min(wMur - rx, rectEquipement.width + 40);
      int rh = math.min(hMur - ry, rectEquipement.height + 40);
      
      double voileAtmospherique = 0.0;
      
      if (rw > 0 && rh > 0) {
        cv.Mat roiMurGray = grayMur.region(cv.Rect(rx, ry, rw, rh));
        var minMaxRoi = cv.minMaxLoc(roiMurGray);
        double minLocal = minMaxRoi.$1;
        double maxLocal = minMaxRoi.$2;
        
        double contrasteLocal = math.max(0.0, maxLocal - minLocal);
        
        double forceLevelLift = (ecartTypeMur / 40.0).clamp(0.3, 1.0);
        voileAtmospherique = (contrasteLocal * 0.15); 
        
        if (estEquipementNoir) {
           voileAtmospherique = math.min(voileAtmospherique, 8.0); 
        } else {
           voileAtmospherique = math.min(voileAtmospherique, 25.0); 
        }
        voileAtmospherique *= forceLevelLift;
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
      // === PHASE 6 : CREATION DU CALQUE PNG TRANSPARENT (SECURISE) ===
      // =========================================================================
      // On combine l'Alpha de l'équipement avec l'intensité de l'Ombre pour créer 
      // un PNG transparent autonome qu'on mettra en cache.
      
      cv.Mat alphaMaskF = alphaMaskTotal.convertTo(cv.MatType.CV_32FC1, alpha: 1.0 / 255.0);
      cv.Mat ombreF = ombreTotale.convertTo(cv.MatType.CV_32FC1, alpha: 1.0 / 255.0);
      
      // Sécurité : On utilise cv.Mat.zeros et setTo pour avoir un fond 1.0 parfait.
      cv.Mat matriceUnAlpha = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC1)..setTo(cv.Scalar.all(1.0));
      cv.Mat invAlphaMaskF = cv.subtract(matriceUnAlpha, alphaMaskF);
      cv.Mat shadowAlphaF = cv.multiply(invAlphaMaskF, ombreF);
      cv.Mat finalAlphaF = cv.add(alphaMaskF, shadowAlphaF);
      
      cv.Mat finalAlpha8u = finalAlphaF.convertTo(cv.MatType.CV_8UC1, alpha: 255.0);

      cv.Mat bgrFinal = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC3);
      equipementRgbFinal.copyTo(bgrFinal, mask: alphaMaskTotal);

      // Sécurité ultime pour le canal Alpha sans utiliser d'array dynamique (qui crashait)
      cv.Mat bgraFinal = cv.cvtColor(bgrFinal, cv.COLOR_BGR2BGRA);
      var bgraChannels = cv.split(bgraFinal);
      bgraChannels[3] = finalAlpha8u; // Remplacement direct du canal Alpha
      cv.Mat finalImage = cv.merge(bgraChannels);

      var encodeResult = cv.imencode('.png', finalImage);
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
      _tracerLigneRectangulaire(maskBinaire, p1, p2, cv.Scalar.all(255), largeurPx);
      if (lenCap >= 1.0) {
        cv.fillPoly(maskBinaire, cv.VecVecPoint.fromList([[ptLeft, ptRight, ptRightMur, ptLeftMur]]), cv.Scalar.all(255), lineType: cv.LINE_AA);
      }

      // 2. Construit l'apparence visuelle de la goulotte
      cv.Mat goulotteBgr = cv.Mat.zeros(h, w, cv.MatType.CV_8UC3);
      _tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(210, 210, 210, 0), largeurPx);
      _tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(240, 245, 245, 0), largeurPx * 0.85);
      _tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(245, 250, 250, 0), largeurPx * 0.50);
      _tracerLigneRectangulaire(goulotteBgr, p1, p2, cv.Scalar(255, 255, 255, 0), largeurPx * 0.15);
      
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
      cv.Mat invOmbreF = invOmbre3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      
      cv.Mat fondF = fondMat.convertTo(cv.MatType.CV_32FC3);
      cv.Mat fondOmbreF = cv.multiply(fondF, invOmbreF);
      cv.Mat murOmbre = fondOmbreF.convertTo(cv.MatType.CV_8UC3);

      // 4. Teinte et Lumière dynamiques (Même traitement que pour l'équipement)
      // CORRECTION MAJEURE : On utilise 'fondMat' (le mur vierge) et non 'murOmbre' !
      // Si on utilise 'murOmbre', la goulotte analyse sa PROPRE ombre et s'assombrit elle-même.
      cv.Mat murUltraSmall = cv.resize(fondMat, (w ~/ 32, h ~/ 32), interpolation: cv.INTER_AREA);
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
      
      // POUR REGLER LA LUMINOSITE (Influence du mur)
      // Si la goulotte est trop sombre sur un mur foncé, baisse cette variable vers 0.30 ou 0.20
      // pour que la goulotte "ignore" l'obscurité du mur en dessous d'elle.
      double influenceMurGoulotte = 0.43; 
      cv.Mat ratioSecurise = cv.addWeighted(ratioMap, influenceMurGoulotte, matriceUn, 1.0 - influenceMurGoulotte, 0.0);
      
      cv.Mat vChannelF = hsvChannels[2].convertTo(cv.MatType.CV_32FC1);
      cv.Mat vShadowedF = cv.multiply(vChannelF, ratioSecurise);

      // LUMINOSITE ADAPTATIVE POUR LA GOULOTTE
      // Comme pour l'équipement, on adapte l'exposition de la goulotte selon la lumière de la pièce.
      // Si le mur derrière la goulotte est sombre (ex: à l'ombre, 80/255), on baisse l'intensité lumineuse de la goulotte.
      double ratioLuminosite = (lumMurLocal / 128.0).clamp(0.65, 1.0);
      double luminositeGoulotteAdaptive = 0.95 * ratioLuminosite;
      
      vShadowedF = vShadowedF.convertTo(cv.MatType.CV_32FC1, alpha: luminositeGoulotteAdaptive);

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