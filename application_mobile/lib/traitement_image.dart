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
      lamaBytes: params['lamaBytes'] as Uint8List?, // NOUVEAU
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
      lamaBytes: params['lamaBytes'] as Uint8List?, // NOUVEAU
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

      // NOUVEAU : DILATATION POUR L'IA (On gonfle le masque avec une marge de 25px pour avaler les bords verts)
      cv.Mat kernelLama = cv.Mat.ones(25, 25, cv.MatType.CV_8UC1);
      cv.Mat maskLama = cv.dilate(maskGeo, kernelLama, iterations: 1); 

      // Épaissit légèrement le masque pour assurer une couverture complète.
      cv.Mat kernelDilate = cv.Mat.ones(15, 15, cv.MatType.CV_8UC1);
      cv.Mat maskTransition = cv.dilate(maskLama, kernelDilate, iterations: 2);
      
      // Calcule le rectangle qui entoure l'autocollant.
      cv.Rect rect = cv.boundingRect(cv.VecPoint.fromList(ptsOri));
      
      cv.Mat murRepare = murMat.clone();
      bool inpaintingReussi = false;

      // =========================================================================
      // === PHASE 1 : EFFACEMENT DU DEFAUT (LAMA INPAINTING ou CLONE STAMP) =====
      // =========================================================================

      // TENTATIVE AVEC IA (LaMa)
      if (lamaBytes != null) {
        try {
          print("[IA Inpainting] Démarrage de l'analyse LaMa...");
          
          // 1. Découpage d'un carré large autour de l'autocollant (pour donner du contexte à l'IA)
          int cropS = (math.max(rect.width, rect.height) * 2.2).toInt();
          int cropX = (rect.x + rect.width / 2 - cropS / 2).toInt();
          int cropY = (rect.y + rect.height / 2 - cropS / 2).toInt();
          
          // Sécurisation des bords
          if (cropX < 0) cropX = 0;
          if (cropY < 0) cropY = 0;
          if (cropX + cropS > wMur) cropX = wMur - cropS;
          if (cropY + cropS > hMur) cropY = hMur - cropS;
          if (cropS > wMur) cropS = wMur;
          if (cropS > hMur) cropS = hMur;

          cv.Rect cropRect = cv.Rect(cropX, cropY, cropS, cropS);
          cv.Mat cropImg = murMat.region(cropRect).clone(); // CLONE indispensable pour ne pas altérer l'original
          cv.Mat cropMaskLama = maskLama.region(cropRect);

          // =================================================================================
          // L'ASTUCE OPENCV + LAMA : On détruit l'autocollant vert AVANT le redimensionnement
          // =================================================================================
          cv.Mat invCropMask = cv.bitwiseNOT(cropMaskLama);
          cv.Scalar couleurMoyenne = cv.mean(cropImg, mask: invCropMask);
          
          // OpenCV remplace le vert par la couleur moyenne du mur. 
          // Ainsi, le vert ne bave pas dans les pixels sains pendant le cv.resize() !
          cropImg.setTo(couleurMoyenne, mask: cropMaskLama);
          // =================================================================================

          // 2. Redimensionnement au standard LaMa (512x512)
          cv.Mat img512 = cv.resize(cropImg, (512, 512));
          cv.Mat mask512 = cv.resize(cropMaskLama, (512, 512));
          
          // LaMa exige du RGB, OpenCV utilise du BGR
          cv.Mat imgRGB = cv.cvtColor(img512, cv.COLOR_BGR2RGB);

          Uint8List rgbBytes = imgRGB.data;
          Uint8List maskBytes = mask512.data;

          // 3. Préparation des tenseurs pour TFLite
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

          // 4. Inférence LaMa
          Interpreter interpreter = Interpreter.fromBuffer(lamaBytes);
          
          // Analyse dynamique de l'ordre d'entrée des tenseurs (Image d'abord ou Masque d'abord)
          var tensor0 = interpreter.getInputTensor(0);
          List<Object> inputs = (tensor0.shape.last == 3) ? [inputImg, inputMask] : [inputMask, inputImg];

          var outputImg = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => List.generate(3, (l) => 0.0))));
          interpreter.runForMultipleInputs(inputs, {0: outputImg});

          // 5. Reconstruction de l'image (Retour en BGR pour OpenCV)
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

          // On passe par un encodage JPEG pour garantir une conversion robuste vers OpenCV
          img.Image repairedImg = img.Image.fromBytes(width: 512, height: 512, bytes: outBytes.buffer, order: img.ChannelOrder.bgr);
          Uint8List jpgBytes = img.encodeJpg(repairedImg, quality: 100);
          cv.Mat patch512 = cv.imdecode(jpgBytes, cv.IMREAD_COLOR);
          
          // On remet la zone à sa taille d'origine et on la colle
          cv.Mat patchFinal = cv.resize(patch512, (cropS, cropS));
          patchFinal.copyTo(murRepare.region(cropRect));
          
          interpreter.close();
          inpaintingReussi = true;
          print("[IA Inpainting] LaMa a rebouché le trou avec succès !");
          
        } catch (e) {
          print("[IA Inpainting] Échec de LaMa, passage au Tampon OpenCV : $e");
        }
      }

      // FALLBACK : SI L'IA N'EST PAS LA OU A PLANTE (TAMPON DE DUPLICATION)
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

        cv.Mat patch = murMat.region(cv.Rect(srcX, srcY, rectW, rectH));
        patch.copyTo(murRepare.region(cv.Rect(rectX, rectY, rectW, rectH)));
      }

      // Crée un dégradé (feathering) sur les bords du patch pour une transition douce.
      cv.Mat smallMask = cv.resize(maskTransition, (wMur ~/ 4, hMur ~/ 4));
      cv.Mat smallMaskFlou = cv.gaussianBlur(smallMask, (13, 13), 0.0);
      cv.Mat maskFeather8u = cv.resize(smallMaskFlou, (wMur, hMur));

      // =================================================================================
      // FIX ANTI-FANTÔME OPENCV : On force l'opacité à 100% au centre du masque flou.
      // Cela garantit qu'aucun pixel de l'autocollant original ne survivra au fondu alpha !
      // =================================================================================
      cv.Mat maskFeatherSecurise = cv.bitwiseOR(maskFeather8u, maskLama);

      cv.Mat maskFeather3c = cv.cvtColor(maskFeatherSecurise, cv.COLOR_GRAY2BGR);
      cv.Mat maskFeatherF = maskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat invMaskFeather8u = cv.bitwiseNOT(maskFeatherSecurise);
      cv.Mat invMaskFeather3c = cv.cvtColor(invMaskFeather8u, cv.COLOR_GRAY2BGR);
      cv.Mat invMaskFeatherF = invMaskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat murRepareF = murRepare.convertTo(cv.MatType.CV_32FC3);
      cv.Mat murOriginalF = murMat.convertTo(cv.MatType.CV_32FC3);

      cv.Mat fgInpaint = cv.multiply(murRepareF, maskFeatherF);
      cv.Mat bgInpaint = cv.multiply(murOriginalF, invMaskFeatherF);
      
      cv.Mat resultImgF = cv.add(fgInpaint, bgInpaint);
      cv.Mat resultImg = resultImgF.convertTo(cv.MatType.CV_8UC3);

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
    required String photoPath,
    required Uint8List climBytes,
    required List<Map<String, double>> pointsIA,
    double decalageX = 0.0,
    double decalageY = 0.0, 
    required String climAssetPath,
    required Uint8List? lamaBytes, // NOUVEAU
  }) async {
    try {
      cv.Mat murMat = cv.imread(photoPath, flags: cv.IMREAD_COLOR);
      int wMur = murMat.cols;
      int hMur = murMat.rows;

      // Calcule les ratios pour redimensionner les points de l'IA.
      double ratioX = wMur / 1024.0;
      double ratioY = hMur / 1024.0;

      // Convertit les points de l'IA en points OpenCV.
      List<cv.Point> ptsOri = pointsIA.map((pt) {
        return cv.Point((pt['x']! * ratioX).toInt(), (pt['y']! * ratioY).toInt());
      }).toList();

      // Crée un masque de la forme de l'autocollant.
      cv.Mat maskGeo = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_8UC1);
      cv.fillPoly(maskGeo, cv.VecVecPoint.fromList([ptsOri]), cv.Scalar.all(255));

      // NOUVEAU : DILATATION POUR L'IA (On gonfle le masque avec une marge de 25px pour avaler les bords verts)
      cv.Mat kernelLama = cv.Mat.ones(25, 25, cv.MatType.CV_8UC1);
      cv.Mat maskLama = cv.dilate(maskGeo, kernelLama, iterations: 1); 

      // Épaissit le masque pour les transitions.
      cv.Mat kernelDilate = cv.Mat.ones(15, 15, cv.MatType.CV_8UC1);
      cv.Mat maskTransition = cv.dilate(maskLama, kernelDilate, iterations: 2); 

      cv.Rect rect = cv.boundingRect(cv.VecPoint.fromList(ptsOri));
      cv.Mat murRepare = murMat.clone();
      bool inpaintingReussi = false;

      // =========================================================================
      // === PHASE 1 : EFFACEMENT DU DEFAUT (LAMA INPAINTING ou CLONE STAMP) =====
      // =========================================================================

      if (lamaBytes != null) {
        try {
          int cropS = (math.max(rect.width, rect.height) * 2.2).toInt();
          int cropX = (rect.x + rect.width / 2 - cropS / 2).toInt();
          int cropY = (rect.y + rect.height / 2 - cropS / 2).toInt();
          
          if (cropX < 0) cropX = 0;
          if (cropY < 0) cropY = 0;
          if (cropX + cropS > wMur) cropX = wMur - cropS;
          if (cropY + cropS > hMur) cropY = hMur - cropS;
          if (cropS > wMur) cropS = wMur;
          if (cropS > hMur) cropS = hMur;

          cv.Rect cropRect = cv.Rect(cropX, cropY, cropS, cropS);
          cv.Mat cropImg = murMat.region(cropRect).clone(); // CLONE IMPORTANT
          cv.Mat cropMaskLama = maskLama.region(cropRect);

          // =================================================================================
          // L'ASTUCE OPENCV + LAMA : Pré-remplissage pour éviter de faire baver le vert
          // =================================================================================
          cv.Mat invCropMask = cv.bitwiseNOT(cropMaskLama);
          cv.Scalar couleurMoyenne = cv.mean(cropImg, mask: invCropMask);
          cropImg.setTo(couleurMoyenne, mask: cropMaskLama);
          // =================================================================================

          cv.Mat img512 = cv.resize(cropImg, (512, 512));
          cv.Mat mask512 = cv.resize(cropMaskLama, (512, 512));
          cv.Mat imgRGB = cv.cvtColor(img512, cv.COLOR_BGR2RGB);

          Uint8List rgbBytes = imgRGB.data;
          Uint8List maskBytes = mask512.data;

          var inputImg = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => List.generate(3, (l) => 0.0))));
          var inputMask = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => List.generate(1, (l) => 0.0))));

          int idx = 0;
          for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
              inputImg[0][y][x][0] = rgbBytes[idx] / 255.0; 
              inputImg[0][y][x][1] = rgbBytes[idx+1] / 255.0; 
              inputImg[0][y][x][2] = rgbBytes[idx+2] / 255.0; 
              inputMask[0][y][x][0] = maskBytes[y * 512 + x] > 127 ? 1.0 : 0.0;
              idx += 3;
            }
          }

          Interpreter interpreter = Interpreter.fromBuffer(lamaBytes);
          var tensor0 = interpreter.getInputTensor(0);
          List<Object> inputs = (tensor0.shape.last == 3) ? [inputImg, inputMask] : [inputMask, inputImg];

          var outputImg = List.generate(1, (i) => List.generate(512, (j) => List.generate(512, (k) => List.generate(3, (l) => 0.0))));
          interpreter.runForMultipleInputs(inputs, {0: outputImg});

          Uint8List outBytes = Uint8List(512 * 512 * 3);
          int outIdx = 0;
          for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
              outBytes[outIdx] = (outputImg[0][y][x][2] * 255).clamp(0, 255).toInt(); 
              outBytes[outIdx+1] = (outputImg[0][y][x][1] * 255).clamp(0, 255).toInt(); 
              outBytes[outIdx+2] = (outputImg[0][y][x][0] * 255).clamp(0, 255).toInt(); 
              outIdx += 3;
            }
          }

          img.Image repairedImg = img.Image.fromBytes(width: 512, height: 512, bytes: outBytes.buffer, order: img.ChannelOrder.bgr);
          Uint8List jpgBytes = img.encodeJpg(repairedImg, quality: 100);
          cv.Mat patch512 = cv.imdecode(jpgBytes, cv.IMREAD_COLOR);
          
          cv.Mat patchFinal = cv.resize(patch512, (cropS, cropS));
          patchFinal.copyTo(murRepare.region(cropRect));
          
          interpreter.close();
          inpaintingReussi = true;
        } catch (e) {
          print("[IA Inpainting] Échec, passage au Tampon : $e");
        }
      }

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

        cv.Mat patch = murMat.region(cv.Rect(srcX, srcY, rectW, rectH));
        patch.copyTo(murRepare.region(cv.Rect(rectX, rectY, rectW, rectH)));
      }

      // Crée un dégradé pour une transition douce entre la zone clonée et le reste de l'image.
      cv.Mat smallMask = cv.resize(maskTransition, (wMur ~/ 4, hMur ~/ 4));
      cv.Mat smallMaskFlou = cv.gaussianBlur(smallMask, (13, 13), 0.0);
      cv.Mat maskFeather8u = cv.resize(smallMaskFlou, (wMur, hMur));

      // =================================================================================
      // FIX ANTI-FANTÔME OPENCV : On force l'opacité à 100% au centre.
      // =================================================================================
      cv.Mat maskFeatherSecurise = cv.bitwiseOR(maskFeather8u, maskLama);

      cv.Mat maskFeather3c = cv.cvtColor(maskFeatherSecurise, cv.COLOR_GRAY2BGR);
      cv.Mat maskFeatherF = maskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat invMaskFeather8u = cv.bitwiseNOT(maskFeatherSecurise);
      cv.Mat invMaskFeather3c = cv.cvtColor(invMaskFeather8u, cv.COLOR_GRAY2BGR);
      cv.Mat invMaskFeatherF = invMaskFeather3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

      cv.Mat murRepareF = murRepare.convertTo(cv.MatType.CV_32FC3);
      cv.Mat murOriginalF = murMat.convertTo(cv.MatType.CV_32FC3);

      cv.Mat fgInpaint = cv.multiply(murRepareF, maskFeatherF);
      cv.Mat bgInpaint = cv.multiply(murOriginalF, invMaskFeatherF);
      
      cv.Mat resultImgF = cv.add(fgInpaint, bgInpaint);
      cv.Mat resultImg = resultImgF.convertTo(cv.MatType.CV_8UC3);

      // Calcule la perspective et l'angle de l'autocollant pour stabiliser l'image.
      cv.Point ptHg = ptsOri[0];
      cv.Point ptHd = ptsOri[1];

      double dx = (ptHd.x - ptHg.x).toDouble();
      double dy = (ptHd.y - ptHg.y).toDouble();
      double largeurPx = math.sqrt(dx * dx + dy * dy);
      double angleRad = math.atan2(dy, dx);

      // Dimensions réelles de l'autocollant en millimètres.
      double hAutoMm = 100.0;
      double wAutoMm = 50.0; 
      double ratioPhysique = hAutoMm / wAutoMm; 
      double hauteurPx = largeurPx * ratioPhysique;

      // Calcule les vecteurs de direction pour la perspective.
      double ux = largeurPx * math.cos(angleRad);
      double uy = largeurPx * math.sin(angleRad);
      double vx = -hauteurPx * math.sin(angleRad);
      double vy = hauteurPx * math.cos(angleRad);

      // Calcule les 4 points de destination pour la déformation, en incluant le décalage manuel.
      List<cv.Point> ptsDstLisses = [
        cv.Point((ptHg.x + decalageX).toInt(), (ptHg.y + decalageY).toInt()),
        cv.Point((ptHg.x + ux + decalageX).toInt(), (ptHg.y + uy + decalageY).toInt()),
        cv.Point((ptHg.x + ux + vx + decalageX).toInt(), (ptHg.y + uy + vy + decalageY).toInt()),
        cv.Point((ptHg.x + vx + decalageX).toInt(), (ptHg.y + vy + decalageY).toInt())
      ];

      // Applique la déformation de perspective à l'image de la climatisation.
      cv.Mat climMat = cv.imdecode(climBytes, cv.IMREAD_UNCHANGED);

      // Dimensions réelles de la climatisation en millimètres.
      double hClimMm = 270.0; 
      double wClimMm = 798.0; 
      int wImgClim = climMat.cols;
      int hImgClim = climMat.rows;

      // Calcule la taille virtuelle de l'autocollant sur l'image de la climatisation.
      double wAutoPx = (wAutoMm / wClimMm) * wImgClim;
      double hAutoPx = (hAutoMm / hClimMm) * hImgClim;

      // Points source sur l'image de la climatisation (correspondant à l'autocollant virtuel).
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
      cv.Mat alphaMask = cv.gaussianBlur(alphaMaskOriginale, (3, 3), 0.0);
      
      cv.Mat climBgr = cv.cvtColor(climWarped, cv.COLOR_BGRA2BGR);
      cv.Mat maskBinaire = cv.threshold(alphaMask, 5, 255, cv.THRESH_BINARY).$2;

      // =========================================================================
      // === PHASE 4 : CALCUL DE LA DIRECTION DE LA LUMIÈRE & OMBRE PROGRESSIVE ===
      // =========================================================================
      cv.Mat grayMur = cv.cvtColor(resultImg, cv.COLOR_BGR2GRAY);
      
      // Utilise un filtre de Sobel pour détecter les gradients de lumière sur le mur.
      cv.Mat grayMurFlou = cv.gaussianBlur(grayMur, (51, 51), 0.0);
      cv.Mat sobelX = cv.sobel(grayMurFlou, cv.MatType.CV_32F, 1, 0, ksize: 3);
      cv.Mat sobelY = cv.sobel(grayMurFlou, cv.MatType.CV_32F, 0, 1, ksize: 3);

      cv.Scalar meanSobelX = cv.mean(sobelX, mask: maskBinaire);
      cv.Scalar meanSobelY = cv.mean(sobelY, mask: maskBinaire);

      double gradX = meanSobelX.val[0];
      double gradY = meanSobelY.val[0];

      // Normalise le vecteur pour obtenir la direction de la lumière.
      double norme = math.sqrt(gradX * gradX + gradY * gradY) + 0.0001; 
      double dirLumiereX = gradX / norme;
      double dirLumiereY = gradY / norme;

      // Génère une ombre portée en fonction de la direction de la lumière détectée.
      double forceOmbre = 20.0; 
      double shiftX = norme < 1.0 ? 5.0 : -dirLumiereX * forceOmbre;
      double shiftY = norme < 1.0 ? 15.0 : -dirLumiereY * forceOmbre;

      // Crée une ombre déformée et floutée (Directionnelle)
      List<cv.Point> ptsDstOmbre = ptsDstLisses.map((pt) {
        return cv.Point((pt.x + shiftX).toInt(), (pt.y + shiftY).toInt());
      }).toList();
      
      cv.Mat hMatrixOmbre = cv.getPerspectiveTransform(vecPtsSrc, cv.VecPoint.fromList(ptsDstOmbre));
      cv.Mat climWarpedOmbre = cv.warpPerspective(climMat, hMatrixOmbre, (wMur, hMur));
      cv.Mat alphaOmbre = cv.split(climWarpedOmbre)[3];
      cv.Mat smallAlpha = cv.resize(alphaOmbre, (wMur ~/ 4, hMur ~/ 4));
      cv.Mat smallOmbreFloue = cv.gaussianBlur(smallAlpha, (21, 21), 0.0);
      cv.Mat ombreFloueUpscaled = cv.resize(smallOmbreFloue, (wMur, hMur), interpolation: cv.INTER_CUBIC);
      cv.Mat ombreFloueDirectionnelle = cv.gaussianBlur(ombreFloueUpscaled, (15, 15), 0.0);

      // NOUVEAU : OMBRE DE CONTACT (Ambient Occlusion) pour ancrer la clim au mur (Évite l'effet PNG)
      // On décale très légèrement vers le bas (3 pixels) avec un flou très sec pour créer le point de contact
      List<cv.Point> ptsDstContact = ptsDstLisses.map((pt) => cv.Point(pt.x, pt.y + 3)).toList();
      cv.Mat hMatrixContact = cv.getPerspectiveTransform(vecPtsSrc, cv.VecPoint.fromList(ptsDstContact));
      cv.Mat climWarpedContact = cv.warpPerspective(climMat, hMatrixContact, (wMur, hMur));
      cv.Mat alphaContact = cv.split(climWarpedContact)[3];
      cv.Mat ombreFloueContact = cv.gaussianBlur(alphaContact, (7, 7), 0.0);

      // Applique l'ombre sur le mur avec une intensité basée sur la luminosité globale.
      cv.Scalar lumMurGlobal = cv.mean(grayMur);
      
      // On diminue un peu l'intensité directionnelle car on ajoute le contact
      double intensiteDir = 0.06 + (lumMurGlobal.val[0] / 255.0) * 0.20;
      double intensiteContact = 0.15; // Le contact est toujours sombre

      cv.Mat ombreDir8u = ombreFloueDirectionnelle.convertTo(cv.MatType.CV_8UC1, alpha: intensiteDir);
      cv.Mat ombreContact8u = ombreFloueContact.convertTo(cv.MatType.CV_8UC1, alpha: intensiteContact);
      
      // Addition mathématique des deux ombres (le croisement des deux est plus noir)
      cv.Mat ombreTotale = cv.add(ombreDir8u, ombreContact8u);
      cv.Mat invOmbre8u = cv.bitwiseNOT(ombreTotale);
      cv.Mat invOmbre3c = cv.cvtColor(invOmbre8u, cv.COLOR_GRAY2BGR);
      
      cv.Mat invOmbreF = invOmbre3c.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
      cv.Mat resultImgForShadow = resultImg.convertTo(cv.MatType.CV_32FC3);
      cv.Mat murOmbreF = cv.multiply(resultImgForShadow, invOmbreF);
      
      cv.Mat murOmbre = murOmbreF.convertTo(cv.MatType.CV_8UC3);

      // =========================================================================
      // === PHASE 5 : LUMIÈRE ET TEMPÉRATURE DE COULEUR (AVEC FIX CLIM NOIRE) ===
      // =========================================================================

      // NOUVEAU : Auto-Détection de la couleur native de la climatisation (Blanche ou Noire ?)
      cv.Scalar meanClimColorO = cv.mean(climBgr, mask: maskBinaire);
      double lumaNativeClim = (0.114 * meanClimColorO.val[0]) + (0.587 * meanClimColorO.val[1]) + (0.299 * meanClimColorO.val[2]);
      bool estClimNoire = lumaNativeClim < 80.0; // Seuil pour différencier le plastique blanc du plastique noir
      
      // Analyse la couleur du mur sous la clim pour teinter légèrement l'unité.
      cv.Scalar meanMurSousClim = cv.mean(murOmbre, mask: maskBinaire);
      double bMur = meanMurSousClim.val[0];
      double gMur = meanMurSousClim.val[1];
      double rMur = meanMurSousClim.val[2];

      double lumMurLocal = (0.114 * bMur) + (0.587 * gMur) + (0.299 * rMur);
      lumMurLocal = math.max(lumMurLocal, 1.0); 

      // Calcule la teinte (color cast) à appliquer.
      double tintB = bMur / lumMurLocal;
      double tintG = gMur / lumMurLocal;
      double tintR = rMur / lumMurLocal;

      // NOUVEAU : Une clim noire reflète la lumière (specular) mais n'absorbe pas la couleur (diffuse)
      double forceTeinte = estClimNoire ? 0.10 : 0.35; 
      tintB = 1.0 + (tintB - 1.0) * forceTeinte;
      tintG = 1.0 + (tintG - 1.0) * forceTeinte;
      tintR = 1.0 + (tintR - 1.0) * forceTeinte;

      // Applique un filtre de couleur pour que la clim s'intègre mieux.
      cv.Mat tintMat = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC3)..setTo(cv.Scalar(tintB, tintG, tintR, 0));
      cv.Mat climF = climBgr.convertTo(cv.MatType.CV_32FC3);
      cv.Mat climTintedF = cv.multiply(climF, tintMat);
      cv.Mat climTinted = climTintedF.convertTo(cv.MatType.CV_8UC3);

      cv.Mat climHsv = cv.cvtColor(climTinted, cv.COLOR_BGR2HSV);
      var hsvChannels = cv.split(climHsv);
      
      // Calcule la luminosité globale de la pièce.
      double lumPiece = lumMurGlobal.val[0] / 255.0; // Ratio de 0.0 à 1.0
      
      // Ajuste la luminosité de la clim en fonction de la luminosité de la pièce.
      double ratioAdouci;
      if (estClimNoire) {
         // La clim noire doit rester sombre en base, mais monter légèrement si la pièce est inondée de lumière
         ratioAdouci = 0.90 + (lumPiece * 0.20);
      } else {
         // Une pièce claire garde la clim blanche, une pièce sombre l'assombrit (ancienne logique blanche)
         ratioAdouci = 0.60 + (lumPiece * 0.50); 
      }
      ratioAdouci = math.max(0.40, math.min(1.1, ratioAdouci)); 

      // Ajuste le contraste et l'exposition de la clim pour correspondre au mur.
      var minMaxMur = cv.minMaxLoc(grayMur);
      double niveauNoirMur = minMaxMur.$1; 
      niveauNoirMur = math.min(niveauNoirMur, 40.0); // Sécurité pour éviter le brouillard gris

      double ratioContraste = ratioAdouci * ((255.0 - niveauNoirMur) / 255.0);
      cv.Mat vScaled = cv.addWeighted(hsvChannels[2], ratioContraste, hsvChannels[2], 0.0, niveauNoirMur);

      // Crée une carte de la lumière ambiante pour simuler les ombres structurelles de la pièce sur la clim.
      cv.Mat smallGris = cv.resize(grayMur, (wMur ~/ 8, hMur ~/ 8), interpolation: cv.INTER_AREA);
      
      // Un flou important permet d'isoler la variation de lumière globale des détails du mur (motifs, peinture).
      cv.Mat smallFlouLumiere = cv.gaussianBlur(smallGris, (41, 41), 0.0);
      cv.Mat carteLumiere8u = cv.resize(smallFlouLumiere, (wMur, hMur), interpolation: cv.INTER_CUBIC);

      cv.Mat carteLumiereF = carteLumiere8u.convertTo(cv.MatType.CV_32FC1);
      
      // Calcule la lumière moyenne sous la clim pour normaliser.
      cv.Scalar moyenneLumiere = cv.mean(carteLumiereF, mask: maskBinaire);
      double baseLumiere = math.max(moyenneLumiere.val[0], 1.0);

      cv.Mat ratioMap = carteLumiereF.convertTo(cv.MatType.CV_32FC1, alpha: 1.0 / baseLumiere);

      // Limite l'assombrissement pour éviter les artefacts (effet "fantôme").
      cv.Mat matriceUn = cv.Mat.zeros(hMur, wMur, cv.MatType.CV_32FC1)..setTo(cv.Scalar.all(1.0));
      cv.Mat ratioSecurise = cv.addWeighted(ratioMap, 0.25, matriceUn, 0.75, 0.0);

      cv.Mat vChannelF = vScaled.convertTo(cv.MatType.CV_32FC1);
      cv.Mat vShadowedF = cv.multiply(vChannelF, ratioSecurise);
      hsvChannels[2] = vShadowedF.convertTo(cv.MatType.CV_8UC1);

      cv.Mat climHsvFinal = cv.merge(hsvChannels);
      cv.Mat climRgbFinal = cv.cvtColor(climHsvFinal, cv.COLOR_HSV2BGR);

      // Fusionne l'image de la climatisation traitée avec le mur (avec son ombre).
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

      // Encode l'image finale en JPEG.
      var encodeResult = cv.imencode('.jpg', resultatFinal);
      return encodeResult.$2;

    } catch (e) {
      print("[OpenCV] ERREUR FATALE : $e");
      return null;
    }
  }
}