import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class InpaintingCore {
  /// Isolate pour effacer l'autocollant de l'image.
  static Future<Uint8List?> effacerAutocollantIsolate(Map<String, dynamic> params) async {
    return await effacerAutocollant(
      photoPath: params['photoPath'] as String,
      pointsIA: (params['pointsIA'] as List).map((e) => Map<String, double>.from(e)).toList(),
      lamaBytes: params['lamaBytes'] as Uint8List?,
    );
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
      // === PRE-CALCUL DE LA ZONE À REPARER ===
      // =========================================================================
      // Elargit la zone de recadrage autour de l'autocollant
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

      // =========================================================================
      // === OPTIMISATION : SMART BYPASS 2.0 (Taille + Structure) ===
      // =========================================================================
      cv.Mat cropGray = cv.cvtColor(cropImg, cv.COLOR_BGR2GRAY);
      cv.Mat invCropMaskBypass = cv.bitwiseNOT(cropMaskLama);
      
      double surfaceTrou = (rect.width * rect.height).toDouble();
      double surfaceTotalePhoto = (wMur * hMur).toDouble(); 
      double ratioTrou = surfaceTrou / surfaceTotalePhoto; 
      
      cv.Mat grayFloute = cv.gaussianBlur(cropGray, (5, 5), 0.0);
      cv.Mat edges = cv.canny(grayFloute, 30.0, 100.0);
      
      var meanEdges = cv.mean(edges, mask: invCropMaskBypass);
      double densiteLignes = meanEdges.val[0]; 
      
      print("[IA Inpainting] Ratio trou (Photo) : ${(ratioTrou*100).toStringAsFixed(2)}% | Densité lignes : ${densiteLignes.toStringAsFixed(2)}");

      bool trouEstPetit = ratioTrou < 0.20;
      bool murSansLigneForte = densiteLignes < 44.0;

      if ((trouEstPetit && murSansLigneForte) && double.parse(densiteLignes.toStringAsFixed(2)) < 1.00) {
        print("[IA Inpainting] Trou petit ou sans ligne détecté ! Bypass IA -> OpenCV (~5ms).");
        
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

        cropRect = cv.Rect(rectX, rectY, rectW, rectH); 
        cv.Mat patch = murMat.region(cv.Rect(srcX, srcY, rectW, rectH)).clone();
        
        cv.Mat patchFloat = patch.convertTo(cv.MatType.CV_32FC3);
        cv.Mat noisePatch = cv.Mat.zeros(rectH, rectW, cv.MatType.CV_32FC3);
        cv.randn(noisePatch, cv.Scalar.all(0.0), cv.Scalar.all(3.0)); 
        cv.Mat patchNoisyFloat = cv.add(patchFloat, noisePatch);
        cv.Mat finalPatch = patchNoisyFloat.convertTo(cv.MatType.CV_8UC3);

        finalPatch.copyTo(murRepare.region(cropRect));
        inpaintingReussi = true;
      } 
      // =========================================================================
      // === TENTATIVE AVEC IA (LAMA) SI LE MUR EST TRÈS TEXTURÉ / COMPLEXE ===
      // =========================================================================
      else if (lamaBytes != null) {
        try {
          print("[IA Inpainting] Mur texturé détecté. Démarrage de l'analyse LaMa...");

          cv.Scalar couleurMoyenne = cv.mean(cropImg, mask: invCropMaskBypass);
          cropImg.setTo(couleurMoyenne, mask: cropMaskLama);

          // RETOUR AU 512x512 (Ultra Stable)
          cv.Mat img512 = cv.resize(cropImg, (512, 512));
          cv.Mat mask512 = cv.resize(cropMaskLama, (512, 512));
          
          cv.Mat imgRGB = cv.cvtColor(img512, cv.COLOR_BGR2RGB);
          Uint8List rgbBytes = imgRGB.data;
          Uint8List maskBytes = mask512.data;

          var inputImgFlat = Float32List(512 * 512 * 3);
          var inputMaskFlat = Float32List(512 * 512 * 1);
          
          int imgIdx = 0;
          int maskIdx = 0;
          int rgbIdx = 0;
          
          for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
              inputImgFlat[imgIdx++] = rgbBytes[rgbIdx] / 255.0;     
              inputImgFlat[imgIdx++] = rgbBytes[rgbIdx+1] / 255.0;   
              inputImgFlat[imgIdx++] = rgbBytes[rgbIdx+2] / 255.0;   
              
              inputMaskFlat[maskIdx++] = maskBytes[y * 512 + x] > 127 ? 1.0 : 0.0;
              rgbIdx += 3;
            }
          }

          // =========================================================================
          // === SÉCURITÉ ISOLATE : DÉLÉGUÉ XNNPACK (CPU OPTIMISÉ MULTI-COEURS) ======
          // =========================================================================
          InterpreterOptions options = InterpreterOptions();
          
          // Dans un Isolate, les drivers GPU crashent nativement (SIGSEGV).
          // On force l'utilisation de XNNPack qui est totalement Thread-Safe.
          options.addDelegate(XNNPackDelegate());
          
          // On compense l'absence du GPU en forçant l'utilisation de 4 coeurs du processeur
          options.threads = 4;

          Interpreter interpreter = Interpreter.fromBuffer(lamaBytes, options: options);
          var tensor0 = interpreter.getInputTensor(0);
          
          List<Object> inputs = (tensor0.shape.last == 3) 
              ? [inputImgFlat.buffer, inputMaskFlat.buffer] 
              : [inputMaskFlat.buffer, inputImgFlat.buffer];
          
          var outputImgFlat = Float32List(512 * 512 * 3);
          interpreter.runForMultipleInputs(inputs, {0: outputImgFlat.buffer});

          Uint8List outBytes = Uint8List(512 * 512 * 3);
          int outIdx = 0;
          int flatIdx = 0;
          
          for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
              outBytes[outIdx]   = (outputImgFlat[flatIdx + 2] * 255).clamp(0, 255).toInt();   
              outBytes[outIdx+1] = (outputImgFlat[flatIdx + 1] * 255).clamp(0, 255).toInt();   
              outBytes[outIdx+2] = (outputImgFlat[flatIdx]     * 255).clamp(0, 255).toInt();   
              outIdx += 3;
              flatIdx += 3;
            }
          }

          img.Image repairedImg = img.Image.fromBytes(width: 512, height: 512, bytes: outBytes.buffer, order: img.ChannelOrder.bgr);
          Uint8List jpgBytes = img.encodeJpg(repairedImg, quality: 100);
          cv.Mat patch512 = cv.imdecode(jpgBytes, cv.IMREAD_COLOR);
          
          cv.Mat patchFinal = cv.resize(patch512, (cropS, cropS), interpolation: cv.INTER_LANCZOS4);
          
          cv.Mat blurredPatch = cv.gaussianBlur(patchFinal, (0, 0), 2.0); 
          cv.Mat patchNet = cv.addWeighted(patchFinal, 1.5, blurredPatch, -0.5, 0.0);
          
          cv.Mat patchFloat = patchNet.convertTo(cv.MatType.CV_32FC3);
          cv.Mat noisePatch = cv.Mat.zeros(cropS, cropS, cv.MatType.CV_32FC3);
          cv.randn(noisePatch, cv.Scalar.all(0.0), cv.Scalar.all(4.5)); 
          cv.Mat patchNoisyFloat = cv.add(patchFloat, noisePatch);
          patchNet = patchNoisyFloat.convertTo(cv.MatType.CV_8UC3);

          patchNet.copyTo(murRepare.region(cropRect));
          
          interpreter.close();
          inpaintingReussi = true;
          print("[IA Inpainting] LaMa a rebouché le trou avec succès en 512x512 via GPU !");
          
        } catch (e) {
          print("[IA Inpainting] Echec de LaMa, passage au Tampon OpenCV : $e");
        }
      }

      // METHODE DE SECOURS (Si l'IA plante ET qu'on n'a pas utilisé le bypass)
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

        cropRect = cv.Rect(rectX, rectY, rectW, rectH);
        cv.Mat patch = murMat.region(cv.Rect(srcX, srcY, rectW, rectH));
        patch.copyTo(murRepare.region(cropRect));
      }

      // FUSION DES BORDS (FEATHERING) : Fusionne en douceur les bords du patch réparé avec le mur d'origine.
      cv.Mat resultImg = murRepare.clone();
      
      try {
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

          petitResultImg.copyTo(resultImg.region(cropRect));
        }
      } catch (e) {
        print("[OpenCV] Erreur lors du blending : $e");
        resultImg = murRepare; // Sécurité
      }

      var encodeResult = cv.imencode('.jpg', resultImg);
      return encodeResult.$2;
      
    } catch (e) {
      print("[OpenCV] Erreur lors de la génération du fond propre : $e");
      return null;
    }
  }
}