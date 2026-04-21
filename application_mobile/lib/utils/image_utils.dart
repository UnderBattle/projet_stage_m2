import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

/// Redimensionne une image si elle dépasse une taille maximale pour éviter de surcharger la mémoire.
/// Conçu pour être exécuté dans un Isolate.
Future<String?> redimensionnerImageLourde(String imagePath) async {
  try {
    final imageBytes = await File(imagePath).readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);
    
    if (image == null) return null;

    int maxSize = 1920; 
    // Vérifie si l'image dépasse la limite autorisée.
    if (image.width > maxSize || image.height > maxSize) {
      print("[Optimisation] L'image est trop grande (${image.width}x${image.height}). Redimensionnement...");
      
      // Calcule le nouveau ratio en gardant les proportions de l'image.
      img.Image resized;
      if (image.width > image.height) {
        resized = img.copyResize(image, width: maxSize);
      } else {
        resized = img.copyResize(image, height: maxSize);
      }

      // Sauvegarde l'image redimensionnée dans un fichier temporaire.
      final directory = await getTemporaryDirectory();
      final path = '${directory.path}/photo_optimisee_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final newFile = File(path);
      
      await newFile.writeAsBytes(img.encodeJpg(resized, quality: 90));
      print("[Optimisation] Nouvelle taille ${resized.width}x${resized.height} prête !");
      return path;
    }
    
    print("[Optimisation] Taille correcte (${image.width}x${image.height}), pas de changement.");
    return imagePath;
    
  } catch (e) {
    print("Erreur d'optimisation image : $e");
    return imagePath; 
  }
}

/// Prépare l'image pour l'analyse par le modèle IA (TFLite).
/// Redimensionne l'image en 1024x1024 et la convertit en matrice de pixels normalisés.
Map<String, dynamic>? prepareImageMatrixForIA(Map<String, dynamic> params) {
  Uint8List imageBytes = params['bytes'];
  bool isNHWC = params['isNHWC'];

  img.Image? originalImage = img.decodeImage(imageBytes);
  if (originalImage == null) return null;

  int w = originalImage.width;
  int h = originalImage.height;

  // Force la taille de l'image à celle requise par le modèle YOLO.
  img.Image resizedImage = img.copyResize(originalImage, width: 1024, height: 1024);

  List<dynamic> inputMatrix;
  
  // Convertit l'image selon le format attendu par le modèle : NHWC (Haut, Largeur, Canaux) ou NCHW (Canaux, Haut, Largeur).
  if (isNHWC) {
    inputMatrix = List.generate(1, (i) => List.generate(1024, (j) => List.generate(1024, (k) => List.generate(3, (l) => 0.0))));
    for (int y = 0; y < 1024; y++) {
      for (int x = 0; x < 1024; x++) {
        final pixel = resizedImage.getPixel(x, y);
        // Normalise les valeurs RGB entre 0.0 et 1.0.
        inputMatrix[0][y][x][0] = pixel.r / 255.0; 
        inputMatrix[0][y][x][1] = pixel.g / 255.0; 
        inputMatrix[0][y][x][2] = pixel.b / 255.0; 
      }
    }
  } else {
    inputMatrix = List.generate(1, (i) => List.generate(3, (j) => List.generate(1024, (k) => List.generate(1024, (l) => 0.0))));
    for (int y = 0; y < 1024; y++) {
      for (int x = 0; x < 1024; x++) {
        final pixel = resizedImage.getPixel(x, y);
        // Normalise les valeurs RGB entre 0.0 et 1.0.
        inputMatrix[0][0][y][x] = pixel.r / 255.0; 
        inputMatrix[0][1][y][x] = pixel.g / 255.0; 
        inputMatrix[0][2][y][x] = pixel.b / 255.0; 
      }
    }
  }

  return {
    'width': w,
    'height': h,
    'matrix': inputMatrix
  };
}