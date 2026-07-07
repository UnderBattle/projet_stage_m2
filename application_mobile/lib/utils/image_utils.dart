import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path_provider/path_provider.dart';

/// Analyse et nettoie récursivement tout le cache.
Future<void> nettoyerCacheImages() async {
  try {
    final directory = await getTemporaryDirectory();

    int count = 0;
    double totalSizeMB = 0;

    print("=== DÉBUT DE L'ANALYSE DU CACHE ===");

    // API asynchrone pour ne pas bloquer le thread principal.
    final Stream<FileSystemEntity> files = directory.list(recursive: true);

    await for (var file in files) {
      if (file is File) {
        // Calcul asynchrone du poids du fichier.
        int sizeBytes = await file.length();
        double sizeMB = sizeBytes / (1024 * 1024);
        totalSizeMB += sizeMB;

        // On affiche dans la console tous les fichiers qui pèsent plus de 1 Mo.
        if (sizeMB > 1.0) {
          print("Gros fichier trouvé : ${file.path.split('/').last} (${sizeMB.toStringAsFixed(2)} Mo)");
        }

        final path = file.path.toLowerCase();

        // On supprime toutes les images du cache.
        if (path.contains('photo_optimisee_') ||
            path.contains('image_picker_') ||
            path.endsWith('.jpg') ||
            path.endsWith('.jpeg') ||
            path.endsWith('.png')) {
          await file.delete();
          count++;
        }
      }
    }

    print("=== RÉSULTAT DU NETTOYAGE ===");
    print("[Optimisation] Poids total scanné : ${totalSizeMB.toStringAsFixed(2)} Mo");
    print("[Optimisation] Fichiers supprimés : $count");
    
  } catch (e) {
    print("[Optimisation] Erreur lors du nettoyage du cache : $e");
  }
}

/// Redimensionne une image si elle dépasse une taille maximale pour éviter de surcharger la mémoire.
/// Conçu pour être exécuté dans un Isolate.
Future<String?> redimensionnerImageLourde(String imagePath) async {
  try {
    final originalFile = File(imagePath);

    cv.Mat image = cv.imread(imagePath, flags: cv.IMREAD_COLOR);
    
    // NOUVEAU : Guard Clause Anti-Crash (Empêche un SIGSEGV C++ si l'image de la galerie est corrompue)
    if (image.isEmpty || image.cols <= 0 || image.rows <= 0) {
      throw Exception("Fichier image corrompu ou illisible par OpenCV.");
    }

    int maxSize = 1920; 
    // Vérifie si l'image dépasse la limite autorisée.
    if (image.cols > maxSize || image.rows > maxSize) {
      print("[Optimisation] L'image est trop grande (${image.cols}x${image.rows}). Redimensionnement via OpenCV...");
      
      // Calcule le nouveau ratio en gardant les proportions de l'image.
      double ratio = maxSize / math.max(image.cols, image.rows);
      int newWidth = (image.cols * ratio).toInt();
      int newHeight = (image.rows * ratio).toInt();

      cv.Mat resized = cv.resize(image, (newWidth, newHeight), interpolation: cv.INTER_AREA);

      // On utilise le dossier parent de l'image source (qui est dans le cache)
      // pour éviter d'appeler un plugin (getTemporaryDirectory) depuis l'isolate.
      final dirPath = originalFile.parent.path;
      final path = '$dirPath/photo_optimisee_${DateTime.now().millisecondsSinceEpoch}.jpg';
      
      // Sauvegarde ultra-rapide avec OpenCV
      cv.imwrite(path, resized, params: cv.VecI32.fromList([cv.IMWRITE_JPEG_QUALITY, 90]));
      print("[Optimisation] Nouvelle taille ${resized.cols}x${resized.rows} prête !");

      // Maintenant que la version légère existe, l'énorme fichier original (ex: 1000057483.jpg de 7 Mo) ne sert plus à rien. On le supprime
      try {
        if (await originalFile.exists()) {
          await originalFile.delete();
          print("[Optimisation] Poubelle : Ancien fichier lourd supprimé avec succès.");
        }
      } catch (e) {
        print("[Optimisation] Impossible de supprimer la source : $e");
      }

      return path;
    }
    
    print("[Optimisation] Taille correcte (${image.cols}x${image.rows}), pas de changement.");
    // Si l'image n'a pas été redimensionnée, on NE la supprime PAS car on va l'utiliser !
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

  // OPTIMISATION : Décodage ultra-rapide en C++ via OpenCV
  cv.Mat originalImage = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
  
  // NOUVEAU : Guard Clause Anti-Crash pour l'analyse IA
  if (originalImage.isEmpty || originalImage.cols <= 0 || originalImage.rows <= 0) {
    throw Exception("Image source corrompue lors de la préparation IA.");
  }

  int w = originalImage.cols;
  int h = originalImage.rows;

  // Force la taille de l'image à celle requise par le modèle YOLO.
  // L'interpolation linéaire C++ est environ 40x plus rapide que package:image
  cv.Mat resizedImageBgr = cv.resize(originalImage, (1024, 1024), interpolation: cv.INTER_LINEAR);
  
  // YOLO s'attend au format RGB
  cv.Mat resizedImage = cv.cvtColor(resizedImageBgr, cv.COLOR_BGR2RGB);

  List<dynamic> inputMatrix;
  
  // Convertit l'image selon le format attendu par le modèle : NHWC (Haut, Largeur, Canaux) ou NCHW (Canaux, Haut, Largeur).
  if (isNHWC) {
    inputMatrix = List.generate(1, (i) => List.generate(1024, (j) => List.generate(1024, (k) => Float32List(3))));
    
    // Extraction directe des bytes mémoires au lieu de boucler sur un objet Image lent
    Uint8List pixels = resizedImage.data;
    int idx = 0;
    
    // Itération linéaire ultra-rapide sur la mémoire de l'image (évite de créer un million d'objets)
    for (int y = 0; y < 1024; y++) {
      for (int x = 0; x < 1024; x++) {
        inputMatrix[0][y][x][0] = pixels[idx] / 255.0; 
        inputMatrix[0][y][x][1] = pixels[idx+1] / 255.0; 
        inputMatrix[0][y][x][2] = pixels[idx+2] / 255.0; 
        idx += 3;
      }
    }
  } else {
    inputMatrix = List.generate(1, (i) => List.generate(3, (j) => List.generate(1024, (k) => Float32List(1024))));
    
    Uint8List pixels = resizedImage.data;
    int idx = 0;
    
    for (int y = 0; y < 1024; y++) {
      for (int x = 0; x < 1024; x++) {
        inputMatrix[0][0][y][x] = pixels[idx] / 255.0; 
        inputMatrix[0][1][y][x] = pixels[idx+1] / 255.0; 
        inputMatrix[0][2][y][x] = pixels[idx+2] / 255.0; 
        idx += 3;
      }
    }
  }

  return {
    'width': w,
    'height': h,
    'matrix': inputMatrix
  };
}