import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

/// Analyse et nettoie récursivement tout le cache.
Future<void> nettoyerCacheImages() async {
  try {
    final directory = await getTemporaryDirectory();
    
    // L'ajout de recursive: true est la clé ! On fouille dans tous les sous-dossiers.
    final files = directory.listSync(recursive: true); 
    
    int count = 0;
    double totalSizeMB = 0;
    
    print("=== DÉBUT DE L'ANALYSE DU CACHE ===");
    
    for (var file in files) {
      if (file is File) {
        // Calcul du poids du fichier
        int sizeBytes = file.lengthSync();
        double sizeMB = sizeBytes / (1024 * 1024);
        totalSizeMB += sizeMB;
        
        // On affiche dans la console tous les fichiers qui pèsent plus de 1 Mo 
        // pour que tu puisses VOIR le coupable.
        if (sizeMB > 1.0) {
          print("Gros fichier trouvé : ${file.path.split('/').last} (${sizeMB.toStringAsFixed(2)} Mo)");
        }
        
        final path = file.path.toLowerCase();
        
        // On supprime sans pitié toutes les images, peu importe où elles sont cachées
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
    final imageBytes = await originalFile.readAsBytes();
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
    
    print("[Optimisation] Taille correcte (${image.width}x${image.height}), pas de changement.");
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

  img.Image? originalImage = img.decodeImage(imageBytes);
  if (originalImage == null) return null;

  int w = originalImage.width;
  int h = originalImage.height;

  // Force la taille de l'image à celle requise par le modèle YOLO.
  img.Image resizedImage = img.copyResize(originalImage, width: 1024, height: 1024);

  List<dynamic> inputMatrix;
  
  // Convertit l'image selon le format attendu par le modèle : NHWC (Haut, Largeur, Canaux) ou NCHW (Canaux, Haut, Largeur).
  if (isNHWC) {
    inputMatrix = List.generate(1, (i) => List.generate(1024, (j) => List.generate(1024, (k) => Float32List(3))));
    int x = 0;
    int y = 0;
    // Itération linéaire ultra-rapide sur la mémoire de l'image (évite de créer un million d'objets)
    for (final pixel in resizedImage) {
      inputMatrix[0][y][x][0] = pixel.r / 255.0; 
      inputMatrix[0][y][x][1] = pixel.g / 255.0; 
      inputMatrix[0][y][x][2] = pixel.b / 255.0; 
      x++;
      if (x >= 1024) {
        x = 0;
        y++;
      }
    }
  } else {
    inputMatrix = List.generate(1, (i) => List.generate(3, (j) => List.generate(1024, (k) => Float32List(1024))));
    int x = 0;
    int y = 0;
    for (final pixel in resizedImage) {
      inputMatrix[0][0][y][x] = pixel.r / 255.0; 
      inputMatrix[0][1][y][x] = pixel.g / 255.0; 
      inputMatrix[0][2][y][x] = pixel.b / 255.0; 
      x++;
      if (x >= 1024) {
        x = 0;
        y++;
      }
    }
  }

  return {
    'width': w,
    'height': h,
    'matrix': inputMatrix
  };
}