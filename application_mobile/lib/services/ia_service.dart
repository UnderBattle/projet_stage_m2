import 'dart:io';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Service Singleton qui conserve les modèles IA dans le RAM
/// Permet de ne charger les modèles qu'une seule fois au lancement de l'application
class IAService {
  // Configuration du Singleton
  static final IAService _instance = IAService._internal();
  factory IAService() => _instance;
  IAService._internal();

  /// Modèle TFLite pour la détection de l'autocollant (YOLOv8).
  Interpreter? yoloModel;
  /// Modèle TFLite pour l'inpainting (LaMa), conservé en bytes pour être passé aux Isolates.
  Uint8List? lamaBytes;
  /// Indique si les modèles ont été chargés avec succès.
  bool isInitialized = false;

  /// Charge les modèles d'IA (YOLO et LaMa) depuis les assets de l'application.
  Future<void> initModels() async {
    // Si les modèles sont déjà chargés, on ne fait rien.
    if (isInitialized) return; 
    
    try {
      print("[IAService] Début du chargement des modèles IA en arrière-plan...");
      final interpreterOptions = InterpreterOptions();
      
      if (Platform.isAndroid) {
        try {
          // On tente de forcer l'accélération GPU (GpuDelegateV2 est spécifique à Android)
          interpreterOptions.addDelegate(GpuDelegateV2()); 
          print("[IAService] Accélération GPU Android (GpuDelegateV2) activée !");
        } catch (e) {
          // Fallback de sécurité : si le téléphone est trop vieux ou ne supporte pas l'API, 
          // on repasse sur le CPU optimisé (XNNPack).
          print("[IAService] Le GPU n'est pas supporté par ce téléphone, fallback sur CPU (XNNPack).");
          interpreterOptions.addDelegate(XNNPackDelegate()); 
        }
      } else if (Platform.isIOS) {
        // Sur iOS, l'API Metal est utilisée par défaut avec GpuDelegate
        interpreterOptions.addDelegate(GpuDelegate());
        print("[IAService] Accélération GPU iOS (Metal) activée !");
      }

      // Chargement de YOLO
      yoloModel = await Interpreter.fromAsset('assets/best.tflite', options: interpreterOptions);
      
      // Chargement de LaMa
      final ByteData lamaData = await rootBundle.load('assets/lama_dynamic_45mo.tflite');
      lamaBytes = lamaData.buffer.asUint8List();
      
      isInitialized = true;
      print("[IAService] Modèles IA chargés avec succès dans la mémoire vive");
    } catch (e) {
      print("[IAService] Erreur lors du chargement des modèles : $e");
    }
  }
}