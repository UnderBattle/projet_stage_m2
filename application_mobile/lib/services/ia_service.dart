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

  Interpreter? yoloModel;
  Uint8List? lamaBytes;
  bool isInitialized = false;

  /// Charge les modèles TFLite depuis les assets.
  Future<void> initModels() async {
    if (isInitialized) return; // Si c'est déjà chargé, on ne fait rien
    
    try {
      print("[IAService] Début du chargement des modèles IA en arrière-plan...");
      final interpreterOptions = InterpreterOptions();
      
      if (Platform.isAndroid) {
        interpreterOptions.addDelegate(XNNPackDelegate()); 
      } else if (Platform.isIOS) {
        interpreterOptions.addDelegate(GpuDelegate());
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