import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;

class FusionCore {
  /// Isolate d'assemblage ultra-rapide (Fusionne n'importe quel fond avec le calque transparent)
  static Future<Uint8List?> fusionnerCalqueIsolate(Map<String, dynamic> params) async {
    try {
      Uint8List fondBytes = params['fondBytes'];
      Uint8List calquePngBytes = params['calquePngBytes']; 
      
      cv.Mat bg = cv.imdecode(fondBytes, cv.IMREAD_COLOR);
      
      // NOUVEAU : Guard Clause Anti-Crash
      if (bg.isEmpty || bg.cols <= 0 || bg.rows <= 0) {
        throw Exception("Image de fond corrompue pour la fusion.");
      }

      // Le calque PNG ayant été encodé avec une compression de 0, 
      // le décodage C++ est immédiat et sans erreur de mapping mémoire.
      cv.Mat overlay = cv.imdecode(calquePngBytes, cv.IMREAD_UNCHANGED);
      
      // NOUVEAU : Guard Clause Anti-Crash
      if (overlay.isEmpty || overlay.cols <= 0 || overlay.rows <= 0) {
        throw Exception("Calque PNG corrompu pour la fusion.");
      }
      
      // Extraction sécurisée du BGR et du canal Alpha
      cv.Mat overlayBgr = cv.cvtColor(overlay, cv.COLOR_BGRA2BGR);

      var channels = cv.split(overlay);

      cv.Mat alpha = channels[3];
      cv.Mat alpha3c8u = cv.cvtColor(alpha, cv.COLOR_GRAY2BGR);
      
      // Inversion de l'Alpha (pour creuser le trou dans l'image de fond)
      cv.Mat invAlpha3c8u = cv.bitwiseNOT(alpha3c8u);
      cv.Mat bgBlended = cv.multiply(bg, invAlpha3c8u, scale: 1.0 / 255.0);
      cv.Mat fgBlended = cv.multiply(overlayBgr, alpha3c8u, scale: 1.0 / 255.0);
      cv.Mat result8u = cv.add(bgBlended, fgBlended);
      
      var encodeResult = cv.imencode('.jpg', result8u);
      return encodeResult.$2;
    } catch (e) {
      print("[OpenCV] Erreur lors de la fusion du calque : $e");
      return null;
    }
  }
}