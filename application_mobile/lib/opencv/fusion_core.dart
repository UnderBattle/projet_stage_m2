import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;

class FusionCore {
  /// Isolate d'assemblage ultra-rapide (Fusionne n'importe quel fond avec le PNG transparent)
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
}