import 'dart:typed_data';
import 'opencv/inpainting_core.dart';
import 'opencv/equipement_core.dart';
import 'opencv/fusion_core.dart';
import 'opencv/goulotte_core.dart';
import 'opencv/geometrie_utils.dart';

/// Fonctions "passerelles" conservant l'ancienne signature.
/// Elles appellent les nouveaux fichiers structurés dans lib/opencv/
class TraitementImage {
  
  static Future<Uint8List?> effacerAutocollantIsolate(Map<String, dynamic> params) async {
    return await InpaintingCore.effacerAutocollantIsolate(params);
  }

  static Future<Uint8List?> genererCalqueEquipementIsolate(Map<String, dynamic> params) async {
    return await EquipementCore.genererCalqueEquipementIsolate(params);
  }

  static Future<Uint8List?> fusionnerCalqueIsolate(Map<String, dynamic> params) async {
    return await FusionCore.fusionnerCalqueIsolate(params);
  }

  static Future<Uint8List?> incrusterGoulotteIsolate(Map<String, dynamic> params) async {
    return await GoulotteCore.incrusterGoulotteIsolate(params);
  }

  static List<Map<String, double>> trierPoints(List<Map<String, double>> points) {
    return GeometrieUtils.trierPoints(points);
  }
}