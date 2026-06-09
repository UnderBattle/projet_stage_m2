import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'opencv/inpainting_core.dart';
import 'opencv/equipement_core.dart';
import 'opencv/fusion_core.dart';
import 'opencv/goulotte_core.dart';
import 'opencv/geometrie_utils.dart';

/// Fonctions "passerelles" conçues pour être exécutées dans des Isolates.
/// Elles permettent d'effectuer les traitements lourds (OpenCV, TFLite) sans bloquer l'interface utilisateur.
class TraitementImage {
  static Isolate? _workerIsolate;
  static SendPort? _workerSendPort;
  
  // CORRECTION 1 : Le port ne doit plus être "final", il sera recréé à chaque lancement
  static ReceivePort? _receivePort; 
  
  static final Map<int, Completer<dynamic>> _tachesEnAttente = {};
  static int _compteurTaches = 0;
  static bool _isInitialized = false;

  /// Initialise l'Isolate persistant. Doit être appelé au démarrage de l'écran.
  static Future<void> initWorker() async {
    if (_isInitialized) return;
    
    // CORRECTION 2 : On instancie un TOUT NOUVEAU port pour réinitialiser le Stream
    _receivePort = ReceivePort(); 
    
    // On écoute les réponses venant de l'Isolate
    _receivePort!.listen((message) {
      if (message is Map<String, dynamic>) {
        int id = message['id'];
        dynamic result = message['result'];
        if (_tachesEnAttente.containsKey(id)) {
          _tachesEnAttente[id]!.complete(result);
          _tachesEnAttente.remove(id);
        }
      }
    });
    
    // Initialisation du canal de communication bi-directionnel
    final initPort = ReceivePort();
    _workerIsolate = await Isolate.spawn(_workerEntry, initPort.sendPort);
    _workerSendPort = await initPort.first as SendPort;
    _isInitialized = true;
    print("[Optimisation] Worker Pool Isolate initialisé ! Prêt pour le temps réel.");
  }

  /// Nettoie l'Isolate quand on quitte l'écran pour libérer la RAM.
  static void disposeWorker() {
    if (!_isInitialized) return;
    _workerIsolate?.kill(priority: Isolate.immediate);
    _workerIsolate = null;
    
    // CORRECTION 3 : On ferme proprement le Stream
    _receivePort?.close(); 
    _receivePort = null;
    
    _isInitialized = false;
    
    // Sécurité : On libère les calculs qui étaient potentiellement en attente
    for (var completer in _tachesEnAttente.values) {
      if (!completer.isCompleted) completer.complete(null);
    }
    _tachesEnAttente.clear();
    print("[Optimisation] Worker Pool Isolate détruit.");
  }

  /// Fonction de routage interne vers le Worker.
  static Future<dynamic> _executerTache(String type, Map<String, dynamic> params) async {
    if (!_isInitialized) await initWorker();
    
    int id = _compteurTaches++;
    var completer = Completer<dynamic>();
    _tachesEnAttente[id] = completer;
    
    _workerSendPort!.send({
      'id': id,
      'type': type,
      'params': params,
      'replyPort': _receivePort!.sendPort, // CORRECTION 4 : Utilisation du nouveau port
    });
    
    return completer.future;
  }

  /// Le point d'entrée de l'Isolate qui tourne en boucle infinie en tâche de fond.
  static void _workerEntry(SendPort sendPort) {
    final port = ReceivePort();
    sendPort.send(port.sendPort);
    
    port.listen((message) async {
      if (message is Map<String, dynamic>) {
        int id = message['id'];
        String type = message['type'];
        Map<String, dynamic> params = message['params'];
        SendPort replyPort = message['replyPort'];
        
        dynamic result;
        try {
          switch (type) {
            case 'effacerAutocollant':
              result = await InpaintingCore.effacerAutocollantIsolate(params);
              break;
            case 'genererCalqueEquipement':
              result = await EquipementCore.genererCalqueEquipementIsolate(params);
              break;
            case 'fusionnerCalque':
              result = await FusionCore.fusionnerCalqueIsolate(params);
              break;
            case 'incrusterGoulotte':
              result = await GoulotteCore.incrusterGoulotteIsolate(params);
              break;
          }
        } catch (e) {
          print("[Worker Error] $e");
          result = null;
        }
        
        // Renvoie le résultat de la fonction lourde au thread principal UI
        replyPort.send({'id': id, 'result': result});
      }
    });
  }

  // =========================================================================
  // === NOUVELLES MÉTHODES RAPIDES VIA LE WORKER PERSISTANT ===
  // =========================================================================

  static Future<Uint8List?> effacerAutocollantWorker(Map<String, dynamic> params) async {
    return await _executerTache('effacerAutocollant', params) as Uint8List?;
  }

  static Future<Uint8List?> genererCalqueEquipementWorker(Map<String, dynamic> params) async {
    return await _executerTache('genererCalqueEquipement', params) as Uint8List?;
  }

  static Future<Uint8List?> fusionnerCalqueWorker(Map<String, dynamic> params) async {
    return await _executerTache('fusionnerCalque', params) as Uint8List?;
  }

  static Future<Uint8List?> incrusterGoulotteWorker(Map<String, dynamic> params) async {
    return await _executerTache('incrusterGoulotte', params) as Uint8List?;
  }

  // =========================================================================
  // === ANCIENNES MÉTHODES CONSERVÉES (Pour rétrocompatibilité) ===
  // =========================================================================
  
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