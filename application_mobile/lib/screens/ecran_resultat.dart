import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image_gallery_saver_plus/image_gallery_saver_plus.dart';

import '../traitement_image.dart';
import '../services/ia_service.dart';
import '../services/catalogue_service.dart';
import '../models/devis_models.dart';
import '../utils/image_utils.dart';
import '../widgets/catalogue_devis.dart';
import '../widgets/boutons_action_devis.dart';
import '../widgets/resultat/overlay_chargement.dart';
import '../widgets/resultat/carte_confirmation_ia.dart';
import '../widgets/resultat/calque_placement_manuel.dart';
import '../widgets/resultat/calque_simulation.dart';

/// Écran affichant l'image capturée, exécutant la détection de l'IA, 
/// et permettant à l'utilisateur d'incruster et de manipuler des modèles d'équipement.
class EcranResultat extends StatefulWidget {
  final String photoPath;

  const EcranResultat({super.key, required this.photoPath});

  @override
  State<EcranResultat> createState() => _EcranResultatState();
}

class _EcranResultatState extends State<EcranResultat> {
  // =========================================================================
  // === VARIABLES D'ÉTAT ===
  // =========================================================================
  
  late String _categorieSelectionnee;
  Equipement? _modeleSelectionne;
  
  bool _isProcessing = true;
  String _loadingMessage = "Analyse en cours...";
  
  Uint8List? _imageResultatBytes; // L'image finale (avec goulotte + equipement)
  
  // =========================================================================
  // === SYSTÈME DE CACHE INDÉPENDANT ===
  // =========================================================================
  Uint8List? _imageFondPropreBytes;
  Uint8List? _imageFondAvecGoulotteBytes; 
  Uint8List? _calqueEquipementPngBytes; 
  
  Uint8List? _dernierCalqueEquipementCompletBytes;
  Offset _decalageDuCalqueComplet = Offset.zero;

  int? _imageWidth;
  int? _imageHeight;
  List<Map<String, double>>? _pointsCibles;
  bool _isManualPlacementMode = false;
  
  bool _attenteConfirmationIA = false;

  final TransformationController _transformationController = TransformationController();

  // =========================================================================
  // GESTION D'ÉTAT OPTIMISÉE (VALUENOTIFIERS)
  // =========================================================================
  final ValueNotifier<Offset> _decalageNotifier = ValueNotifier(Offset.zero);
  final ValueNotifier<double> _splitNotifier = ValueNotifier(1.0);
  final ValueNotifier<bool> _isDraggingEquipementNotifier = ValueNotifier(false);

  final List<Offset> _historiqueDecalages = [Offset.zero];
  final List<Offset> _historiqueRedoDecalages = [];
  
  final ValueNotifier<int> _historiqueLengthNotifier = ValueNotifier(1);
  final ValueNotifier<int> _historiqueRedoLengthNotifier = ValueNotifier(0);

  bool _isDrawGoulotteMode = false;
  final ValueNotifier<LigneGoulotte?> _goulotteNotifier = ValueNotifier(null);
  LigneGoulotte? _goulotteInitiale;
  LigneGoulotte? _goulotteRedo;
  
  final ValueNotifier<bool> _isDraggingGoulotteNotifier = ValueNotifier(false);
  final ValueNotifier<Offset?> _goulotteCurrentEndOrigNotifier = ValueNotifier(null);
  final ValueNotifier<Offset?> _magnifierPositionNotifier = ValueNotifier(null);

  int _activePointers = 0;

  @override
  void initState() {
    super.initState();
    TraitementImage.initWorker();
    _categorieSelectionnee = CatalogueService().catalogueGlobal.keys.first;
    _analyserImage();
  }

  @override
  void dispose() {
    _transformationController.dispose();
    _decalageNotifier.dispose();
    _splitNotifier.dispose();
    _isDraggingEquipementNotifier.dispose();
    _historiqueLengthNotifier.dispose();
    _historiqueRedoLengthNotifier.dispose();
    _goulotteNotifier.dispose();
    _isDraggingGoulotteNotifier.dispose();
    _goulotteCurrentEndOrigNotifier.dispose();
    _magnifierPositionNotifier.dispose(); 
    
    TraitementImage.disposeWorker();
    super.dispose();
  }

  void _montrerErreur(String message) {
    if (!mounted) return;
    final theme = Theme.of(context);
    
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message, style: TextStyle(color: theme.colorScheme.onError)),
        backgroundColor: theme.colorScheme.error,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 4),
        action: SnackBarAction(label: 'OK', textColor: theme.colorScheme.onError, onPressed: () {}),
      ),
    );
  }

  // =========================================================================
  // === LOGIQUE MÉTIER ET IA ===
  // =========================================================================
  Future<void> _analyserImage() async {
    final yoloModel = IAService().yoloModel;
    if (yoloModel == null) {
      _montrerErreur("Le modèle IA (YOLO) n'a pas pu être chargé.");
      return;
    }

    try {
      setState(() => _loadingMessage = "Détection de l'autocollant...");

      final imageBytes = await File(widget.photoPath).readAsBytes();
      var inputShape = yoloModel.getInputTensor(0).shape;
      bool isNHWC = inputShape[3] == 3;
      
      final resultMatrixPrep = await compute(prepareImageMatrixForIA, {
        'bytes': imageBytes,
        'isNHWC': isNHWC
      });

      if (resultMatrixPrep == null) throw Exception("Impossible de lire l'image.");

      _imageWidth = resultMatrixPrep['width'];
      _imageHeight = resultMatrixPrep['height'];
      var inputMatrix = resultMatrixPrep['matrix'];

      var outputShape = yoloModel.getOutputTensor(0).shape;
      var outputMatrix = List.generate(outputShape[0], (i) => List.generate(outputShape[1], (j) => List<double>.filled(outputShape[2], 0.0)));

      yoloModel.run(inputMatrix, outputMatrix);

      double maxConfiance = 0;
      int meilleurIndex = 0;
      bool isTransposed = outputShape[1] == 21504;
      int nbColonnes = isTransposed ? 21504 : outputShape[2];

      for (int i = 0; i < nbColonnes; i++) {
        double confiance = isTransposed ? outputMatrix[0][i][4] : outputMatrix[0][4][i];
        if (confiance > maxConfiance) {
          maxConfiance = confiance;
          meilleurIndex = i;
        }
      }

      if (maxConfiance > 0.5) {
        double boxX = isTransposed ? outputMatrix[0][meilleurIndex][0] : outputMatrix[0][0][meilleurIndex];
        double boxY = isTransposed ? outputMatrix[0][meilleurIndex][1] : outputMatrix[0][1][meilleurIndex];
        double boxW = isTransposed ? outputMatrix[0][meilleurIndex][2] : outputMatrix[0][2][meilleurIndex];
        double boxH = isTransposed ? outputMatrix[0][meilleurIndex][3] : outputMatrix[0][3][meilleurIndex];
        
        double scale = (boxW < 2.0 && boxH < 2.0) ? 1024.0 : 1.0;
        List<Map<String, double>> rawPoints = [];
        double confMoyennePoints = 0;

        for(int point = 0; point < 4; point++) {
           int idxX = 5 + (point * 3);
           int idxY = idxX + 1;
           int idxConf = idxX + 2;
           
           double px = isTransposed ? outputMatrix[0][meilleurIndex][idxX] : outputMatrix[0][idxX][meilleurIndex];
           double py = isTransposed ? outputMatrix[0][meilleurIndex][idxY] : outputMatrix[0][idxY][meilleurIndex];
           double pConf = isTransposed ? outputMatrix[0][meilleurIndex][idxConf] : outputMatrix[0][idxConf][meilleurIndex];
           
           confMoyennePoints += pConf;
           rawPoints.add({'x': px * scale, 'y': py * scale});
        }
        confMoyennePoints = confMoyennePoints / 4.0;

        print("[IA] Confiance moyenne des points : $confMoyennePoints");

        if (confMoyennePoints >= 0.9775) {
          _pointsCibles = TraitementImage.trierPoints(rawPoints);
        } else {
          double xMin = (boxX * scale) - (boxW * scale) / 2;
          double yMin = (boxY * scale) - (boxH * scale) / 2;
          double xMax = (boxX * scale) + (boxW * scale) / 2;
          double yMax = (boxY * scale) + (boxH * scale) / 2;
          
          _pointsCibles = [
            {'x': xMin, 'y': yMin},
            {'x': xMax, 'y': yMin},
            {'x': xMax, 'y': yMax},
            {'x': xMin, 'y': yMax} 
          ];
        }

        if (_pointsCibles != null) {
          HapticFeedback.heavyImpact(); 
          
          setState(() {
            _isProcessing = false;
            _isManualPlacementMode = true; 
            _attenteConfirmationIA = true; 
          });
        }
      } else {
        setState(() {
          _pointsCibles = null;
          _isProcessing = false;
        });
        
        WidgetsBinding.instance.addPostFrameCallback((_) {
          _demanderPlacementManuel();
        });
      }
    } catch (e) {
      print("[IA - ERREUR] Exception : $e");
      if (mounted) {
        setState(() {
          _pointsCibles = null;
          _isProcessing = false;
        });
        _montrerErreur("Une erreur est survenue lors de l'analyse de l'image par l'IA.");
      }
    }
  }

  void _demanderPlacementManuel() {
    final theme = Theme.of(context);
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        backgroundColor: theme.cardColor,
        title: Text("Autocollant introuvable", style: TextStyle(color: theme.colorScheme.onSurface)),
        content: Text("L'IA n'a pas pu détecter l'autocollant avec certitude.\nVoulez-vous placer la zone manuellement ?", style: TextStyle(color: theme.colorScheme.onSurface)),
        actions: [
          ElevatedButton.icon(
            icon: const Icon(Icons.touch_app),
            label: const Text("Placer manuellement"),
            style: ElevatedButton.styleFrom(backgroundColor: theme.colorScheme.primary, foregroundColor: theme.colorScheme.onPrimary),
            onPressed: () {
              Navigator.pop(context);
              _activerModeManuel();
            },
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context); 
              Navigator.pop(context); 
            },
            child: Text("Annuler", style: TextStyle(color: theme.colorScheme.onSurface.withValues(alpha: 0.6))),
          ),
        ],
      ),
    );
  }

  void _activerModeManuel() {
    setState(() {
      _isManualPlacementMode = true;
      _attenteConfirmationIA = false;
      _pointsCibles = [
        {'x': 512.0 - 75.0, 'y': 512.0 - 150.0},
        {'x': 512.0 + 75.0, 'y': 512.0 - 150.0},
        {'x': 512.0 + 75.0, 'y': 512.0 + 150.0},
        {'x': 512.0 - 75.0, 'y': 512.0 + 150.0},
      ];
    });
  }

  Future<void> _validerPlacementManuel() async {
    _transformationController.value = Matrix4.identity();
    setState(() {
      _isManualPlacementMode = false;
      _isProcessing = true;
      _loadingMessage = "Nettoyage de la zone sélectionnée..."; 
      _pointsCibles = TraitementImage.trierPoints(_pointsCibles!); 
    });

    try {
      _imageFondPropreBytes = await TraitementImage.effacerAutocollantWorker({
        'photoPath': widget.photoPath,
        'pointsIA': _pointsCibles!,
        'lamaBytes': IAService().lamaBytes,
      });
      _imageFondAvecGoulotteBytes = null; 
      _calqueEquipementPngBytes = null;
      _dernierCalqueEquipementCompletBytes = null; 
      HapticFeedback.mediumImpact();

    } catch (e) {
      print("Erreur inpainting manuel : $e");
      if (mounted) _montrerErreur("Impossible de nettoyer le mur (Erreur OpenCV).");
    }

    setState(() {
      _isProcessing = false;
    });
  }

  Future<void> _genererIncrustation({bool recomputeGoulotte = false, bool recomputeEquipement = false}) async {
    if (_pointsCibles == null || _modeleSelectionne == null || _imageFondPropreBytes == null) return;
    
    setState(() {
      _isProcessing = true;
      _loadingMessage = "Préparation du rendu...";
    });

    try {
      String equipementPath = _modeleSelectionne!.chemin;
      final ByteData data = await DefaultAssetBundle.of(context).load(equipementPath);
      Uint8List equipementBytes = data.buffer.asUint8List();
      
      double profondeur = _modeleSelectionne!.profondeur;
      double hauteur = _modeleSelectionne!.hauteur;
      double largeur = _modeleSelectionne!.largeur;

      double ptHgXOrig = _pointsCibles![0]['x']! * (_imageWidth! / 1024.0);
      double ptHgYOrig = _pointsCibles![0]['y']! * (_imageHeight! / 1024.0);
      double ptHdXOrig = _pointsCibles![1]['x']! * (_imageWidth! / 1024.0);
      double ptHdYOrig = _pointsCibles![1]['y']! * (_imageHeight! / 1024.0);
      double dx = ptHdXOrig - ptHgXOrig;
      double dy = ptHdYOrig - ptHgYOrig;
      
      double autoWPxOrig = math.sqrt(dx * dx + dy * dy);
      double ratioPxParMm = autoWPxOrig / 50.0;
      double largeurGoulotteOrig = 80.0 * ratioPxParMm; 

      if (recomputeGoulotte || (_goulotteNotifier.value != null && _imageFondAvecGoulotteBytes == null)) {
        if (_goulotteNotifier.value != null) {
          setState(() => _loadingMessage = "Incrustation de la goulotte...");
          
          _imageFondAvecGoulotteBytes = await TraitementImage.incrusterGoulotteWorker({
            'imageDeFondBytes': _imageFondPropreBytes!, 
            'ptDepartX': _goulotteNotifier.value!.start.dx,
            'ptDepartY': _goulotteNotifier.value!.start.dy,
            'ptArriveeX': _goulotteNotifier.value!.end.dx,
            'ptArriveeY': _goulotteNotifier.value!.end.dy,
            'largeurPx': largeurGoulotteOrig, 
          });
        } else {
          _imageFondAvecGoulotteBytes = null;
        }
      }

      if (recomputeEquipement || _calqueEquipementPngBytes == null) {
        setState(() => _loadingMessage = "Calcul des ombres de l'équipement...");
        
        _calqueEquipementPngBytes = await TraitementImage.genererCalqueEquipementWorker({
          'fondPropreBytes': _imageFondPropreBytes!,
          'equipementBytes': equipementBytes,
          'pointsIA': _pointsCibles!,
          'decalageX': _decalageNotifier.value.dx,
          'decalageY': _decalageNotifier.value.dy,
          'equipementAssetPath': equipementPath,
          'profondeurMm': profondeur,
          'hauteurMm': hauteur,
          'largeurMm': largeur,
        });

        double equipementWPxOrig = (largeur / 50.0) * autoWPxOrig;
        double equipementHPxOrig = equipementWPxOrig * (hauteur / largeur);
        double eqX = ptHgXOrig + _decalageNotifier.value.dx;
        double eqY = ptHgYOrig + _decalageNotifier.value.dy;
        
        bool isCropped = (eqX < 50 || eqY < 50 || (eqX + equipementWPxOrig + 100) > _imageWidth! || (eqY + equipementHPxOrig + 100) > _imageHeight!);

        if (!isCropped || _dernierCalqueEquipementCompletBytes == null) {
           _dernierCalqueEquipementCompletBytes = _calqueEquipementPngBytes;
           _decalageDuCalqueComplet = _decalageNotifier.value;
        }
      }

      setState(() => _loadingMessage = "Assemblage final...");
      Uint8List fondBase = _imageFondAvecGoulotteBytes ?? _imageFondPropreBytes!;

      Uint8List? resultImage = await TraitementImage.fusionnerCalqueWorker({
        'fondBytes': fondBase,
        'calquePngBytes': _calqueEquipementPngBytes!
      });

      if (resultImage != null) {
        setState(() {
          _imageResultatBytes = resultImage;
          _splitNotifier.value = 1.0;
        });
      } else {
        throw Exception("L'image fusionnée est nulle.");
      }
    } catch (e) {
      print("[UI/OpenCV - ERREUR] Exception : $e");
      if (mounted) _montrerErreur("Erreur lors de la génération 3D de l'équipement.");
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  void _reinitialiserPosition() {
    if (_isProcessing) return;

    if (_isDrawGoulotteMode) {
      if (_goulotteNotifier.value != null && _goulotteInitiale != null) {
        _goulotteRedo = _goulotteNotifier.value;
        _historiqueRedoLengthNotifier.value = 1;
        
        _goulotteNotifier.value = _goulotteInitiale;
        _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
      }
    } else {
      if (_historiqueDecalages.length > 1) {
        Offset currentPos = _historiqueDecalages.removeLast(); 
        _historiqueRedoDecalages.add(currentPos);
        
        _decalageNotifier.value = _historiqueDecalages.last; 
        _historiqueLengthNotifier.value = _historiqueDecalages.length; 
        _historiqueRedoLengthNotifier.value = _historiqueRedoDecalages.length;
        
        _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true);
      }
    }
  }

  void _retablirPosition() {
    if (_isProcessing) return;

    if (_isDrawGoulotteMode) {
      if (_goulotteRedo != null) {
        _goulotteNotifier.value = _goulotteRedo;
        _goulotteRedo = null;
        _historiqueRedoLengthNotifier.value = 0;
        _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
      }
    } else {
      if (_historiqueRedoDecalages.isNotEmpty) {
        Offset redonePos = _historiqueRedoDecalages.removeLast();
        _decalageNotifier.value = redonePos;
        _historiqueDecalages.add(redonePos);
        
        _historiqueLengthNotifier.value = _historiqueDecalages.length;
        _historiqueRedoLengthNotifier.value = _historiqueRedoDecalages.length;
        
        _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true);
      }
    }
  }

  void _resetPositionEquipement() {
    if (_isProcessing) return;

    setState(() {
      _decalageNotifier.value = Offset.zero;
      _historiqueDecalages.clear();
      _historiqueDecalages.add(Offset.zero);
      _historiqueLengthNotifier.value = 1; 
      
      _historiqueRedoDecalages.clear();
      _historiqueRedoLengthNotifier.value = 0;
    });

    _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true);
  }

  Future<void> _sauvegarderImage() async {
    if (_imageResultatBytes == null) return;

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Sauvegarde en cours...'), duration: Duration(milliseconds: 500)),
    );

    try {
      final result = await ImageGallerySaverPlus.saveImage(
        _imageResultatBytes!,
        quality: 100,
        name: "Devis_Simulation_${DateTime.now().millisecondsSinceEpoch}", 
      );
      
      if (!mounted) return;
      
      if (result['isSuccess'] == true) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Simulation sauvegardée dans la galerie'), backgroundColor: Colors.green, duration: Duration(seconds: 3)),
        );
      } else {
        throw Exception("Échec de la sauvegarde.");
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Erreur lors de la sauvegarde.'), backgroundColor: Colors.red));
    }
  }

  void _alignerZoneSelectionSurRectangle() {
    double minX = _pointsCibles!.map((p) => p['x']!).reduce(math.min);
    double maxX = _pointsCibles!.map((p) => p['x']!).reduce(math.max);
    double minY = _pointsCibles!.map((p) => p['y']!).reduce(math.min);
    double maxY = _pointsCibles!.map((p) => p['y']!).reduce(math.max);
    
    _pointsCibles = [
      {'x': minX, 'y': minY}, 
      {'x': maxX, 'y': minY}, 
      {'x': maxX, 'y': maxY}, 
      {'x': minX, 'y': maxY}, 
    ];
  }

  void _deplacerZoneSelectionComplete(double dx1024, double dy1024) {
    bool canMove = true;
    for (var p in _pointsCibles!) {
      double newX = p['x']! + dx1024;
      double newY = p['y']! + dy1024;
      if (newX < 0 || newX > 1024 || newY < 0 || newY > 1024) {
        canMove = false;
        break;
      }
    }
    
    if (canMove) {
      for (int i = 0; i < 4; i++) {
        _pointsCibles![i]['x'] = _pointsCibles![i]['x']! + dx1024;
        _pointsCibles![i]['y'] = _pointsCibles![i]['y']! + dy1024;
      }
    }
  }

  void _redimensionnerRectangleManuel(int idx, double newX, double newY) {
    if (idx == 0) {
      if (newX >= _pointsCibles![1]['x']! - 10) newX = _pointsCibles![1]['x']! - 10;
      if (newY >= _pointsCibles![3]['y']! - 10) newY = _pointsCibles![3]['y']! - 10;
      _pointsCibles![0]['x'] = newX;
      _pointsCibles![0]['y'] = newY;
      _pointsCibles![1]['y'] = newY; 
      _pointsCibles![3]['x'] = newX; 
    } else if (idx == 1) { 
      if (newX <= _pointsCibles![0]['x']! + 10) newX = _pointsCibles![0]['x']! + 10;
      if (newY >= _pointsCibles![2]['y']! - 10) newY = _pointsCibles![2]['y']! - 10;
      _pointsCibles![1]['x'] = newX;
      _pointsCibles![1]['y'] = newY;
      _pointsCibles![0]['y'] = newY; 
      _pointsCibles![2]['x'] = newX; 
    } else if (idx == 2) { 
      if (newX <= _pointsCibles![3]['x']! + 10) newX = _pointsCibles![3]['x']! + 10;
      if (newY <= _pointsCibles![1]['y']! + 10) newY = _pointsCibles![1]['y']! + 10;
      _pointsCibles![2]['x'] = newX;
      _pointsCibles![2]['y'] = newY;
      _pointsCibles![3]['y'] = newY; 
      _pointsCibles![1]['x'] = newX; 
    } else if (idx == 3) { 
      if (newX >= _pointsCibles![2]['x']! - 10) newX = _pointsCibles![2]['x']! - 10;
      if (newY <= _pointsCibles![0]['y']! + 10) newY = _pointsCibles![0]['y']! + 10;
      _pointsCibles![3]['x'] = newX;
      _pointsCibles![3]['y'] = newY;
      _pointsCibles![2]['y'] = newY; 
      _pointsCibles![0]['x'] = newX; 
    }
  }

  // =========================================================================
  // === GESTION DU PANNEAU D'ACTION EN BAS DE L'ÉCRAN ===
  // =========================================================================
  Widget? _buildPanneauActionBasse(ThemeData theme) {
    if (_attenteConfirmationIA) {
      return CarteConfirmationIA(
        theme: theme,
        onAjuster: () {
          setState(() {
            _attenteConfirmationIA = false;
            _alignerZoneSelectionSurRectangle();
          });
        },
        onValider: () {
          setState(() {
            _attenteConfirmationIA = false;
          });
          _validerPlacementManuel();
        },
      );
    } else if (_isManualPlacementMode) {
      return FloatingActionButton.extended(
        onPressed: _validerPlacementManuel,
        label: const Text("Valider la position", style: TextStyle(fontWeight: FontWeight.bold)),
        icon: const Icon(Icons.check),
        backgroundColor: theme.colorScheme.secondary,
        foregroundColor: Colors.white,
      );
    } else if (_imageResultatBytes != null && !_isProcessing) {
      return FloatingActionButton.extended(
        onPressed: _sauvegarderImage,
        label: const Text("Sauvegarder", style: TextStyle(fontWeight: FontWeight.bold)),
        icon: const Icon(Icons.download),
        backgroundColor: theme.colorScheme.secondary,
        foregroundColor: Colors.white,
      );
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Configuration du Devis'),
      ),
      body: Listener(
        onPointerDown: (_) {
          _activePointers++;
          if (_activePointers > 1) {
            _isDraggingEquipementNotifier.value = false;
            _isDraggingGoulotteNotifier.value = false;
            _goulotteCurrentEndOrigNotifier.value = null;
            _magnifierPositionNotifier.value = null; 
          }
        },
        onPointerUp: (_) {
          _activePointers = math.max(0, _activePointers - 1);
        },
        onPointerCancel: (_) {
          _activePointers = math.max(0, _activePointers - 1);
        },
        child: Column(
          children: [
            Expanded(
              child: Container(
                width: double.infinity,
                margin: const EdgeInsets.only(left: 16.0, right: 16.0, top: 16.0, bottom: 8.0),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [BoxShadow(color: theme.shadowColor.withValues(alpha: 0.1), blurRadius: 15, spreadRadius: 1)], 
                ),
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(20),
                        child: ValueListenableBuilder<LigneGoulotte?>(
                          valueListenable: _goulotteNotifier,
                          builder: (context, goulotte, _) {
                            bool isPanZoomEnabled = !_isDrawGoulotteMode || goulotte != null;
                            
                            return InteractiveViewer(
                              transformationController: _transformationController, 
                              panEnabled: isPanZoomEnabled,
                              scaleEnabled: isPanZoomEnabled,
                              minScale: 1.0,
                              maxScale: 9.0, 
                              child: LayoutBuilder(
                                builder: (context, constraints) {
                                  if (_imageWidth == null || _imageHeight == null || (_pointsCibles == null && !_isManualPlacementMode)) {
                                     return Hero(tag: 'image_mur', child: Image.file(File(widget.photoPath), fit: BoxFit.contain));
                                  }

                                  double scale = math.min(constraints.maxWidth / _imageWidth!, constraints.maxHeight / _imageHeight!);
                                  double offsetX = (constraints.maxWidth - (_imageWidth! * scale)) / 2;
                                  double offsetY = (constraints.maxHeight - (_imageHeight! * scale)) / 2;

                                  if (_isManualPlacementMode) {
                                    // APPEL AU NOUVEAU WIDGET D'ÉDITION MANUELLE
                                    return CalquePlacementManuel(
                                      photoPath: widget.photoPath,
                                      pointsCibles: _pointsCibles!,
                                      imageWidth: _imageWidth!,
                                      imageHeight: _imageHeight!,
                                      scale: scale,
                                      offsetX: offsetX,
                                      offsetY: offsetY,
                                      transformationController: _transformationController,
                                      attenteConfirmationIA: _attenteConfirmationIA,
                                      activePointers: _activePointers,
                                      theme: theme,
                                      onAlignerZone: () {
                                        setState(() {
                                          _attenteConfirmationIA = false;
                                          _alignerZoneSelectionSurRectangle();
                                        });
                                      },
                                      onDeplacerZone: (dx, dy) {
                                        setState(() => _deplacerZoneSelectionComplete(dx, dy));
                                      },
                                      onRedimensionnerZone: (idx, nx, ny) {
                                        setState(() => _redimensionnerRectangleManuel(idx, nx, ny));
                                      }
                                    );
                                  } else {
                                    // APPEL AU NOUVEAU WIDGET DE RENDU 3D
                                    return CalqueSimulation(
                                      photoPath: widget.photoPath,
                                      scale: scale,
                                      offsetX: offsetX,
                                      offsetY: offsetY,
                                      constraints: constraints,
                                      theme: theme,
                                      imageWidth: _imageWidth!,
                                      imageHeight: _imageHeight!,
                                      pointsCibles: _pointsCibles!,
                                      modeleSelectionne: _modeleSelectionne,
                                      imageFondPropreBytes: _imageFondPropreBytes,
                                      imageFondAvecGoulotteBytes: _imageFondAvecGoulotteBytes,
                                      imageResultatBytes: _imageResultatBytes,
                                      calqueEquipementPngBytes: _calqueEquipementPngBytes,
                                      dernierCalqueEquipementCompletBytes: _dernierCalqueEquipementCompletBytes,
                                      decalageDuCalqueComplet: _decalageDuCalqueComplet,
                                      isDrawGoulotteMode: _isDrawGoulotteMode,
                                      activePointers: _activePointers,
                                      decalageNotifier: _decalageNotifier,
                                      splitNotifier: _splitNotifier,
                                      isDraggingEquipementNotifier: _isDraggingEquipementNotifier,
                                      goulotteNotifier: _goulotteNotifier,
                                      isDraggingGoulotteNotifier: _isDraggingGoulotteNotifier,
                                      goulotteCurrentEndOrigNotifier: _goulotteCurrentEndOrigNotifier,
                                      magnifierPositionNotifier: _magnifierPositionNotifier,
                                      lastHistoriqueDecalage: _historiqueDecalages.isNotEmpty ? _historiqueDecalages.last : Offset.zero,
                                      onEquipementDropped: () {
                                        if (_historiqueDecalages.isEmpty || _historiqueDecalages.last != _decalageNotifier.value) {
                                          _historiqueDecalages.add(_decalageNotifier.value);
                                          _historiqueLengthNotifier.value = _historiqueDecalages.length;
                                          _historiqueRedoDecalages.clear();
                                          _historiqueRedoLengthNotifier.value = 0;
                                        }
                                        _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true); 
                                      },
                                      onGoulotteCreated: (LigneGoulotte newGoulotte) {
                                        _goulotteInitiale = newGoulotte; 
                                        _goulotteNotifier.value = newGoulotte;
                                        _goulotteRedo = null;
                                        _historiqueRedoLengthNotifier.value = 0;
                                        _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
                                      },
                                      onGoulotteEdited: () {
                                        _goulotteRedo = null;
                                        _historiqueRedoLengthNotifier.value = 0;
                                        _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
                                      },
                                    );
                                  }
                                },
                              ),
                            );
                          }
                        ),
                      ),
                    ),
                    
                    if (_modeleSelectionne != null && !_isManualPlacementMode)
                      Positioned(
                        top: 10,
                        right: 10,
                        child: BoutonsActionDevis(
                          isDraggingEquipementNotifier: _isDraggingEquipementNotifier,
                          isDraggingGoulotteNotifier: _isDraggingGoulotteNotifier,
                          historiqueLengthNotifier: _historiqueLengthNotifier,
                          historiqueRedoLengthNotifier: _historiqueRedoLengthNotifier, 
                          goulotteNotifier: _goulotteNotifier,
                          goulotteInitiale: _goulotteInitiale,
                          isDrawGoulotteMode: _isDrawGoulotteMode,
                          isProcessing: _isProcessing,
                          onUndo: _reinitialiserPosition,
                          onRedo: _retablirPosition, 
                          onResetPosition: _resetPositionEquipement,
                          onToggleGoulotteMode: () {
                            setState(() {
                              _isDrawGoulotteMode = !_isDrawGoulotteMode;
                              _isDraggingEquipementNotifier.value = false; 
                            });
                          },
                          onDeleteConfirmed: () {
                            _goulotteNotifier.value = null; 
                            _goulotteInitiale = null; 
                            _goulotteRedo = null;
                            _historiqueRedoLengthNotifier.value = 0;
                            _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false); 
                          },
                        ),
                      ),

                    if (_isProcessing)
                      Positioned.fill(
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(20),
                          child: ValueListenableBuilder<bool>(
                            valueListenable: _isDraggingEquipementNotifier,
                            builder: (context, isDragging, _) {
                               if (isDragging) return const SizedBox.shrink(); 
                               // APPEL AU NOUVEL OVERLAY DE CHARGEMENT
                               return OverlayChargement(message: _loadingMessage);
                            }
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),

            if (_pointsCibles != null && !_isManualPlacementMode)
              CatalogueDevis(
                categorieSelectionnee: _categorieSelectionnee,
                modeleSelectionne: _modeleSelectionne,
                isProcessing: _isProcessing,
                onCategorieChanged: (catName) {
                  setState(() => _categorieSelectionnee = catName);
                },
                onModeleSelected: (equipement) {
                  setState(() {
                    _modeleSelectionne = equipement;
                    _calqueEquipementPngBytes = null;
                    _dernierCalqueEquipementCompletBytes = null;
                    _historiqueRedoDecalages.clear();
                    _historiqueRedoLengthNotifier.value = 0;

                    _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true);
                  });
                },
              ),

            const SizedBox(height: 80),
          ],
        ),
      ),

      floatingActionButton: _buildPanneauActionBasse(theme),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}