import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:image_gallery_saver_plus/image_gallery_saver_plus.dart';
import 'package:flutter/services.dart';

import '../utils/image_utils.dart';
import '../traitement_image.dart';
import '../services/ia_service.dart';
import '../services/catalogue_service.dart';

// Structure de données pour mémoriser la goulotte unique
class LigneGoulotte {
  final Offset start;
  final Offset end;
  LigneGoulotte(this.start, this.end);
}

// Les différents états d'interaction avec la goulotte
enum DragMode { none, start, end, body, drawingNew }

/// Écran affichant l'image capturée, exécutant la détection de l'IA, 
/// et permettant à l'utilisateur d'incruster et de manipuler des modèles de climatisation.
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
  String _categorieSelectionnee = 'Climatisations';
  Equipement? _modeleSelectionne;
  
  bool _isProcessing = true;
  String _loadingMessage = "Analyse en cours...";
  
  Uint8List? _imageResultatBytes; // L'image finale (avec goulotte + clim)
  
  // NOUVEAU CACHE OPTIMISÉ : On mémorise le Mur + Goulotte.
  // Ainsi, si seule la clim bouge, on ne recalcule JAMAIS la goulotte !
  Uint8List? _imageFondAvecGoulotteBytes; 
  Uint8List? _imageFondPropreBytes;
  
  int? _imageWidth;
  int? _imageHeight;
  List<Map<String, double>>? _pointsCibles;
  bool _isManualPlacementMode = false;

  // Contrôleur pour gérer programmatiquement le zoom et le déplacement de l'image
  final TransformationController _transformationController = TransformationController();

  // =========================================================================
  // GESTION D'ÉTAT OPTIMISÉE (VALUENOTIFIERS)
  // =========================================================================
  final ValueNotifier<Offset> _decalageNotifier = ValueNotifier(Offset.zero);
  final ValueNotifier<double> _splitNotifier = ValueNotifier(1.0);
  final ValueNotifier<bool> _isDraggingNotifier = ValueNotifier(false);

  // Variables d'état pour la Goulotte interactive
  bool _isDrawGoulotteMode = false;
  final ValueNotifier<LigneGoulotte?> _goulotteNotifier = ValueNotifier(null);
  final ValueNotifier<bool> _isDraggingGoulotteNotifier = ValueNotifier(false);

  // Variables temporaires pour ne pas casser le drag du tout premier tracé
  Offset? _goulotteStartOrig;
  final ValueNotifier<Offset?> _goulotteCurrentEndOrigNotifier = ValueNotifier(null);

  // Notifie pour la position de la loupe de précision
  final ValueNotifier<Offset?> _magnifierPositionNotifier = ValueNotifier(null);

  // Compteur de doigts sur l'écran pour empêcher les erreurs lors du zoom
  int _activePointers = 0;

  @override
  void initState() {
    super.initState();
    _analyserImage();
  }

  @override
  void dispose() {
    _transformationController.dispose();
    _decalageNotifier.dispose();
    _splitNotifier.dispose();
    _isDraggingNotifier.dispose();
    _goulotteNotifier.dispose();
    _isDraggingGoulotteNotifier.dispose();
    _goulotteCurrentEndOrigNotifier.dispose();
    _magnifierPositionNotifier.dispose(); // Libération de la mémoire de la loupe
    super.dispose();
  }
  
  // Fonction pour empêcher un point de sortir des limites strictes de l'image
  Offset _clampToImageBounds(Offset point) {
    if (_imageWidth == null || _imageHeight == null) return point;
    return Offset(
      point.dx.clamp(0.0, _imageWidth!.toDouble()),
      point.dy.clamp(0.0, _imageHeight!.toDouble()),
    );
  }

  // Fonction pour forcer la goulotte à être parfaitement droite (horizontale ou verticale)
  Offset _snapToOrthogonal(Offset reference, Offset target) {
    double dx = (target.dx - reference.dx).abs();
    double dy = (target.dy - reference.dy).abs();
    
    if (dx > dy) {
      // Mouvement majoritairement horizontal -> On force l'alignement sur l'axe Y de référence
      return Offset(target.dx, reference.dy);
    } else {
      // Mouvement majoritairement vertical -> On force l'alignement sur l'axe X de référence
      return Offset(reference.dx, target.dy);
    }
  }

  // =========================================================================
  // === LOGIQUE MÉTIER ET IA ===
  // =========================================================================
  Future<void> _analyserImage() async {
    final yoloModel = IAService().yoloModel;
    if (yoloModel == null) return;

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
      var outputMatrix = List.generate(outputShape[0], (i) => 
        List.generate(outputShape[1], (j) => 
          List<double>.filled(outputShape[2], 0.0)
        )
      );

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

        if (confMoyennePoints >= 0.92) {
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
          setState(() => _loadingMessage = "Nettoyage du mur en cours...");
          _imageFondPropreBytes = await compute(TraitementImage.effacerAutocollantIsolate, {
            'photoPath': widget.photoPath,
            'pointsIA': _pointsCibles!,
            'lamaBytes': IAService().lamaBytes,
          });
          _imageFondAvecGoulotteBytes = null; // Sécurité Cache
        }
        
        setState(() {
          _isProcessing = false;
        });
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
      setState(() {
        _pointsCibles = null;
        _isProcessing = false;
      });
    }
  }

  void _demanderPlacementManuel() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Text("Autocollant introuvable"),
        content: const Text("L'IA n'a pas pu détecter l'autocollant avec certitude.\nVoulez-vous placer la zone manuellement ?"),
        actions: [
          ElevatedButton.icon(
            icon: const Icon(Icons.touch_app),
            label: const Text("Placer manuellement"),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.teal, foregroundColor: Colors.white),
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
            child: const Text("Annuler", style: TextStyle(color: Colors.grey)),
          ),
        ],
      ),
    );
  }

  void _activerModeManuel() {
    setState(() {
      _isManualPlacementMode = true;
      _pointsCibles = [
        {'x': 512.0 - 75.0, 'y': 512.0 - 150.0}, // Haut Gauche
        {'x': 512.0 + 75.0, 'y': 512.0 - 150.0}, // Haut Droit
        {'x': 512.0 + 75.0, 'y': 512.0 + 150.0}, // Bas Droit
        {'x': 512.0 - 75.0, 'y': 512.0 + 150.0}, // Bas Gauche
      ];
    });
  }

  Future<void> _validerPlacementManuel() async {
    _transformationController.value = Matrix4.identity();
    setState(() {
      _isManualPlacementMode = false;
      _isProcessing = true;
      _loadingMessage = "Nettoyage de la zone manuelle...";
      _pointsCibles = TraitementImage.trierPoints(_pointsCibles!); 
    });

    try {
      _imageFondPropreBytes = await compute(TraitementImage.effacerAutocollantIsolate, {
        'photoPath': widget.photoPath,
        'pointsIA': _pointsCibles!,
        'lamaBytes': IAService().lamaBytes,
      });
      _imageFondAvecGoulotteBytes = null; // On nettoie le cache car la base a changé
    } catch (e) {
      print("Erreur inpainting manuel : $e");
    }

    setState(() {
      _isProcessing = false;
    });
  }

  // NOUVELLE ARCHITECTURE DE CACHE (Mur -> Goulotte -> Clim)
  Future<void> _genererIncrustation({bool recomputeGoulotte = false}) async {
    if (_pointsCibles == null || _modeleSelectionne == null || _imageFondPropreBytes == null) return;
    
    setState(() {
      _isProcessing = true;
      _loadingMessage = "Préparation du rendu...";
    });

    try {
      String climPath = _modeleSelectionne!.chemin;
      final ByteData data = await DefaultAssetBundle.of(context).load(climPath);
      Uint8List climBytes = data.buffer.asUint8List();
      
      double profondeur = _modeleSelectionne!.profondeur;
      double hauteur = _modeleSelectionne!.hauteur;
      double largeur = _modeleSelectionne!.largeur;

      // Si on modifie la goulotte, on vide son cache pour forcer le recalcul
      if (recomputeGoulotte) {
        _imageFondAvecGoulotteBytes = null;
      }

      double ptHgXOrig = _pointsCibles![0]['x']! * (_imageWidth! / 1024.0);
      double ptHgYOrig = _pointsCibles![0]['y']! * (_imageHeight! / 1024.0);
      double ptHdXOrig = _pointsCibles![1]['x']! * (_imageWidth! / 1024.0);
      double ptHdYOrig = _pointsCibles![1]['y']! * (_imageHeight! / 1024.0);
      double dx = ptHdXOrig - ptHgXOrig;
      double dy = ptHdYOrig - ptHgYOrig;
      
      // autoWPxOrig correspond aux 50 mm physiques de l'autocollant
      double autoWPxOrig = math.sqrt(dx * dx + dy * dy);
      
      // La goulotte a une largeur réelle fixe de 80mm
      double ratioPxParMm = autoWPxOrig / 50.0;
      double largeurGoulotteOrig = 80.0 * ratioPxParMm; 

      // 1. GÉNÉRATION DE LA GOULOTTE (Seulement si elle a changé ou vient d'être tracée)
      if (_goulotteNotifier.value != null && _imageFondAvecGoulotteBytes == null) {
        setState(() => _loadingMessage = "Incrustation de la goulotte...");
        
        _imageFondAvecGoulotteBytes = await compute(TraitementImage.incrusterGoulotteIsolate, {
          'imageAvecClimBytes': _imageFondPropreBytes!, // La Goulotte est dessinée sur le mur propre
          'ptDepartX': _goulotteNotifier.value!.start.dx,
          'ptDepartY': _goulotteNotifier.value!.start.dy,
          'ptArriveeX': _goulotteNotifier.value!.end.dx,
          'ptArriveeY': _goulotteNotifier.value!.end.dy,
          'largeurPx': largeurGoulotteOrig, 
        });
      } else if (_goulotteNotifier.value == null) {
        _imageFondAvecGoulotteBytes = null;
      }

      // 2. GÉNÉRATION DE LA CLIMATISATION PAR-DESSUS LA GOULOTTE
      setState(() => _loadingMessage = "Calcul des ombres de la clim...");
      
      // La base pour OpenCV est le mur contenant déjà la goulotte (ou le mur propre si pas de goulotte)
      Uint8List basePourClim = _imageFondAvecGoulotteBytes ?? _imageFondPropreBytes!;

      Uint8List? resultImage = await compute(TraitementImage.incrusterClimatisationIsolate, {
        'fondPropreBytes': basePourClim,
        'climBytes': climBytes,
        'pointsIA': _pointsCibles!,
        'decalageX': _decalageNotifier.value.dx,
        'decalageY': _decalageNotifier.value.dy,
        'climAssetPath': climPath,
        'profondeurMm': profondeur,
        'hauteurMm': hauteur, 
        'largeurMm': largeur, 
      });

      if (resultImage != null) {
        setState(() {
          _imageResultatBytes = resultImage;
          _splitNotifier.value = 1.0;
        });
      }
    } catch (e) {
      print("[UI/OpenCV - ERREUR] Exception : $e");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  void _reinitialiserPosition() {
    if (_isProcessing) return;
    if (_decalageNotifier.value == Offset.zero) return;
    _decalageNotifier.value = Offset.zero;
    // La goulotte n'ayant pas bougé, on utilise le cache !
    _genererIncrustation(recomputeGoulotte: false); 
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
        name: "Devis_Clim_${DateTime.now().millisecondsSinceEpoch}", 
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

  // =========================================================================
  // === UI : DÉCOUPAGE EN WIDGETS (POUR ALLÉGER LE BUILD) ===
  // =========================================================================

  Widget _buildCalquePlacementManuel(double scale, double offsetX, double offsetY) {
    List<Offset> screenPoints = _pointsCibles!.map((p) {
      double pxOrig = p['x']! * (_imageWidth! / 1024.0);
      double pyOrig = p['y']! * (_imageHeight! / 1024.0);
      return Offset(pxOrig * scale + offsetX, pyOrig * scale + offsetY);
    }).toList();

    double minX = screenPoints.map((p) => p.dx).reduce(math.min);
    double maxX = screenPoints.map((p) => p.dx).reduce(math.max);
    double minY = screenPoints.map((p) => p.dy).reduce(math.min);
    double maxY = screenPoints.map((p) => p.dy).reduce(math.max);

    return Stack(
      children: [
        Positioned.fill(child: Image.file(File(widget.photoPath), fit: BoxFit.contain)),
        Positioned.fill(child: CustomPaint(painter: _BoundingBoxPainter(points: screenPoints))),
        
        Positioned(
          left: minX,
          top: minY,
          width: maxX - minX,
          height: maxY - minY,
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onPanUpdate: (details) {
              // Sécurité : on ignore le déplacement si plusieurs doigts sont détectés
              if (_activePointers > 1) return;
              
              setState(() {
                double dxOrig = details.delta.dx / scale;
                double dyOrig = details.delta.dy / scale;
                double dx1024 = dxOrig * (1024.0 / _imageWidth!);
                double dy1024 = dyOrig * (1024.0 / _imageHeight!);
                
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
              });
            },
            child: Container(color: Colors.transparent),
          ),
        ),

        ...screenPoints.asMap().entries.map((entry) {
          int idx = entry.key;
          Offset pt = entry.value;
          return Positioned(
            left: pt.dx - 15, 
            top: pt.dy - 15,
            child: GestureDetector(
              behavior: HitTestBehavior.opaque, 
              onPanUpdate: (details) {
                // Sécurité multi-touch
                if (_activePointers > 1) return;
                
                setState(() {
                  double dxOrig = details.delta.dx / scale;
                  double dyOrig = details.delta.dy / scale;
                  double dx1024 = dxOrig * (1024.0 / _imageWidth!);
                  double dy1024 = dyOrig * (1024.0 / _imageHeight!);
                  
                  _pointsCibles![idx]['x'] = (_pointsCibles![idx]['x']! + dx1024).clamp(0.0, 1024.0);
                  _pointsCibles![idx]['y'] = (_pointsCibles![idx]['y']! + dy1024).clamp(0.0, 1024.0);
                });
              },
              child: Container(
                width: 30, 
                height: 30,
                color: Colors.transparent, 
                alignment: Alignment.center,
                child: Container(
                  width: 18, 
                  height: 18,
                  decoration: BoxDecoration(
                    color: Colors.blueAccent.withValues(alpha: 0.3),
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.blueAccent, width: 2), 
                  ),
                ),
              ),
            ),
          );
        }),
      ],
    );
  }

  // Construit les zones tactiles (nœuds) pour éditer une goulotte existante sans bloquer le zoom
  List<Widget> _buildGoulotteDraggers(LigneGoulotte goulotte, double scale, double offsetX, double offsetY) {
    Offset p1 = Offset(goulotte.start.dx * scale + offsetX, goulotte.start.dy * scale + offsetY);
    Offset p2 = Offset(goulotte.end.dx * scale + offsetX, goulotte.end.dy * scale + offsetY);
    double len = (p2 - p1).distance;
    double angle = math.atan2(p2.dy - p1.dy, p2.dx - p1.dx);

    return [
      // 1. Noeud de déplacement complet (Corps de la goulotte)
      Positioned(
        left: p1.dx,
        top: p1.dy - 20, // Décale pour centrer le hitbox sur la ligne
        child: Transform.rotate(
          angle: angle,
          alignment: Alignment.centerLeft,
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onPanStart: (_) {
              if (_activePointers > 1) return;
              _isDraggingGoulotteNotifier.value = true;
              // On ne déclenche pas la loupe pour le déplacement du corps entier
            },
            onPanUpdate: (details) {
              if (_activePointers > 1 || !_isDraggingGoulotteNotifier.value) return;
              
              // CORRECTION DU BUG DE ROTATION : 
              // Transform.rotate altère le repère local du GestureDetector.
              // On ré-oriente le "delta" pour retrouver les vraies coordonnées globales de l'écran.
              double cosA = math.cos(angle);
              double sinA = math.sin(angle);
              double globalDx = details.delta.dx * cosA - details.delta.dy * sinA;
              double globalDy = details.delta.dx * sinA + details.delta.dy * cosA;
              Offset deltaOrig = Offset(globalDx, globalDy) / scale;
              
              var g = _goulotteNotifier.value!;
              Offset newStart = g.start + deltaOrig;
              Offset newEnd = g.end + deltaOrig;

              // Sécurité pour empêcher le corps entier de sortir de l'image
              double minX = math.min(newStart.dx, newEnd.dx);
              double maxX = math.max(newStart.dx, newEnd.dx);
              double minY = math.min(newStart.dy, newEnd.dy);
              double maxY = math.max(newStart.dy, newEnd.dy);
              
              double adjustDx = 0;
              double adjustDy = 0;
              
              if (minX < 0) adjustDx = -minX;
              if (maxX > _imageWidth!) adjustDx = _imageWidth! - maxX;
              if (minY < 0) adjustDy = -minY;
              if (maxY > _imageHeight!) adjustDy = _imageHeight! - maxY;
              
              newStart = Offset(newStart.dx + adjustDx, newStart.dy + adjustDy);
              newEnd = Offset(newEnd.dx + adjustDx, newEnd.dy + adjustDy);
              
              _goulotteNotifier.value = LigneGoulotte(newStart, newEnd);
            },
            onPanEnd: (_) {
              if (!_isDraggingGoulotteNotifier.value) return;
              _isDraggingGoulotteNotifier.value = false;
              _genererIncrustation(recomputeGoulotte: true);
            },
            onPanCancel: () {
              if (!_isDraggingGoulotteNotifier.value) return;
              _isDraggingGoulotteNotifier.value = false;
              _genererIncrustation(recomputeGoulotte: true);
            },
            child: Container(width: len, height: 40, color: Colors.transparent),
          ),
        ),
      ),
      // 2. Noeud de redimensionnement de Départ
      Positioned(
        left: p1.dx - 25,
        top: p1.dy - 25,
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onPanStart: (_) {
            if (_activePointers > 1) return;
            _isDraggingGoulotteNotifier.value = true;
            
            // Déclenche la loupe sur ce nœud
            _magnifierPositionNotifier.value = Offset(goulotte.start.dx * scale + offsetX, goulotte.start.dy * scale + offsetY);
          },
          onPanUpdate: (details) {
            if (_activePointers > 1 || !_isDraggingGoulotteNotifier.value) return;
            Offset deltaOrig = details.delta / scale;
            var g = _goulotteNotifier.value!;
            Offset rawStart = g.start + deltaOrig;
            
            // Clamper pour ne pas sortir des limites de l'image
            rawStart = _clampToImageBounds(rawStart);
            
            // On force l'alignement rectiligne parfait (horizontal ou vertical)
            Offset snappedStart = _snapToOrthogonal(g.end, rawStart);
            
            // Re-clamp par sécurité au cas où le snap pousserait le point dehors
            _goulotteNotifier.value = LigneGoulotte(_clampToImageBounds(snappedStart), g.end);
            
            // Mise à jour de la loupe
            _magnifierPositionNotifier.value = Offset(snappedStart.dx * scale + offsetX, snappedStart.dy * scale + offsetY);
          },
          onPanEnd: (_) {
            _magnifierPositionNotifier.value = null; // Cache la loupe
            if (!_isDraggingGoulotteNotifier.value) return;
            _isDraggingGoulotteNotifier.value = false;
            _genererIncrustation(recomputeGoulotte: true);
          },
          onPanCancel: () {
            _magnifierPositionNotifier.value = null; // Cache la loupe
            if (!_isDraggingGoulotteNotifier.value) return;
            _isDraggingGoulotteNotifier.value = false;
            _genererIncrustation(recomputeGoulotte: true);
          },
          child: Container(width: 50, height: 50, color: Colors.transparent),
        ),
      ),
      // 3. Noeud de redimensionnement de Fin
      Positioned(
        left: p2.dx - 25,
        top: p2.dy - 25,
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onPanStart: (_) {
            if (_activePointers > 1) return;
            _isDraggingGoulotteNotifier.value = true;
            
            // Déclenche la loupe sur ce nœud
            _magnifierPositionNotifier.value = Offset(goulotte.end.dx * scale + offsetX, goulotte.end.dy * scale + offsetY);
          },
          onPanUpdate: (details) {
            if (_activePointers > 1 || !_isDraggingGoulotteNotifier.value) return;
            Offset deltaOrig = details.delta / scale;
            var g = _goulotteNotifier.value!;
            Offset rawEnd = g.end + deltaOrig;
            
            // Clamper pour ne pas sortir des limites de l'image
            rawEnd = _clampToImageBounds(rawEnd);
            
            // On force l'alignement rectiligne parfait (horizontal ou vertical)
            Offset snappedEnd = _snapToOrthogonal(g.start, rawEnd);
            
            // Re-clamp par sécurité
            _goulotteNotifier.value = LigneGoulotte(g.start, _clampToImageBounds(snappedEnd));
            
            // Mise à jour de la loupe
            _magnifierPositionNotifier.value = Offset(snappedEnd.dx * scale + offsetX, snappedEnd.dy * scale + offsetY);
          },
          onPanEnd: (_) {
            _magnifierPositionNotifier.value = null; // Cache la loupe
            if (!_isDraggingGoulotteNotifier.value) return;
            _isDraggingGoulotteNotifier.value = false;
            _genererIncrustation(recomputeGoulotte: true);
          },
          onPanCancel: () {
            _magnifierPositionNotifier.value = null; // Cache la loupe
            if (!_isDraggingGoulotteNotifier.value) return;
            _isDraggingGoulotteNotifier.value = false;
            _genererIncrustation(recomputeGoulotte: true);
          },
          child: Container(width: 50, height: 50, color: Colors.transparent),
        ),
      ),
    ];
  }

  Widget _buildCalqueResultat(double scale, double offsetX, double offsetY, BoxConstraints constraints) {
    double ptHgXOrig = _pointsCibles![0]['x']! * (_imageWidth! / 1024.0);
    double ptHgYOrig = _pointsCibles![0]['y']! * (_imageHeight! / 1024.0);
    double ptHdXOrig = _pointsCibles![1]['x']! * (_imageWidth! / 1024.0);
    double ptHdYOrig = _pointsCibles![1]['y']! * (_imageHeight! / 1024.0);

    double dx = ptHdXOrig - ptHgXOrig;
    double dy = ptHdYOrig - ptHgYOrig;
    double autoWPxOrig = math.sqrt(dx * dx + dy * dy);
    
    // On récupère les dimensions depuis le catalogue pour le calcul d'affichage UI
    double largeurMm = 798.0; 
    double hauteurMm = 270.0;
    if (_modeleSelectionne != null) {
      largeurMm = _modeleSelectionne!.largeur;
      hauteurMm = _modeleSelectionne!.hauteur;
    }

    double climWPxOrig = (largeurMm / 50.0) * autoWPxOrig;
    double climHPxOrig = climWPxOrig * (hauteurMm / largeurMm);

    double climScreenW = climWPxOrig * scale;
    double climScreenH = climHPxOrig * scale;

    double angleRad = math.atan2(dy, dx);
    
    // Calcul de l'épaisseur pour le Painter
    double ratioPxParMm = autoWPxOrig / 50.0;
    double largeurGoulotteOrig = 80.0 * ratioPxParMm; 

    return Stack(
      children: [
        // 1. Couche de Fond : Mur Propre OU Mur avec OpenCV Goulotte
        Positioned.fill(
          child: ValueListenableBuilder<bool>( // Écoute le drag de la clim
            valueListenable: _isDraggingNotifier,
            builder: (context, isDraggingClim, _) {
              return ValueListenableBuilder<bool>( // Écoute le drag de la goulotte
                valueListenable: _isDraggingGoulotteNotifier,
                builder: (context, isDraggingGoulotte, _) {
                  Uint8List? imgToShow;
                  if (isDraggingGoulotte) {
                    // Quand on déplace la goulotte, on affiche le mur propre en fond.
                    imgToShow = _imageFondPropreBytes;
                  } else {
                    // Sinon (déplacement clim ou repos), le fond est le mur avec la goulotte (si tracée).
                    imgToShow = _imageFondAvecGoulotteBytes ?? _imageFondPropreBytes;
                  }
                  if (imgToShow == null) {
                    return Image.file(File(widget.photoPath), fit: BoxFit.contain);
                  }
                  return Image.memory(imgToShow, fit: BoxFit.contain, gaplessPlayback: true);
                }
              );
            }
          ),
        ),

        // 2. Couche OpenCV Résultat Complet (Cachée pendant TOUT glissement)
        if (_imageResultatBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: _isDraggingNotifier,
            builder: (context, isDraggingClim, _) {
              return ValueListenableBuilder<bool>(
                valueListenable: _isDraggingGoulotteNotifier,
                builder: (context, isDraggingGoulotte, _) {
                  // Si l'utilisateur touche une pièce, on cache le rendu final
                  if (isDraggingClim || isDraggingGoulotte) return const SizedBox.shrink(); 
                  
                  return ValueListenableBuilder<double>(
                    valueListenable: _splitNotifier,
                    builder: (context, splitVal, _) {
                      double currentSplit = _isDrawGoulotteMode ? 1.0 : splitVal;
                      return Positioned.fill(
                        child: ClipRect(
                          clipper: _SplitClipper(currentSplit),
                          child: Image.memory(_imageResultatBytes!, fit: BoxFit.contain, gaplessPlayback: true), 
                        ),
                      );
                    }
                  );
                }
              );
            }
          ),

        // Le painter global de la goulotte
        Positioned.fill(
          child: IgnorePointer(
            child: ValueListenableBuilder<LigneGoulotte?>(
              valueListenable: _goulotteNotifier,
              builder: (context, goulotte, _) {
                return ValueListenableBuilder<bool>(
                  valueListenable: _isDraggingNotifier, // Etat du drag clim
                  builder: (context, isDraggingClim, _) {
                    return ValueListenableBuilder<bool>(
                      valueListenable: _isDraggingGoulotteNotifier, // Etat du drag goulotte
                      builder: (context, isDraggingGoulotte, _) {
                        
                        // On affiche la ligne 2D temporaire si on drag la goulotte OU LA CLIM !
                        bool showLine = isDraggingClim || isDraggingGoulotte;
                        // On affiche les ronds bleus (nœuds) uniquement si l'outil pinceau est activé
                        bool showNodes = _isDrawGoulotteMode;

                        return CustomPaint(
                          painter: _GoulottePainter(
                            goulotte: goulotte,
                            scale: scale,
                            offsetX: offsetX,
                            offsetY: offsetY,
                            thicknessOrig: largeurGoulotteOrig,
                            showLine: showLine,
                            showNodes: showNodes,
                          ),
                        );
                      }
                    );
                  }
                );
              }
            )
          )
        ),

        // 3. Curseur du slider avant/après
        if (_imageResultatBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: _isDraggingNotifier,
            builder: (context, isDragging, _) {
              // On masque la barre de slider si l'utilisateur trace une goulotte ou drag
              if (isDragging || _isDrawGoulotteMode) return const SizedBox.shrink();
              return ValueListenableBuilder<double>(
                valueListenable: _splitNotifier,
                builder: (context, splitVal, _) {
                  return Positioned(
                    key: const ValueKey('slider_interactif'),
                    top: 0,
                    bottom: 0,
                    left: (constraints.maxWidth * splitVal) - 20, 
                    child: GestureDetector(
                      behavior: HitTestBehavior.opaque,
                      onHorizontalDragUpdate: (details) {
                        // Anti-Missclick pendant le zoom
                        if (_activePointers > 1) return;
                        _splitNotifier.value = (_splitNotifier.value + details.delta.dx / constraints.maxWidth).clamp(0.0, 1.0);
                      },
                      child: SizedBox(
                        width: 40, 
                        child: Stack(
                          alignment: Alignment.center, 
                          children: [
                            Container(width: 3, color: Colors.white),
                            Positioned(
                              bottom: 20, 
                              child: Container(
                                height: 35,
                                width: 35,
                                decoration: const BoxDecoration(
                                  color: Colors.white,
                                  shape: BoxShape.circle,
                                  boxShadow: [BoxShadow(color: Colors.black38, blurRadius: 6, spreadRadius: 1)]
                                ),
                                child: const Icon(Icons.compare_arrows, size: 20, color: Colors.teal),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  );
                }
              );
            }
          ),

        // 4. L'image brute de la Clim (qui s'affiche en transparence UNIQUEMENT pendant un drag)
        if (_modeleSelectionne != null)
          ValueListenableBuilder<Offset>(
            valueListenable: _decalageNotifier,
            builder: (context, decalage, _) {
              return ValueListenableBuilder<bool>(
                valueListenable: _isDraggingNotifier,
                builder: (context, isDraggingClim, _) {
                  double climScreenX = (ptHgXOrig + decalage.dx) * scale + offsetX;
                  double climScreenY = (ptHgYOrig + decalage.dy) * scale + offsetY;
                  
                  return Positioned(
                    key: const ValueKey('clim_draggable'),
                    left: climScreenX,
                    top: climScreenY,
                    width: climScreenW,
                    height: climScreenH,
                    child: IgnorePointer(
                      // Désactive le drag de la clim si on est en mode dessin de goulotte
                      ignoring: _isDrawGoulotteMode,
                      child: GestureDetector(
                        behavior: HitTestBehavior.translucent,
                        onPanStart: (_) {
                          if (_activePointers > 1) return;
                          _isDraggingNotifier.value = true;
                        },
                        onPanUpdate: (details) { 
                           if (_activePointers > 1 || !_isDraggingNotifier.value) return;
                           
                           double cosA = math.cos(angleRad);
                           double sinA = math.sin(angleRad);
                           double globalDx = details.delta.dx * cosA - details.delta.dy * sinA;
                           double globalDy = details.delta.dx * sinA + details.delta.dy * cosA;

                           _decalageNotifier.value = Offset(
                             _decalageNotifier.value.dx + globalDx / scale,
                             _decalageNotifier.value.dy + globalDy / scale
                           );
                        },
                        onPanEnd: (_) { 
                           if (!_isDraggingNotifier.value) return;
                           _isDraggingNotifier.value = false;
                           // On ne re-calcule que la Clim ! C'est ce qui fait gagner du temps.
                           _genererIncrustation(recomputeGoulotte: false); 
                         },
                        onPanCancel: () { 
                           if (!_isDraggingNotifier.value) return;
                           _isDraggingNotifier.value = false;
                           _genererIncrustation(recomputeGoulotte: false); 
                         },
                        child: Transform.rotate(
                          angle: angleRad,
                          alignment: Alignment.topLeft, 
                          // On enveloppe d'un ValueListenableBuilder Goulotte pour que la Clim apparaisse semi-transparente SI la goulotte est en cours de drag
                          child: ValueListenableBuilder<bool>(
                            valueListenable: _isDraggingGoulotteNotifier,
                            builder: (context, isDraggingGoulotte, _) {
                              return Opacity(
                                opacity: (isDraggingClim || isDraggingGoulotte) ? 0.65 : 0.0, 
                                child: Image.asset(_modeleSelectionne!.chemin, fit: BoxFit.fill),
                              );
                            }
                          ),
                        ),
                      ),
                    ),
                  );
                }
              );
            }
          ),

        // Calque interactif invisible avec système Nodal pour la Goulotte
        if (_isDrawGoulotteMode)
          Positioned.fill(
            child: ValueListenableBuilder<LigneGoulotte?>(
              valueListenable: _goulotteNotifier,
              builder: (context, goulotte, _) {
                // La couche d'interaction (CORRECTION DU BUG DE DESSIN)
                if (goulotte == null) {
                  // A - Mode plein écran pour tracer la toute première goulotte
                  return GestureDetector(
                    behavior: HitTestBehavior.opaque,
                    onPanStart: (details) {
                      if (_activePointers > 1) return;
                      // On garde la trace initiale sans provoquer la disparition de la zone tactile
                      Offset touchOrig = (details.localPosition - Offset(offsetX, offsetY)) / scale;
                      
                      // Clamper le point de départ pour ne pas commencer en dehors
                      _goulotteStartOrig = _clampToImageBounds(touchOrig);
                      _goulotteCurrentEndOrigNotifier.value = _goulotteStartOrig;
                      _isDraggingGoulotteNotifier.value = true;
                      
                      // Affiche la loupe
                      _magnifierPositionNotifier.value = Offset(_goulotteStartOrig!.dx * scale + offsetX, _goulotteStartOrig!.dy * scale + offsetY);
                    },
                    onPanUpdate: (details) {
                      if (_activePointers > 1 || !_isDraggingGoulotteNotifier.value) return;
                      if (_goulotteStartOrig != null) {
                        Offset rawEnd = (details.localPosition - Offset(offsetX, offsetY)) / scale;
                        
                        // On clamp pour ne pas sortir de l'image
                        rawEnd = _clampToImageBounds(rawEnd);
                        
                        // Force le tracé à être parfaitement droit
                        Offset snappedEnd = _snapToOrthogonal(_goulotteStartOrig!, rawEnd);
                        
                        // Re-clamp au cas où le snap aurait poussé la ligne en dehors
                        _goulotteCurrentEndOrigNotifier.value = _clampToImageBounds(snappedEnd);
                        
                        // Met à jour la position de la loupe
                        _magnifierPositionNotifier.value = Offset(snappedEnd.dx * scale + offsetX, snappedEnd.dy * scale + offsetY);
                      }
                    },
                    onPanEnd: (_) {
                      _magnifierPositionNotifier.value = null; // Cache la loupe
                      if (!_isDraggingGoulotteNotifier.value) return;
                      if (_goulotteStartOrig != null && _goulotteCurrentEndOrigNotifier.value != null) {
                        _isDraggingGoulotteNotifier.value = false;
                        // Validation finale de la goulotte, ce qui active les nœuds de Drag
                        _goulotteNotifier.value = LigneGoulotte(_goulotteStartOrig!, _goulotteCurrentEndOrigNotifier.value!);
                        _goulotteStartOrig = null;
                        _goulotteCurrentEndOrigNotifier.value = null;
                        _genererIncrustation(recomputeGoulotte: true);
                      }
                    },
                    onPanCancel: () {
                      _magnifierPositionNotifier.value = null; // Cache la loupe
                      if (!_isDraggingGoulotteNotifier.value) return;
                      _isDraggingGoulotteNotifier.value = false;
                      _goulotteStartOrig = null;
                      _goulotteCurrentEndOrigNotifier.value = null;
                    },
                    // Le painter temporaire pour afficher le trait PENDANT sa création
                    child: ValueListenableBuilder<Offset?>(
                      valueListenable: _goulotteCurrentEndOrigNotifier,
                      builder: (context, currentEnd, _) {
                        if (_goulotteStartOrig == null || currentEnd == null) return const SizedBox.shrink();
                        return CustomPaint(
                          painter: _GoulottePainter(
                            goulotte: LigneGoulotte(_goulotteStartOrig!, currentEnd),
                            scale: scale,
                            offsetX: offsetX,
                            offsetY: offsetY,
                            thicknessOrig: largeurGoulotteOrig,
                            showLine: true, 
                            showNodes: true, // Affiche les nœuds pendant la création
                          ),
                        );
                      },
                    ),
                  );
                } else {
                  // B - Mode interactif nodal (laisse 90% de l'écran libre pour le zoom !)
                  return Stack(
                    children: _buildGoulotteDraggers(goulotte, scale, offsetX, offsetY),
                  );
                }
              }
            ),
          ),

        // AJOUT 5. Le calque ultime au-dessus de tout : La Loupe de Précision (Magnifier)
        ValueListenableBuilder<Offset?>(
          valueListenable: _magnifierPositionNotifier,
          builder: (context, magPos, _) {
            // Si aucune position n'est définie (pas de drag), on ne montre rien
            if (magPos == null) return const SizedBox.shrink();
            
            // On centre la loupe horizontalement (magPos.dx - 60)
            // On positionne la loupe 130 pixels AU-DESSUS du doigt (pour ne pas être cachée)
            return Positioned(
              left: magPos.dx - 60, // 60 = moitié de la largeur de la loupe (120/2)
              top: magPos.dy - 130, // Décale vers le haut
              child: RawMagnifier(
                decoration: const MagnifierDecoration(
                  shape: CircleBorder(
                    side: BorderSide(color: Colors.teal, width: 2),
                  ),
                  shadows: [
                    BoxShadow(color: Colors.black26, blurRadius: 8, spreadRadius: 2)
                  ],
                ),
                size: const Size(120, 120),
                magnificationScale: 2.0,
                // Le point focal regarde 70 pixels plus bas que le centre de la loupe
                // Ce qui pointe EXACTEMENT sous le doigt de l'utilisateur !
                focalPointOffset: const Offset(0, 70),
              ),
            );
          }
        ),

        if (_isProcessing)
          ValueListenableBuilder<bool>(
            valueListenable: _isDraggingNotifier,
            builder: (context, isDragging, _) {
               if (isDragging) return const SizedBox.shrink(); 
               return Positioned.fill(
                child: Container(
                  color: Colors.black54,
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const CircularProgressIndicator(color: Colors.white),
                        const SizedBox(height: 16),
                        Text(_loadingMessage, style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold), textAlign: TextAlign.center),
                      ],
                    )
                  )
                ),
              );
            }
          ),
      ],
    );
  }

  Widget _buildCatalogue() {
    final catalogueGlobal = CatalogueService().catalogueGlobal;

    return Container(
      height: 190,
      padding: const EdgeInsets.only(top: 10, bottom: 10),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10, offset: const Offset(0, -5))],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            height: 40,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              itemCount: catalogueGlobal.keys.length,
              itemBuilder: (context, index) {
                String catName = catalogueGlobal.keys.elementAt(index);
                bool isSelected = _categorieSelectionnee == catName;
                
                return Padding(
                  padding: const EdgeInsets.only(right: 8.0),
                  child: ChoiceChip(
                    label: Text(catName, style: TextStyle(fontWeight: isSelected ? FontWeight.bold : FontWeight.normal)),
                    selected: isSelected,
                    selectedColor: Colors.teal.shade100,
                    checkmarkColor: Colors.teal.shade800,
                    onSelected: (bool selected) {
                      if (selected && !_isProcessing) {
                        setState(() => _categorieSelectionnee = catName);
                      }
                    },
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 15),
          
          Expanded(
            child: catalogueGlobal[_categorieSelectionnee]!.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.build_circle_outlined, size: 40, color: Colors.grey.shade400),
                        const SizedBox(height: 8),
                        Text("Cette catégorie sera ajoutée prochainement", style: TextStyle(color: Colors.grey.shade600, fontStyle: FontStyle.italic, fontWeight: FontWeight.w500)),
                      ],
                    ),
                  )
                : ListView.builder(
                    scrollDirection: Axis.horizontal,
                    itemCount: catalogueGlobal[_categorieSelectionnee]!.length,
                    itemBuilder: (context, index) {
                      final clim = catalogueGlobal[_categorieSelectionnee]![index];
                      final bool isSelected = _modeleSelectionne == clim;

                      return GestureDetector(
                        onTap: () {
                          if (_isProcessing) return;
                          setState(() {
                            _modeleSelectionne = clim;
                            // En cas de changement de modèle, on ne recalcule PAS la goulotte car la base n'a pas changé.
                            _genererIncrustation(recomputeGoulotte: false);
                          });
                        },
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 200),
                          width: 120,
                          margin: EdgeInsets.only(left: 16.0, right: index == catalogueGlobal[_categorieSelectionnee]!.length - 1 ? 16.0 : 0.0),
                          decoration: BoxDecoration(
                            color: isSelected ? Colors.teal.withValues(alpha : 0.1) : Colors.white,
                            border: Border.all(color: isSelected ? Colors.teal : Colors.grey.shade300, width: isSelected ? 3 : 1),
                            borderRadius: BorderRadius.circular(15),
                            boxShadow: [if (isSelected) BoxShadow(color: Colors.teal.withValues(alpha : 0.2), blurRadius: 8, offset: const Offset(0, 4))],
                          ),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Expanded(child: Padding(padding: const EdgeInsets.all(8.0), child: Image.asset(clim.chemin, fit: BoxFit.contain))),
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 4.0, vertical: 8.0),
                                child: Text(clim.nom, style: TextStyle(fontSize: 12, fontWeight: isSelected ? FontWeight.bold : FontWeight.normal, color: isSelected ? Colors.teal.shade800 : Colors.black87), textAlign: TextAlign.center, maxLines: 2),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  // =========================================================================
  // === BUILD PRINCIPAL ===
  // =========================================================================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Configuration du Devis'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      // Permet de compter le nombre de doigts à l'écran
      body: Listener(
        onPointerDown: (_) {
          _activePointers++;
          if (_activePointers > 1) {
            // Sécurité Anti-Missclick pendant le Zoom
            // On annule silencieusement tous les glissements en cours (clim ou goulotte)
            _isDraggingNotifier.value = false;
            _isDraggingGoulotteNotifier.value = false;
            _goulotteStartOrig = null;
            _goulotteCurrentEndOrigNotifier.value = null;
            _magnifierPositionNotifier.value = null; // Cache la loupe
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
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(15),
                        // Écoute l'état de la goulotte pour réactiver intelligemment le Pan/Zoom
                        child: ValueListenableBuilder<LigneGoulotte?>(
                          valueListenable: _goulotteNotifier,
                          builder: (context, goulotte, _) {
                            // Si le mode Goulotte n'est pas actif, OU si une goulotte existe déjà (et utilise le drag nodale) : on autorise le zoom !
                            bool isPanZoomEnabled = !_isDrawGoulotteMode || goulotte != null;
                            
                            return InteractiveViewer(
                              transformationController: _transformationController, 
                              panEnabled: isPanZoomEnabled,
                              scaleEnabled: isPanZoomEnabled,
                              minScale: 1.0,
                              maxScale: 8.0,
                              child: LayoutBuilder(
                                builder: (context, constraints) {
                                  if (_imageWidth == null || _imageHeight == null) {
                                     return _isProcessing
                                         ? Center(
                                            child: Column(
                                              mainAxisSize: MainAxisSize.min,
                                              children: [
                                                const CircularProgressIndicator(),
                                                const SizedBox(height: 16),
                                                Text(_loadingMessage, style: const TextStyle(fontWeight: FontWeight.bold)),
                                              ],
                                            ),
                                          )
                                        : Image.file(File(widget.photoPath), fit: BoxFit.contain);
                                  }

                                  if (_pointsCibles == null && !_isManualPlacementMode) {
                                     return Image.file(File(widget.photoPath), fit: BoxFit.contain);
                                  }

                                  double scale = math.min(constraints.maxWidth / _imageWidth!, constraints.maxHeight / _imageHeight!);
                                  double offsetX = (constraints.maxWidth - (_imageWidth! * scale)) / 2;
                                  double offsetY = (constraints.maxHeight - (_imageHeight! * scale)) / 2;

                                  if (_isManualPlacementMode) {
                                    return _buildCalquePlacementManuel(scale, offsetX, offsetY);
                                  } else {
                                    return _buildCalqueResultat(scale, offsetX, offsetY, constraints);
                                  }
                                },
                              ),
                            );
                          }
                        ),
                      ),
                    ),
                    
                    // Colonne de boutons d'action au dessus de l'image
                    if (_modeleSelectionne != null && !_isManualPlacementMode)
                      Positioned(
                        top: 10,
                        right: 10,
                        child: Column(
                          children: [
                            ValueListenableBuilder<Offset>(
                              valueListenable: _decalageNotifier,
                              builder: (context, decalage, _) {
                                if (decalage == Offset.zero) return const SizedBox.shrink();
                                return Padding(
                                  padding: const EdgeInsets.only(bottom: 8.0),
                                  child: Material(
                                    color: Colors.white.withValues(alpha: 0.9),
                                    shape: const CircleBorder(),
                                    elevation: 4,
                                    child: IconButton(
                                      icon: const Icon(Icons.restore),
                                      color: Colors.teal,
                                      tooltip: 'Réinitialiser la position',
                                      onPressed: _isProcessing ? null : _reinitialiserPosition,
                                    ),
                                  ),
                                );
                              }
                            ),
                            
                            // Bouton d'activation du Mode Goulotte
                            Padding(
                              padding: const EdgeInsets.only(bottom: 8.0),
                              child: Material(
                                color: _isDrawGoulotteMode ? Colors.teal : Colors.white.withValues(alpha: 0.9),
                                shape: const CircleBorder(),
                                elevation: 4,
                                child: IconButton(
                                  icon: Icon(Icons.format_paint, color: _isDrawGoulotteMode ? Colors.white : Colors.teal),
                                  tooltip: 'Tracer une goulotte',
                                  onPressed: _isProcessing ? null : () {
                                    setState(() {
                                      _isDrawGoulotteMode = !_isDrawGoulotteMode;
                                      _isDraggingNotifier.value = false; 
                                    });
                                  },
                                ),
                              ),
                            ),
                            
                            // Bouton pour ENLEVER la goulotte (Unique goulotte)
                            if (_isDrawGoulotteMode) // Apparaît uniquement en mode goulotte
                              ValueListenableBuilder<LigneGoulotte?>(
                                valueListenable: _goulotteNotifier,
                                builder: (context, goulotteActuelle, _) {
                                  if (goulotteActuelle == null) return const SizedBox.shrink();
                                  return Padding(
                                    padding: const EdgeInsets.only(bottom: 8.0),
                                    child: Material(
                                      color: Colors.white.withValues(alpha: 0.9),
                                      shape: const CircleBorder(),
                                      elevation: 4,
                                      child: IconButton(
                                        icon: const Icon(Icons.delete_outline),
                                        color: Colors.red,
                                        tooltip: 'Supprimer la goulotte',
                                        onPressed: _isProcessing ? null : () {
                                          // Sécurité pour éviter le missclick
                                          showDialog(
                                            context: context,
                                            builder: (BuildContext context) {
                                              return AlertDialog(
                                                title: const Text("Supprimer la goulotte"),
                                                content: const Text("Êtes-vous sûr de vouloir effacer cette goulotte ?"),
                                                actions: [
                                                  TextButton(
                                                    onPressed: () => Navigator.of(context).pop(),
                                                    child: const Text("Annuler", style: TextStyle(color: Colors.grey)),
                                                  ),
                                                  ElevatedButton(
                                                    style: ElevatedButton.styleFrom(backgroundColor: Colors.red, foregroundColor: Colors.white),
                                                    onPressed: () {
                                                      Navigator.of(context).pop();
                                                      _goulotteNotifier.value = null; // Vide la goulotte unique
                                                      // On force le recalcul de la goulotte (qui devient nulle)
                                                      _genererIncrustation(recomputeGoulotte: true); 
                                                    },
                                                    child: const Text("Supprimer"),
                                                  ),
                                                ],
                                              );
                                            },
                                          );
                                        },
                                      ),
                                    ),
                                  );
                                }
                              ),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ),

            if (_pointsCibles != null && !_isManualPlacementMode) _buildCatalogue(),

            const SizedBox(height: 80),
          ],
        ),
      ),
      
      floatingActionButton: _isManualPlacementMode
          ? FloatingActionButton.extended(
              onPressed: _validerPlacementManuel,
              label: const Text("Valider la position", style: TextStyle(fontWeight: FontWeight.bold)),
              icon: const Icon(Icons.check),
              backgroundColor: Colors.blueAccent,
              foregroundColor: Colors.white,
            )
          : (_imageResultatBytes != null && !_isProcessing)
              ? FloatingActionButton.extended(
                  onPressed: _sauvegarderImage,
                  label: const Text("Sauvegarder", style: TextStyle(fontWeight: FontWeight.bold)),
                  icon: const Icon(Icons.download),
                  backgroundColor: Theme.of(context).colorScheme.primary,
                  foregroundColor: Colors.white,
                )
              : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}

/// =========================================================================
/// === CLASSES UTILITAIRES (DESSIN ET DÉCOUPAGE) ===
/// =========================================================================

/// CustomPainter responsable de dessiner la goulotte vectorielle et ses points nodaux.
/// L'affichage est conditionnel pour optimiser les performances.
class _GoulottePainter extends CustomPainter {
  final LigneGoulotte? goulotte;
  final double scale;
  final double offsetX;
  final double offsetY;
  final double thicknessOrig; 
  final bool showLine; // Gère l'affichage du corps de la goulotte
  final bool showNodes; // Gère l'affichage des nœuds bleus (ronds)

  _GoulottePainter({
    required this.goulotte, 
    required this.scale, 
    required this.offsetX, 
    required this.offsetY, 
    required this.thicknessOrig, 
    required this.showLine,
    required this.showNodes,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (goulotte == null) return;

    final p1 = Offset(goulotte!.start.dx * scale + offsetX, goulotte!.start.dy * scale + offsetY);
    final p2 = Offset(goulotte!.end.dx * scale + offsetX, goulotte!.end.dy * scale + offsetY);
    
    // On dessine le trait vectoriel si demandé (quand l'image cuite d'OpenCV est masquée par un déplacement)
    if (showLine) {
      final paintLine = Paint()
        ..color = Colors.white70
        ..strokeWidth = thicknessOrig * scale 
        ..strokeCap = StrokeCap.butt; // Bout parfaitement plat
        
      canvas.drawLine(p1, p2, paintLine);
    }

    // Indicateurs nodaux (les ronds) pour montrer à l'utilisateur où grab/tirer la ligne
    if (showNodes) {
      final paintHandle = Paint()..color = Colors.blueAccent;
      final paintHandleBorder = Paint()..color = Colors.white..style = PaintingStyle.stroke..strokeWidth = 1.5; // Bordure plus fine
      
      // Ronds plus petits (7.0 au lieu de 15.0) pour ne pas cacher les extrémités
      canvas.drawCircle(p1, 7.0, paintHandle);
      canvas.drawCircle(p1, 7.0, paintHandleBorder);
      canvas.drawCircle(p2, 7.0, paintHandle);
      canvas.drawCircle(p2, 7.0, paintHandleBorder);
    }
  }

  @override
  bool shouldRepaint(covariant _GoulottePainter oldDelegate) => true;
}

/// CustomClipper utilisé pour créer l'effet de séparation (Slider Split Screen).
/// Permet de comparer le mur original avec le mur traité par l'IA.
class _SplitClipper extends CustomClipper<Rect> {
  final double percentage;
  _SplitClipper(this.percentage);

  @override
  Rect getClip(Size size) {
    return Rect.fromLTRB(0, 0, size.width * percentage, size.height);
  }

  @override
  bool shouldReclip(_SplitClipper oldClipper) => percentage != oldClipper.percentage;
}

/// CustomPainter utilisé pour dessiner la zone de sélection manuelle (Bounding Box) et sa surface bleutée.
class _BoundingBoxPainter extends CustomPainter {
  final List<Offset> points;
  _BoundingBoxPainter({required this.points});

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length != 4) return;
    
    final paint = Paint()
      ..color = Colors.blueAccent
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;
      
    final path = Path()
      ..moveTo(points[0].dx, points[0].dy)
      ..lineTo(points[1].dx, points[1].dy)
      ..lineTo(points[2].dx, points[2].dy)
      ..lineTo(points[3].dx, points[3].dy)
      ..close();
      
    canvas.drawPath(path, paint);

    final fillPaint = Paint()
      ..color = Colors.blueAccent.withValues(alpha: 0.2)
      ..style = PaintingStyle.fill;
    canvas.drawPath(path, fillPaint);
  }

  @override
  bool shouldRepaint(_BoundingBoxPainter oldDelegate) => true;
}