import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:image_gallery_saver_plus/image_gallery_saver_plus.dart';
import 'package:flutter/services.dart';

import '../utils/image_utils.dart';
import '../traitement_image.dart';
import '../services/ia_service.dart';

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
  
  // Catalogue rangé par catégories pour faciliter l'affichage et la sélection dans l'UI.
  final Map<String, List<Map<String, dynamic>>> catalogueGlobal = {
    'Climatisations': [
      {
        'nom': 'Takao Plus Blanc',
        'chemin': 'assets/installations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png',
        'profondeur': 240.0
      },
      {
        'nom': 'Takao Plus Noir',
        'chemin': 'assets/installations/clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png',
        'profondeur': 240.0
      }
    ],
    'Pompes à Chaleur': [], 
    'Chaudières': [],       
  };

  String _categorieSelectionnee = 'Climatisations';
  String? _modeleSelectionneChemin;
  bool _isProcessing = true;
  String _loadingMessage = "Analyse en cours...";
  
  Uint8List? _imageResultatBytes;
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
    super.dispose();
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
          List.generate(outputShape[2], (k) => 0.0)
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
    } catch (e) {
      print("Erreur inpainting manuel : $e");
    }

    setState(() {
      _isProcessing = false;
    });
  }

  Future<void> _genererIncrustation() async {
    if (_pointsCibles == null || _modeleSelectionneChemin == null || _imageFondPropreBytes == null) return;
    
    setState(() {
      _isProcessing = true;
      _loadingMessage = "Calcul des ombres et lumières...";
    });

    try {
      String climPath = _modeleSelectionneChemin!;
      final ByteData data = await DefaultAssetBundle.of(context).load(climPath);
      Uint8List climBytes = data.buffer.asUint8List();
      
      double profondeur = catalogueGlobal[_categorieSelectionnee]!
          .firstWhere((c) => c['chemin'] == climPath)['profondeur'] as double;

      Uint8List? resultImage = await compute(TraitementImage.incrusterClimatisationIsolate, {
        'fondPropreBytes': _imageFondPropreBytes!,
        'climBytes': climBytes,
        'pointsIA': _pointsCibles!,
        'decalageX': _decalageNotifier.value.dx,
        'decalageY': _decalageNotifier.value.dy,
        'climAssetPath': climPath,
        'profondeurMm': profondeur,
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
    _genererIncrustation();
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

  Widget _buildCalqueResultat(double scale, double offsetX, double offsetY, BoxConstraints constraints) {
    double ptHgXOrig = _pointsCibles![0]['x']! * (_imageWidth! / 1024.0);
    double ptHgYOrig = _pointsCibles![0]['y']! * (_imageHeight! / 1024.0);
    double ptHdXOrig = _pointsCibles![1]['x']! * (_imageWidth! / 1024.0);
    double ptHdYOrig = _pointsCibles![1]['y']! * (_imageHeight! / 1024.0);

    double dx = ptHdXOrig - ptHgXOrig;
    double dy = ptHdYOrig - ptHgYOrig;
    double autoWPxOrig = math.sqrt(dx * dx + dy * dy);
    
    double climWPxOrig = (798.0 / 50.0) * autoWPxOrig;
    double climHPxOrig = climWPxOrig * (270.0 / 798.0);
    double climScreenW = climWPxOrig * scale;
    double climScreenH = climHPxOrig * scale;
    double angleRad = math.atan2(dy, dx);

    return Stack(
      children: [
        Positioned.fill(
          child: _imageFondPropreBytes != null
              ? Image.memory(_imageFondPropreBytes!, fit: BoxFit.contain)
              : Image.file(File(widget.photoPath), fit: BoxFit.contain),
        ),

        if (_imageResultatBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: _isDraggingNotifier,
            builder: (context, isDragging, _) {
              if (isDragging) return const SizedBox.shrink(); 
              return ValueListenableBuilder<double>(
                valueListenable: _splitNotifier,
                builder: (context, splitVal, _) {
                  return Positioned.fill(
                    child: ClipRect(
                      clipper: _SplitClipper(splitVal),
                      child: Image.memory(_imageResultatBytes!, fit: BoxFit.contain),
                    ),
                  );
                }
              );
            }
          ),

        if (_imageResultatBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: _isDraggingNotifier,
            builder: (context, isDragging, _) {
              if (isDragging) return const SizedBox.shrink();
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

        if (_modeleSelectionneChemin != null && catalogueGlobal[_categorieSelectionnee]!.any((c) => c['chemin'] == _modeleSelectionneChemin))
          ValueListenableBuilder<Offset>(
            valueListenable: _decalageNotifier,
            builder: (context, decalage, _) {
              return ValueListenableBuilder<bool>(
                valueListenable: _isDraggingNotifier,
                builder: (context, isDragging, _) {
                  double climScreenX = (ptHgXOrig + decalage.dx) * scale + offsetX;
                  double climScreenY = (ptHgYOrig + decalage.dy) * scale + offsetY;
                  
                  return Positioned(
                    key: const ValueKey('clim_draggable'),
                    left: climScreenX,
                    top: climScreenY,
                    width: climScreenW,
                    height: climScreenH,
                    child: GestureDetector(
                      behavior: HitTestBehavior.translucent,
                      onPanStart: (_) => _isDraggingNotifier.value = true,
                      onPanUpdate: (details) {
                         _decalageNotifier.value = Offset(
                           _decalageNotifier.value.dx + details.delta.dx / scale,
                           _decalageNotifier.value.dy + details.delta.dy / scale
                         );
                      },
                      onPanEnd: (_) {
                         _isDraggingNotifier.value = false;
                         _genererIncrustation(); 
                      },
                      onPanCancel: () {
                         _isDraggingNotifier.value = false;
                         _genererIncrustation(); 
                      },
                      child: Transform.rotate(
                        angle: angleRad,
                        alignment: Alignment.topLeft, 
                        child: Opacity(
                          opacity: isDragging ? 0.65 : 0.0, 
                          child: Image.asset(_modeleSelectionneChemin!, fit: BoxFit.fill),
                        ),
                      ),
                    ),
                  );
                }
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
                      final bool isSelected = _modeleSelectionneChemin == clim['chemin'];

                      return GestureDetector(
                        onTap: () {
                          if (_isProcessing) return;
                          setState(() {
                            _modeleSelectionneChemin = clim['chemin'];
                            _genererIncrustation();
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
                              Expanded(child: Padding(padding: const EdgeInsets.all(8.0), child: Image.asset(clim['chemin']!, fit: BoxFit.contain))),
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 4.0, vertical: 8.0),
                                child: Text(clim['nom']!, style: TextStyle(fontSize: 12, fontWeight: isSelected ? FontWeight.bold : FontWeight.normal, color: isSelected ? Colors.teal.shade800 : Colors.black87), textAlign: TextAlign.center, maxLines: 2),
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
      body: Column(
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
                      child: InteractiveViewer(
                          transformationController: _transformationController, 
                          panEnabled: true,
                          scaleEnabled: true,
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
                        ),
                    ),
                  ),
                  
                  if (_modeleSelectionneChemin != null && !_isManualPlacementMode)
                    ValueListenableBuilder<Offset>(
                      valueListenable: _decalageNotifier,
                      builder: (context, decalage, _) {
                        if (decalage == Offset.zero) return const SizedBox.shrink();
                        return Positioned(
                          top: 10,
                          right: 10,
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
                ],
              ),
            ),
          ),

          if (_pointsCibles != null && !_isManualPlacementMode) _buildCatalogue(),
          const SizedBox(height: 80),
        ],
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

// =========================================================================
// === CLASSES UTILITAIRES (DESSIN ET DÉCOUPAGE) ===
// =========================================================================

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