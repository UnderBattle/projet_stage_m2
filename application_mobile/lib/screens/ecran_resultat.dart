import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image_gallery_saver_plus/image_gallery_saver_plus.dart';
import 'package:flutter/services.dart';

import '../utils/image_utils.dart';
import '../traitement_image.dart';

/// Écran affichant l'image capturée, exécutant la détection de l'IA, 
/// et permettant à l'utilisateur d'incruster et de manipuler des modèles de climatisation.
class EcranResultat extends StatefulWidget {
  final String photoPath; 
  const EcranResultat({super.key, required this.photoPath});

  @override
  State<EcranResultat> createState() => _EcranResultatState();
}

class _EcranResultatState extends State<EcranResultat> {
  
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
    'Pompes à Chaleur': [], // Vide pour le moment
    'Chaudières': [],       // Vide pour le moment
  };

  // Suivi de la catégorie actuellement affichée à l'écran
  String _categorieSelectionnee = 'Climatisations';
  
  // Variables d'état gérant le modèle sélectionné, l'indicateur de chargement et le modèle IA.
  String? _modeleSelectionneChemin;
  bool _isProcessing = true;
  
  // Variable pour informer l'utilisateur de l'étape en cours
  String _loadingMessage = "Initialisation...";
  
  Interpreter? _iaModel;
  
  // Stockage en mémoire vive du modèle LaMa
  Uint8List? _lamaBytes;
  
  Uint8List? _imageResultatBytes;
  Uint8List? _imageFondPropreBytes;

  int? _imageWidth;
  int? _imageHeight;
  List<Map<String, double>>? _pointsCibles;
  double _decalageX = 0.0;
  double _decalageY = 0.0;
  bool _isDragging = false;

  @override
  void initState() {
    super.initState();
    _lancerProcessusAutomatique();
  }

  @override
  void dispose() {
    _iaModel?.close(); 
    super.dispose();
  }

  /// Charge le modèle TFLite en mémoire puis déclenche l'analyse de l'image.
  Future<void> _lancerProcessusAutomatique() async {
    try {
      setState(() => _loadingMessage = "Chargement des modèles IA...");
      
      // Configure les options d'interpréteur pour optimiser les performances selon la plateforme
      final interpreterOptions = InterpreterOptions();
      
      if (Platform.isAndroid) {
        interpreterOptions.addDelegate(XNNPackDelegate()); 
      } else if (Platform.isIOS) {
        interpreterOptions.addDelegate(GpuDelegate());
      }

      // On passe les options au chargement de YOLO
      _iaModel = await Interpreter.fromAsset(
        'assets/best.tflite', 
        options: interpreterOptions
      );
      
      try {
        final ByteData lamaData = await rootBundle.load('assets/lama_dynamic_45mo.tflite');
        _lamaBytes = lamaData.buffer.asUint8List();
      } catch (e) {
        print("Erreur LaMa: $e");
      }

      await _analyserImage();
    } catch (e) {
      print("[IA] Erreur : $e");
    }
  }

  /// Prépare l'image, exécute l'inférence avec YOLOv8, extrait les points de l'autocollant, 
  /// puis lance l'effacement de l'autocollant (Inpainting) dans un Isolate.
  Future<void> _analyserImage() async {
    if (_iaModel == null) return;

    try {
      setState(() => _loadingMessage = "Détection de l'autocollant par l'IA...");
      final imageBytes = await File(widget.photoPath).readAsBytes();
      var inputShape = _iaModel!.getInputTensor(0).shape;
      bool isNHWC = inputShape[3] == 3;
      
      final resultMatrixPrep = await compute(prepareImageMatrixForIA, {
        'bytes': imageBytes,
        'isNHWC': isNHWC
      });

      if (resultMatrixPrep == null) throw Exception("Impossible de lire l'image.");

      _imageWidth = resultMatrixPrep['width'];
      _imageHeight = resultMatrixPrep['height'];
      var inputMatrix = resultMatrixPrep['matrix'];

      // Prépare la matrice de sortie pour stocker les prédictions du modèle.
      var outputShape = _iaModel!.getOutputTensor(0).shape;
      var outputMatrix = List.generate(outputShape[0], (i) =>
        List.generate(outputShape[1], (j) =>
          List.generate(outputShape[2], (k) => 0.0)
        )
      );

      // Exécute le modèle IA.
      _iaModel!.run(inputMatrix, outputMatrix);

      // Parcourt les prédictions pour trouver celle avec la plus haute confiance.
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
        // Extrait les coordonnées de la boîte englobante (bounding box).
        double boxX = isTransposed ? outputMatrix[0][meilleurIndex][0] : outputMatrix[0][0][meilleurIndex];
        double boxY = isTransposed ? outputMatrix[0][meilleurIndex][1] : outputMatrix[0][1][meilleurIndex];
        double boxW = isTransposed ? outputMatrix[0][meilleurIndex][2] : outputMatrix[0][2][meilleurIndex];
        double boxH = isTransposed ? outputMatrix[0][meilleurIndex][3] : outputMatrix[0][3][meilleurIndex];
        
        double scale = (boxW < 2.0 && boxH < 2.0) ? 1024.0 : 1.0;

        List<Map<String, double>> rawPoints = [];
        double confMoyennePoints = 0;

        // Extrait les 4 points clés (pose) correspondants aux coins de l'autocollant.
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

        // Si la confiance des points est très bonne, on les utilise directement.
        if (confMoyennePoints >= 0.92) {
          _pointsCibles = TraitementImage.trierPoints(rawPoints);
        } else {
          // Sinon, par sécurité, on se rabat sur les 4 coins de la boîte englobante.
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
           // Lance l'effacement de l'autocollant du mur en tâche de fond.
           _imageFondPropreBytes = await compute(TraitementImage.effacerAutocollantIsolate, {
             'photoPath': widget.photoPath,
             'pointsIA': _pointsCibles!,
             'lamaBytes': _lamaBytes, 
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
      }

    } catch (e) {
      print("[IA - ERREUR] Exception : $e");
      setState(() {
        _pointsCibles = null;
        _isProcessing = false;
      });
    } 
  }

  /// Génère l'incrustation finale de la climatisation choisie sur le mur.
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
      
      // On récupère la profondeur depuis la catégorie actuellement sélectionnée
      double profondeur = catalogueGlobal[_categorieSelectionnee]!
          .firstWhere((c) => c['chemin'] == climPath)['profondeur'] as double;

      // On passe le "Fond Propre" déjà calculé pour éviter de relancer l'Inpainting IA !
      Uint8List? resultImage = await compute(TraitementImage.incrusterClimatisationIsolate, {
        'fondPropreBytes': _imageFondPropreBytes!,
        'climBytes': climBytes,
        'pointsIA': _pointsCibles!,
        'decalageX': _decalageX,
        'decalageY': _decalageY,
        'climAssetPath': climPath,
        'profondeurMm': profondeur,
      });

      if (resultImage != null) {
        setState(() {
          _imageResultatBytes = resultImage;
        });
      }
    } catch (e) {
      print("[UI/OpenCV - ERREUR] Exception : $e");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  // Fonction pour remettre la climatisation à son point d'origine
  void _reinitialiserPosition() {
    if (_isProcessing) return; 
    if (_decalageX == 0.0 && _decalageY == 0.0) return; 

    setState(() {
      _decalageX = 0.0;
      _decalageY = 0.0;
    });

    _genererIncrustation();
  }

  /// Enregistre l'image finale résultant de l'incrustation dans la galerie de l'appareil.
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
          const SnackBar(
            content: Text('Simulation sauvegardée dans la galerie'),
            backgroundColor: Colors.green,
            duration: Duration(seconds: 3),
          ),
        );
      } else {
        throw Exception("Échec de la sauvegarde.");
      }
    } catch (e) {
      if (!mounted) return; 
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Erreur lors de la sauvegarde.'), backgroundColor: Colors.red),
      );
    }
  }

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
              // On englobe tout le visualiseur dans un Stack
              child: Stack(
                children: [
                  Positioned.fill(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(15),
                      // Zone de visualisation interactive permettant de zoomer et de se déplacer dans l'image.
                      child: InteractiveViewer(
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

                              if (_pointsCibles == null) {
                                 return Stack(
                                   children: [
                                     Positioned.fill(child: Image.file(File(widget.photoPath), fit: BoxFit.contain)),
                                     if (!_isProcessing)
                                       Center(
                                         child: Container(
                                           padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                                           decoration: BoxDecoration(color: Colors.teal.withValues(alpha: 0.8), borderRadius: BorderRadius.circular(12)),
                                           child: const Column(
                                             mainAxisSize: MainAxisSize.min,
                                             children: [
                                               Icon(Icons.error_outline, color: Colors.redAccent, size: 40),
                                               SizedBox(height: 10),
                                               Text("Aucun autocollant détecté sur la photo", style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold), textAlign: TextAlign.center),
                                             ],
                                            ),
                                         ),
                                       ),
                                   ],
                                 );
                              }

                              // Calcule l'échelle d'affichage de l'image pour positionner la climatisation correctement.
                              double viewW = constraints.maxWidth;
                              double viewH = constraints.maxHeight;
                              double scale = math.min(viewW / _imageWidth!, viewH / _imageHeight!);
                              
                              double displayW = _imageWidth! * scale;
                              double displayH = _imageHeight! * scale;
                              double offsetX = (viewW - displayW) / 2;
                              double offsetY = (viewH - displayH) / 2;

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
                              
                              double climScreenX = (ptHgXOrig + _decalageX) * scale + offsetX;
                              double climScreenY = (ptHgYOrig + _decalageY) * scale + offsetY;

                              double angleRad = math.atan2(dy, dx);

                              return Stack(
                                children: [
                                  Positioned.fill(
                                    // On affiche le fond propre dès qu'il est prêt, même si aucune clim n'est choisie !
                                    child: (_isDragging && _imageFondPropreBytes != null)
                                        ? Image.memory(_imageFondPropreBytes!, fit: BoxFit.contain)
                                        : (_imageResultatBytes != null)
                                            ? Image.memory(_imageResultatBytes!,   fit: BoxFit.contain)
                                            : (_imageFondPropreBytes != null)
                                                ? Image.memory(_imageFondPropreBytes!, fit: BoxFit.contain)
                                                : Image.file(File(widget.photoPath), fit: BoxFit.contain),
                                  ),

                                  // On n'affiche la machine que si le chemin correspond à une machine de la catégorie sélectionnée (sécurité)
                                  if (_modeleSelectionneChemin != null && catalogueGlobal[_categorieSelectionnee]!.any((c) => c['chemin'] == _modeleSelectionneChemin))
                                    Positioned(
                                      left: climScreenX,
                                      top: climScreenY,
                                      width: climScreenW,
                                      height: climScreenH,
                                      // Rend la climatisation glissable par l'utilisateur.
                                      child: GestureDetector(
                                        behavior: HitTestBehavior.translucent, 
                                        onPanStart: (details) => setState(() => _isDragging = true),
                                        onPanUpdate: (details) {
                                           setState(() {
                                             _decalageX += details.delta.dx / scale;
                                             _decalageY += details.delta.dy / scale;
                                           });
                                        },
                                        onPanEnd: (details) {
                                           setState(() => _isDragging = false);
                                           _genererIncrustation(); 
                                        },
                                        child: Transform.rotate(
                                          angle: angleRad,
                                          alignment: Alignment.topLeft, 
                                          child: Opacity(
                                            opacity: _isDragging ? 0.65 : 0.0, 
                                            child: Image.asset(_modeleSelectionneChemin!, fit: BoxFit.fill),
                                          ),
                                        ),
                                      ),
                                    ),

                                  // Affichage de l'écran de chargement dynamique avec texte
                                  if (_isProcessing && !_isDragging)
                                    Positioned.fill(
                                      child: Container(
                                        color: Colors.black54, 
                                        child: Center(
                                          child: Column(
                                            mainAxisSize: MainAxisSize.min,
                                            children: [
                                              const CircularProgressIndicator(color: Colors.white),
                                              const SizedBox(height: 16),
                                              Text(
                                                _loadingMessage, 
                                                style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                                                textAlign: TextAlign.center,
                                              ),
                                            ],
                                          )
                                        )
                                      ),
                                    ),
                                ],
                              );
                            },
                          ),
                        ),
                    ),
                  ),
                  
                  // Bouton de réinitialisation flottant
                  if (_modeleSelectionneChemin != null && (_decalageX != 0.0 || _decalageY != 0.0))
                    Positioned(
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
                    ),
                ],
              ),
            ),
          ),

          // Section du bas restructurée avec Onglets et Liste des équipements
          if (_pointsCibles != null) 
            Container(
              height: 190, // Un peu plus haut pour laisser la place aux onglets
              padding: const EdgeInsets.only(top: 10, bottom: 10),
              decoration: BoxDecoration(
                color: Colors.white,
                boxShadow: [
                  BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10, offset: const Offset(0, -5))
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 1. La barre des catégories (Onglets)
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
                                setState(() {
                                  _categorieSelectionnee = catName;
                                });
                              }
                            },
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 15),
                  
                  // 2. La liste des machines OU le message "Bientôt disponible"
                  Expanded(
                    child: catalogueGlobal[_categorieSelectionnee]!.isEmpty
                        // Message si la catégorie est vide
                        ? Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.build_circle_outlined, size: 40, color: Colors.grey.shade400),
                                const SizedBox(height: 8),
                                Text(
                                  "Cette catégorie sera ajoutée prochainement",
                                  style: TextStyle(color: Colors.grey.shade600, fontStyle: FontStyle.italic, fontWeight: FontWeight.w500),
                                ),
                              ],
                            ),
                          )
                        // Liste de ton design actuel si la catégorie contient des machines
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
            ),
          const SizedBox(height: 80),
        ],
      ),
      
      // Bouton permettant de sauvegarder la composition finale une fois l'incrustation terminée.
      floatingActionButton: (_imageResultatBytes != null && !_isProcessing)
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