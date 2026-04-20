import 'dart:io';
import 'dart:math' as math; 
import 'package:flutter/foundation.dart'; 
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:image_gallery_saver_plus/image_gallery_saver_plus.dart'; 
import 'package:path_provider/path_provider.dart';

import 'traitement_image.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    print('Erreur caméra : ${e.code}, ${e.description}');
  }
  runApp(const MonApplication());
}

// ==========================================
// OPTIMISATION IMAGE (En Isolate pour ne pas avoir l'impression de freeze)
// ==========================================
Future<String?> _redimensionnerImageLourde(String imagePath) async {
  try {
    final imageBytes = await File(imagePath).readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);
    
    if (image == null) return null;

    // Si l'image est plus grande que du 1920, on la réduit
    int maxSize = 1920; 
    if (image.width > maxSize || image.height > maxSize) {
      print("[Optimisation] L'image est trop grande (${image.width}x${image.height}). Redimensionnement...");
      
      // On conserve les proportions (Aspect Ratio)
      img.Image resized;
      if (image.width > image.height) {
        resized = img.copyResize(image, width: maxSize);
      } else {
        resized = img.copyResize(image, height: maxSize);
      }

      // On la sauvegarde dans un dossier temporaire
      final directory = await getTemporaryDirectory();
      final path = '${directory.path}/photo_optimisee_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final newFile = File(path);
      
      // Encodage en JPEG (Qualité 90 pour garder les détails)
      await newFile.writeAsBytes(img.encodeJpg(resized, quality: 90));
      print("[Optimisation] Nouvelle taille ${resized.width}x${resized.height} prête !");
      return path;
    }
    
    // Si l'image est déjà petite, on ne touche à rien
    print("[Optimisation] Taille correcte (${image.width}x${image.height}), pas de changement.");
    return imagePath;
    
  } catch (e) {
    print("Erreur d'optimisation image : $e");
    return imagePath; // En cas de plantage, on rend l'originale pour ne pas bloquer l'appli
  }
}

// ==========================================
// FONCTION ISOLATE : DÉCODAGE IMAGE (Pour l'IA)
// ==========================================
Map<String, dynamic>? prepareImageMatrixForIA(Map<String, dynamic> params) {
  Uint8List imageBytes = params['bytes'];
  bool isNHWC = params['isNHWC'];

  img.Image? originalImage = img.decodeImage(imageBytes);
  if (originalImage == null) return null;

  int w = originalImage.width;
  int h = originalImage.height;

  img.Image resizedImage = img.copyResize(originalImage, width: 1024, height: 1024);

  List<dynamic> inputMatrix;
  if (isNHWC) {
    inputMatrix = List.generate(1, (i) => List.generate(1024, (j) => List.generate(1024, (k) => List.generate(3, (l) => 0.0))));
    for (int y = 0; y < 1024; y++) {
      for (int x = 0; x < 1024; x++) {
        final pixel = resizedImage.getPixel(x, y);
        inputMatrix[0][y][x][0] = pixel.r / 255.0; 
        inputMatrix[0][y][x][1] = pixel.g / 255.0; 
        inputMatrix[0][y][x][2] = pixel.b / 255.0; 
      }
    }
  } else {
    inputMatrix = List.generate(1, (i) => List.generate(3, (j) => List.generate(1024, (k) => List.generate(1024, (l) => 0.0))));
    for (int y = 0; y < 1024; y++) {
      for (int x = 0; x < 1024; x++) {
        final pixel = resizedImage.getPixel(x, y);
        inputMatrix[0][0][y][x] = pixel.r / 255.0; 
        inputMatrix[0][1][y][x] = pixel.g / 255.0; 
        inputMatrix[0][2][y][x] = pixel.b / 255.0; 
      }
    }
  }

  return {
    'width': w,
    'height': h,
    'matrix': inputMatrix
  };
}

class MonApplication extends StatelessWidget {
  const MonApplication({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Simulateur Clim',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const EcranAccueil(),
    );
  }
}

// ==========================================
// ÉCRAN 1 : LA CAMERA ET LA GALERIE
// ==========================================
class EcranAccueil extends StatefulWidget {
  const EcranAccueil({super.key});
  @override
  State<EcranAccueil> createState() => _EcranAccueilState();
}

class _EcranAccueilState extends State<EcranAccueil> {
  CameraController? _controller;
  final ImagePicker _picker = ImagePicker();
  bool _isOptimizing = false; // Pour afficher un petit loader pendant la compression

  @override
  void initState() {
    super.initState();
    if (cameras.isNotEmpty) {
      _controller = CameraController(
        cameras[0],
        ResolutionPreset.high,
        enableAudio: false,
      );
      _controller!.initialize().then((_) {
        if (!mounted) return;
        setState(() {});
      }).catchError((Object e) {
        print("Erreur initialisation caméra : $e");
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _prendrePhoto() async {
    if (_controller != null && _controller!.value.isInitialized) {
      try {
        setState(() => _isOptimizing = true);
        final rawImage = await _controller!.takePicture();
        
        // On optimise la photo en arrière-plan
        String? optimizedPath = await compute(_redimensionnerImageLourde, rawImage.path);
        
        if (!mounted) return;
        setState(() => _isOptimizing = false);
        _allerVersResultat(optimizedPath ?? rawImage.path);

      } catch (e) {
        print("Erreur appareil photo : $e");
        setState(() => _isOptimizing = false);
      }
    }
  }

  Future<void> _ouvrirGalerie() async {
    try {
      final XFile? rawImage = await _picker.pickImage(source: ImageSource.gallery);
      if (rawImage != null && mounted) {
        setState(() => _isOptimizing = true);
        
        // On optimise la photo de la galerie en arrière-plan
        String? optimizedPath = await compute(_redimensionnerImageLourde, rawImage.path);
        
        if (!mounted) return;
        setState(() => _isOptimizing = false);
        _allerVersResultat(optimizedPath ?? rawImage.path);
      }
    } catch (e) {
      print("Erreur galerie : $e");
      setState(() => _isOptimizing = false);
    }
  }

  void _allerVersResultat(String imagePath) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => EcranResultat(photoPath: imagePath), // On passe juste un String
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Choisir le mur'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Stack(
        children: [
          Column(
            children: [
              Expanded(
                child: Container(
                  width: double.infinity,
                  margin: const EdgeInsets.all(16.0),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(15),
                    child: _controller != null && _controller!.value.isInitialized
                        ? SizedBox(
                            width: double.infinity,
                            height: double.infinity,
                            child: FittedBox(
                              fit: BoxFit.cover,
                              child: SizedBox(
                                width: _controller!.value.previewSize?.height ?? 1,
                                height: _controller!.value.previewSize?.width ?? 1,
                                child: CameraPreview(_controller!),
                              ),
                            ),
                          )
                        : const Center(child: CircularProgressIndicator()),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(bottom: 40.0, left: 16.0, right: 16.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _isOptimizing ? null : _ouvrirGalerie,
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Galerie'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                        backgroundColor: Colors.white,
                        foregroundColor: Colors.teal,
                      ),
                    ),
                    ElevatedButton.icon(
                      onPressed: _isOptimizing ? null : _prendrePhoto,
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Photo'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                        backgroundColor: Theme.of(context).colorScheme.primary,
                        foregroundColor: Theme.of(context).colorScheme.onPrimary,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          
          // Un léger voile noir avec un texte rassurant pendant que le téléphone redimensionne la grosse photo
          if (_isOptimizing)
            Container(
              color: Colors.black54,
              child: const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(color: Colors.white),
                    SizedBox(height: 15),
                    Text("Préparation de l'image...", style: TextStyle(color: Colors.white, fontSize: 16)),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ==========================================
// ÉCRAN 2 : LE RÉSULTAT, L'IA ET OPENCV
// ==========================================
class EcranResultat extends StatefulWidget {
  final String photoPath; // Changé de XFile à String pour correspondre à notre chemin optimisé
  const EcranResultat({super.key, required this.photoPath});

  @override
  State<EcranResultat> createState() => _EcranResultatState();
}

class _EcranResultatState extends State<EcranResultat> {
  final List<Map<String, String>> catalogueClims = [
    {
      'nom': 'Takao Plus Blanc',
      'chemin': 'assets/installations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png'
    },
    {
      'nom': 'Takao Plus Noir',
      'chemin': 'assets/installations/clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png'
    }
  ];
  
  String? _modeleSelectionneChemin;

  bool _isProcessing = true;
  Interpreter? _iaModel;
  
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

  Future<void> _lancerProcessusAutomatique() async {
    try {
      _iaModel = await Interpreter.fromAsset('assets/best.tflite');
      await _analyserImage();
    } catch (e) {
      print("[IA - ERREUR FATALE] Échec au chargement du modèle : $e");
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _analyserImage() async {
    if (_iaModel == null) return;

    try {
      print("\n=======================================================");
      print("[IA] DÉBUT DE L'ANALYSE TFLITE");
      print("=======================================================");

      final imageBytes = await File(widget.photoPath).readAsBytes();
      var inputShape = _iaModel!.getInputTensor(0).shape;
      bool isNHWC = inputShape[3] == 3;
      
      print("[IA] Préparation de la matrice (IsNHWC: $isNHWC)...");
      final resultMatrixPrep = await compute(prepareImageMatrixForIA, {
        'bytes': imageBytes,
        'isNHWC': isNHWC
      });

      if (resultMatrixPrep == null) throw Exception("Impossible de lire l'image.");

      _imageWidth = resultMatrixPrep['width'];
      _imageHeight = resultMatrixPrep['height'];
      print("[IA] Image décodée. Dimensions originales: ${_imageWidth}x$_imageHeight");
      
      var inputMatrix = resultMatrixPrep['matrix'];

      var outputShape = _iaModel!.getOutputTensor(0).shape;
      var outputMatrix = List.generate(outputShape[0], (i) =>
        List.generate(outputShape[1], (j) =>
          List.generate(outputShape[2], (k) => 0.0)
        )
      );

      print("[IA] Exécution du modèle (Run)...");
      _iaModel!.run(inputMatrix, outputMatrix);

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

      print("[IA] Meilleure détection trouvée - Confiance Boîte: ${(maxConfiance * 100).toStringAsFixed(2)}% (Index: $meilleurIndex)");

      if (maxConfiance > 0.5) {
        double boxX = isTransposed ? outputMatrix[0][meilleurIndex][0] : outputMatrix[0][0][meilleurIndex];
        double boxY = isTransposed ? outputMatrix[0][meilleurIndex][1] : outputMatrix[0][1][meilleurIndex];
        double boxW = isTransposed ? outputMatrix[0][meilleurIndex][2] : outputMatrix[0][2][meilleurIndex];
        double boxH = isTransposed ? outputMatrix[0][meilleurIndex][3] : outputMatrix[0][3][meilleurIndex];
        
        double scale = (boxW < 2.0 && boxH < 2.0) ? 1024.0 : 1.0;
        print("[IA] Bounding Box (avant scale) : X=${boxX.toStringAsFixed(2)}, Y=${boxY.toStringAsFixed(2)}, W=${boxW.toStringAsFixed(2)}, H=${boxH.toStringAsFixed(2)}");
        print("[IA] Facteur de Scale appliqué : $scale");

        List<Map<String, double>> rawPoints = [];
        double confMoyennePoints = 0;

        print("[IA] Extraction des 4 Keypoints (Pose) :");
        for(int point = 0; point < 4; point++) {
           int idxX = 5 + (point * 3);
           int idxY = idxX + 1;
           int idxConf = idxX + 2;

           double px = isTransposed ? outputMatrix[0][meilleurIndex][idxX] : outputMatrix[0][idxX][meilleurIndex];
           double py = isTransposed ? outputMatrix[0][meilleurIndex][idxY] : outputMatrix[0][idxY][meilleurIndex];
           double pConf = isTransposed ? outputMatrix[0][meilleurIndex][idxConf] : outputMatrix[0][idxConf][meilleurIndex];
           
           confMoyennePoints += pConf;
           rawPoints.add({'x': px * scale, 'y': py * scale});
           
           print("  -> Point ${point + 1} : X=${(px * scale).toStringAsFixed(1)}, Y=${(py * scale).toStringAsFixed(1)} | Confiance: ${(pConf * 100).toStringAsFixed(1)}%");
        }

        confMoyennePoints = confMoyennePoints / 4.0;
        print("[IA] Confiance moyenne des 4 points : ${(confMoyennePoints * 100).toStringAsFixed(2)}%");

        if (confMoyennePoints >= 0.92) {
          print("[IA] VALIDATION: Confiance > 92%. Utilisation des points de l'IA.");
          _pointsCibles = TraitementImage.trierPoints(rawPoints);
        } else {
          print("[IA] FALLBACK: Confiance < 92%. Utilisation des coins de la Bounding Box.");
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

        print("[IA] Points finaux cibles pour OpenCV : $_pointsCibles");

        if (_pointsCibles != null) {
           print("[OpenCV] Envoi vers Isolate pour effacer l'autocollant (Fond Propre)...");
           _imageFondPropreBytes = await compute(TraitementImage.effacerAutocollantIsolate, {
             'photoPath': widget.photoPath,
             'pointsIA': _pointsCibles!,
           });
           print("[OpenCV] Fond Propre généré avec succès.");
        }

        setState(() {
          _isProcessing = false;
        });

      } else {
        print("[IA] ÉCHEC: Aucune boîte détectée avec confiance > 50%. (Max=$maxConfiance)");
        setState(() {
          _pointsCibles = null;
          _isProcessing = false;
        });
      }

    } catch (e) {
      print("[IA - ERREUR] Exception durant l'analyse : $e");
      setState(() {
        _pointsCibles = null;
        _isProcessing = false;
      });
    } 
  }

  Future<void> _genererIncrustation() async {
    if (_pointsCibles == null || _modeleSelectionneChemin == null) return;
    
    setState(() => _isProcessing = true);

    try {
      print("\n=======================================================");
      print("[UI/OpenCV] LANCEMENT DE L'INCRUSTATION TOTALE");
      print("=======================================================");
      String climPath = _modeleSelectionneChemin!;
      
      print("[UI/OpenCV] Modèle sélectionné : $climPath");
      print("[UI/OpenCV] Décalage utilisateur : X=${_decalageX.toStringAsFixed(2)}, Y=${_decalageY.toStringAsFixed(2)}");
      
      final ByteData data = await DefaultAssetBundle.of(context).load(climPath);
      Uint8List climBytes = data.buffer.asUint8List();

      print("[UI/OpenCV] Envoi des données vers TraitementImage.incrusterClimatisationIsolate...");
      Uint8List? resultImage = await compute(TraitementImage.incrusterClimatisationIsolate, {
        'photoPath': widget.photoPath,
        'climBytes': climBytes,
        'pointsIA': _pointsCibles!,
        'decalageX': _decalageX,
        'decalageY': _decalageY,
        'climAssetPath': climPath,
      });

      if (resultImage != null) {
        print("[UI/OpenCV] Image finale générée et reçue avec succès.");
        setState(() {
          _imageResultatBytes = resultImage;
        });
      } else {
        print("[UI/OpenCV - ERREUR] L'isolate a retourné une image nulle.");
      }
    } catch (e) {
      print("[UI/OpenCV - ERREUR] Exception pendant l'incrustation : $e");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _sauvegarderImage() async {
    if (_imageResultatBytes == null) return;

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('⏳ Sauvegarde en cours...'), duration: Duration(milliseconds: 500)),
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
            content: Text('✅ Simulation sauvegardée dans la galerie !'),
            backgroundColor: Colors.green,
            duration: Duration(seconds: 3),
          ),
        );
      } else {
        throw Exception("Échec de la sauvegarde interne.");
      }
    } catch (e) {
      print("Erreur de sauvegarde : $e");

      if (!mounted) return; 

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('❌ Erreur lors de la sauvegarde.'),
          backgroundColor: Colors.red,
        ),
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
              child: ClipRRect(
                borderRadius: BorderRadius.circular(15),
                child: InteractiveViewer(
                    panEnabled: true,
                    scaleEnabled: true,
                    minScale: 1.0,
                    maxScale: 8.0,
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        
                        if (_imageWidth == null || _imageHeight == null) {
                           return _isProcessing 
                              ? const Center(child: CircularProgressIndicator()) 
                              : Image.file(File(widget.photoPath), fit: BoxFit.contain);
                        }

                        if (_pointsCibles == null) {
                           return Stack(
                             children: [
                               Positioned.fill(
                                 child: Image.file(File(widget.photoPath), fit: BoxFit.contain),
                               ),
                               if (!_isProcessing)
                                 Center(
                                   child: Container(
                                     padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                                     decoration: BoxDecoration(
                                       color: Colors.teal.withValues(alpha: 0.8),
                                       borderRadius: BorderRadius.circular(12),
                                     ),
                                     child: const Column(
                                       mainAxisSize: MainAxisSize.min,
                                       children: [
                                         Icon(Icons.error_outline, color: Colors.redAccent, size: 40),
                                         SizedBox(height: 10),
                                         Text(
                                           "Aucun autocollant détecté sur la photo",
                                           style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                                           textAlign: TextAlign.center,
                                         ),
                                       ],
                                     ),
                                   ),
                                 ),
                               if (_isProcessing)
                                 const Center(child: CircularProgressIndicator()),
                             ],
                           );
                        }

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
                              child: _modeleSelectionneChemin == null
                                  ? Image.file(File(widget.photoPath), fit: BoxFit.contain)
                                  : (_isDragging && _imageFondPropreBytes != null)
                                      ? Image.memory(_imageFondPropreBytes!, fit: BoxFit.contain)
                                      : (_imageResultatBytes != null
                                          ? Image.memory(_imageResultatBytes!, fit: BoxFit.contain)
                                          : Image.file(File(widget.photoPath), fit: BoxFit.contain)),
                            ),

                            if (_modeleSelectionneChemin != null)
                              Positioned(
                                left: climScreenX,
                                top: climScreenY,
                                width: climScreenW,
                                height: climScreenH,
                                child: GestureDetector(
                                  behavior: HitTestBehavior.translucent, 
                                  onPanStart: (details) {
                                     setState(() => _isDragging = true);
                                  },
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

                            if (_isProcessing && !_isDragging && _modeleSelectionneChemin != null)
                              Positioned.fill(
                                child: Container(
                                  color: Colors.black38,
                                  child: const Center(
                                    child: CircularProgressIndicator(color: Colors.white),
                                  ),
                                ),
                              ),
                          ],
                        );
                      },
                    ),
                  ),
              ),
            ),
          ),

          if (_pointsCibles != null) 
            Container(
              height: 140,
              padding: const EdgeInsets.symmetric(vertical: 10),
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: catalogueClims.length,
                itemBuilder: (context, index) {
                  final clim = catalogueClims[index];
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
                      margin: EdgeInsets.only(left: 16.0, right: index == catalogueClims.length - 1 ? 16.0 : 0.0),
                      decoration: BoxDecoration(
                        color: isSelected ? Colors.teal.withValues(alpha : 0.1) : Colors.white,
                        border: Border.all(
                          color: isSelected ? Colors.teal : Colors.grey.shade300,
                          width: isSelected ? 3 : 1,
                        ),
                        borderRadius: BorderRadius.circular(15),
                        boxShadow: [
                          if (isSelected)
                            BoxShadow(color: Colors.teal.withValues(alpha : 0.2), blurRadius: 8, offset: const Offset(0, 4))
                        ],
                      ),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Expanded(
                            child: Padding(
                              padding: const EdgeInsets.all(8.0),
                              child: Image.asset(clim['chemin']!, fit: BoxFit.contain),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 4.0, vertical: 8.0),
                            child: Text(
                              clim['nom']!,
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                                color: isSelected ? Colors.teal.shade800 : Colors.black87,
                              ),
                              textAlign: TextAlign.center,
                              maxLines: 2,
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
            
          const SizedBox(height: 80),
        ],
      ),
      
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