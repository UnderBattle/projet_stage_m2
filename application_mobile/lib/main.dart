import 'dart:io';
import 'dart:typed_data'; 
import 'dart:math' as math; 
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

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
        final image = await _controller!.takePicture();
        if (!mounted) return;
        _allerVersResultat(image);
      } catch (e) {
        print("Erreur appareil photo : $e");
      }
    }
  }

  Future<void> _ouvrirGalerie() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null && mounted) {
        _allerVersResultat(image);
      }
    } catch (e) {
      print("Erreur galerie : $e");
    }
  }

  void _allerVersResultat(XFile image) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => EcranResultat(photo: image),
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
      body: Column(
        children: [
          Expanded(
            child: Container(
              width: double.infinity,
              margin: const EdgeInsets.all(16.0),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(15),
                child: _controller != null && _controller!.value.isInitialized
                    ? CameraPreview(_controller!)
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
                  onPressed: _ouvrirGalerie,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Galerie'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                    backgroundColor: Colors.white,
                    foregroundColor: Colors.teal,
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: _prendrePhoto,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Photo'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                    backgroundColor: Theme.of(context).colorScheme.primary,
                    foregroundColor: Theme.of(context).colorScheme.onPrimary,
                  ),
                ),
              ],
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
  final XFile photo;
  const EcranResultat({super.key, required this.photo});

  @override
  State<EcranResultat> createState() => _EcranResultatState();
}

class _EcranResultatState extends State<EcranResultat> {
  String modeleSelectionne = 'Takao Plus Blanc';
  final List<String> catalogueClims = ['Takao Plus Blanc', 'Takao Plus Noir'];

  bool _isProcessing = false;
  Interpreter? _iaModel;
  Uint8List? _imageResultatBytes; 

  int? _imageWidth;
  int? _imageHeight;
  List<Map<String, double>>? _pointsCibles; 
  double _decalageX = 0.0; 
  double _decalageY = 0.0; 
  bool _isDragging = false; 

  @override
  void initState() {
    super.initState();
    _chargerModeleIA();
  }

  @override
  void dispose() {
    _iaModel?.close(); 
    super.dispose();
  }

  Future<void> _chargerModeleIA() async {
    try {
      _iaModel = await Interpreter.fromAsset('assets/best.tflite');
    } catch (e) {
      print("[IA] ERREUR FATALE : $e");
    }
  }

  Future<void> _analyserImage() async {
    if (_iaModel == null) return;

    setState(() {
      _isProcessing = true;
      _imageResultatBytes = null; 
      _pointsCibles = null;
      _decalageX = 0.0;
      _decalageY = 0.0;
    });

    try {
      print("\n[IA] === DÉBUT DE L'ANALYSE ===");

      final imageBytes = await File(widget.photo.path).readAsBytes();
      img.Image? originalImage = img.decodeImage(imageBytes);
      if (originalImage == null) throw Exception("Impossible de lire l'image.");

      _imageWidth = originalImage.width;
      _imageHeight = originalImage.height;

      img.Image resizedImage = img.copyResize(originalImage, width: 1024, height: 1024);

      var inputShape = _iaModel!.getInputTensor(0).shape;
      bool isNHWC = inputShape[3] == 3; 
      
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

      var outputShape = _iaModel!.getOutputTensor(0).shape;
      var outputMatrix = List.generate(outputShape[0], (i) => 
        List.generate(outputShape[1], (j) => 
          List.generate(outputShape[2], (k) => 0.0)
        )
      );

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

      if (maxConfiance > 0.5) { 
        print("✅ IA : Autocollant détecté à ${(maxConfiance * 100).toStringAsFixed(1)}%");
        
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

        if (confMoyennePoints >= 0.8) {
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

        await _genererIncrustation();

      } else {
        print("❌ IA : Aucun autocollant trouvé !");
        setState(() => _isProcessing = false);
      }

    } catch (e) {
      print("[IA] ERREUR : $e");
      setState(() => _isProcessing = false);
    } 
  }

  Future<void> _genererIncrustation() async {
    if (_pointsCibles == null) return;
    
    setState(() => _isProcessing = true);

    try {
      String climPath = modeleSelectionne == 'Takao Plus Blanc'
          ? 'assets/installations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png'
          : 'assets/installations/clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png';

      Uint8List? resultImage = await TraitementImage.incrusterClimatisation(
        photoPath: widget.photo.path,
        climAssetPath: climPath,
        pointsIA: _pointsCibles!,
        decalageX: _decalageX, 
        decalageY: _decalageY, 
      );

      if (resultImage != null) {
        setState(() {
          _imageResultatBytes = resultImage;
        });
      }
    } catch (e) {
      print("[Incrustation] ERREUR : $e");
    } finally {
      setState(() => _isProcessing = false);
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
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text('Modèle : ', style: TextStyle(fontSize: 18)),
                const SizedBox(width: 10),
                DropdownButton<String>(
                  value: modeleSelectionne,
                  icon: const Icon(Icons.arrow_downward),
                  elevation: 16,
                  style: const TextStyle(color: Colors.teal, fontSize: 18, fontWeight: FontWeight.bold),
                  underline: Container(height: 2, color: Colors.tealAccent),
                  onChanged: (String? nouveauChoix) {
                    setState(() {
                      modeleSelectionne = nouveauChoix!;
                      if (_pointsCibles != null) {
                         _genererIncrustation();
                      }
                    });
                  },
                  items: catalogueClims.map<DropdownMenuItem<String>>((String modele) {
                    return DropdownMenuItem<String>(value: modele, child: Text(modele));
                  }).toList(),
                ),
              ],
            ),
          ),

          Expanded(
            child: Container(
              width: double.infinity,
              margin: const EdgeInsets.all(16.0),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(15),
                // L'INTERACTIVE VIEWER EST TOUJOURS ACTIF POUR LE ZOOM ET LE PAN !
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
                              : Image.file(File(widget.photo.path), fit: BoxFit.contain);
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

                        String climPath = modeleSelectionne == 'Takao Plus Blanc'
                          ? 'assets/installations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png'
                          : 'assets/installations/clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png';

                        return Stack(
                          children: [
                            // 1. LE FOND
                            Positioned.fill(
                              child: _isDragging || _imageResultatBytes == null
                                  ? Image.file(File(widget.photo.path), fit: BoxFit.contain)
                                  : Image.memory(_imageResultatBytes!, fit: BoxFit.contain),
                            ),

                            // 2. LA CLIM "FANTÔME" & DÉTECTEUR DE TOUCHER
                            if (_pointsCibles != null)
                              Positioned(
                                left: climScreenX,
                                top: climScreenY,
                                width: climScreenW,
                                height: climScreenH,
                                // Ce GestureDetector n'intercepte QUE si on touche la clim !
                                child: GestureDetector(
                                  behavior: HitTestBehavior.translucent, // Pour capter même si transparent
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
                                      // Transparent (0.0) au repos, 0.65 quand on déplace
                                      opacity: _isDragging ? 0.65 : 0.0, 
                                      child: Image.asset(climPath, fit: BoxFit.fill),
                                    ),
                                  ),
                                ),
                              ),

                            // 3. CHARGEMENT
                            if (_isProcessing && !_isDragging && _imageResultatBytes != null)
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
          const SizedBox(height: 80),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isProcessing ? null : _analyserImage,
        label: const Text("Analyser l'image"),
        icon: const Icon(Icons.auto_fix_high),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}