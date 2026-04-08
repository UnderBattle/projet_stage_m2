import 'dart:io';
import 'dart:typed_data'; // NOUVEAU : Pour afficher l'image en mémoire
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

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
// ÉCRAN 2 : LE RÉSULTAT ET L'IA
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
  
  // NOUVEAU : On stocke l'image modifiée avec les dessins de l'IA
  Uint8List? _imageDebugBytes; 

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
      print("[IA] Chargement du modèle best.tflite...");
      _iaModel = await Interpreter.fromAsset('assets/best.tflite');
      print("[IA] Cerveau chargé avec succès ! Prêt à analyser.");
    } catch (e) {
      print("[IA] ERREUR FATALE : Impossible de charger l'IA : $e");
    }
  }

  Future<void> _lancerAnalyseIA() async {
    if (_iaModel == null) {
      print("[IA] Le modèle n'est pas encore prêt.");
      return;
    }

    setState(() {
      _isProcessing = true;
      _imageDebugBytes = null; // On réinitialise le dessin
    });

    try {
      print("\n[IA] === DÉBUT DE L'ANALYSE ===");

      final imageBytes = await File(widget.photo.path).readAsBytes();
      img.Image? originalImage = img.decodeImage(imageBytes);

      if (originalImage == null) throw Exception("Impossible de lire l'image.");

      print("[IA] Formatage de l'image (1024x1024)...");
      img.Image resizedImage = img.copyResize(originalImage, width: 1024, height: 1024);

      var inputShape = _iaModel!.getInputTensor(0).shape;
      bool isNHWC = inputShape[3] == 3; 
      
      var inputMatrix;
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

      print("[IA] Le cerveau réfléchit...");
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

      print("\n[RÉSULTAT IA]");
      if (maxConfiance > 0.5) { 
        print("Autocollant détecté avec ${(maxConfiance * 100).toStringAsFixed(1)}% de certitude !");
        
        double boxX = isTransposed ? outputMatrix[0][meilleurIndex][0] : outputMatrix[0][0][meilleurIndex];
        double boxY = isTransposed ? outputMatrix[0][meilleurIndex][1] : outputMatrix[0][1][meilleurIndex];
        double boxW = isTransposed ? outputMatrix[0][meilleurIndex][2] : outputMatrix[0][2][meilleurIndex];
        double boxH = isTransposed ? outputMatrix[0][meilleurIndex][3] : outputMatrix[0][3][meilleurIndex];
        
        double scale = (boxW < 2.0 && boxH < 2.0) ? 1024.0 : 1.0;

        print("Boîte brute : X:${(boxX * scale).toStringAsFixed(1)}, Y:${(boxY * scale).toStringAsFixed(1)} | W:${(boxW * scale).toStringAsFixed(1)}, H:${(boxH * scale).toStringAsFixed(1)}");
        
        // Calcul des coins de la Bounding Box
        int x1 = ((boxX * scale) - (boxW * scale) / 2).toInt();
        int y1 = ((boxY * scale) - (boxH * scale) / 2).toInt();
        int x2 = ((boxX * scale) + (boxW * scale) / 2).toInt();
        int y2 = ((boxY * scale) + (boxH * scale) / 2).toInt();

        // Dessine le rectangle Rouge sur l'image
        img.drawRect(
          resizedImage, 
          x1: x1, y1: y1, x2: x2, y2: y2, 
          color: img.ColorRgb8(255, 0, 0), // Rouge
          thickness: 5
        );

        print("Points Clés sur l'image 1024x1024 :");
        for(int point = 0; point < 4; point++) {
           int idxX = 5 + (point * 3);
           int idxY = idxX + 1;
           int idxConf = idxX + 2;

           double px = isTransposed ? outputMatrix[0][meilleurIndex][idxX] : outputMatrix[0][idxX][meilleurIndex];
           double py = isTransposed ? outputMatrix[0][meilleurIndex][idxY] : outputMatrix[0][idxY][meilleurIndex];
           double pConf = isTransposed ? outputMatrix[0][meilleurIndex][idxConf] : outputMatrix[0][idxConf][meilleurIndex];
           
           print(" -> Point ${point+1} : X=${(px * scale).toStringAsFixed(1)}, Y=${(py * scale).toStringAsFixed(1)} (Confiance: ${(pConf*100).toStringAsFixed(1)}%)");

           // Dessine un point Vert pour chaque coin trouvé
           img.fillCircle(
             resizedImage, 
             x: (px * scale).toInt(), 
             y: (py * scale).toInt(), 
             radius: 12, 
             color: img.ColorRgb8(0, 255, 0) // Vert
           );
        }

        // Enregistre l'image modifiée pour l'afficher dans l'interface
        _imageDebugBytes = img.encodeJpg(resizedImage);

      } else {
        print("Aucun autocollant trouvé (Meilleur score : ${(maxConfiance * 100).toStringAsFixed(1)}%)");
      }

    } catch (e) {
      print("[IA] ERREUR pendant l'analyse : $e");
    } finally {
      setState(() {
        _isProcessing = false;
      });
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
                // NOUVEAU : Logique d'affichage
                child: _isProcessing 
                ? const Center(child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 20),
                      Text("L'IA analyse l'image...")
                    ],
                ))
                : (_imageDebugBytes != null) 
                    // Affiche l'image de Débug avec les points et la boîte
                    ? Image.memory(_imageDebugBytes!, fit: BoxFit.contain)
                    // Sinon affiche l'image originale
                    : (kIsWeb
                        ? Image.network(widget.photo.path, fit: BoxFit.cover)
                        : Image.file(File(widget.photo.path), fit: BoxFit.cover)),
              ),
            ),
          ),
          const SizedBox(height: 80),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isProcessing ? null : _lancerAnalyseIA,
        label: const Text("Chercher l'autocollant"),
        icon: const Icon(Icons.search),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}