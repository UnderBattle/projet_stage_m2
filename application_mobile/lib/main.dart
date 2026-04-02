import 'dart:io';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

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
// ÉCRAN 1 : LA CAMERA
// ==========================================
class EcranAccueil extends StatefulWidget {
  const EcranAccueil({super.key});
  @override
  State<EcranAccueil> createState() => _EcranAccueilState();
}

class _EcranAccueilState extends State<EcranAccueil> {
  CameraController? _controller;

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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Prendre le mur en photo'),
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
          const SizedBox(height: 80), 
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () async {
          if (_controller != null && _controller!.value.isInitialized) {
             try {
               final image = await _controller!.takePicture();
               if (!context.mounted) return; 

               // On ne passe plus que l'image à l'écran suivant !
               Navigator.push(
                 context,
                 MaterialPageRoute(
                   builder: (context) => EcranResultat(photo: image),
                 ),
               );
             } catch (e) {
               print("Erreur : $e");
             }
          }
        },
        label: const Text('Prendre en photo'),
        icon: const Icon(Icons.camera),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}

// ==========================================
// ÉCRAN 2 : LE RÉSULTAT
// ==========================================
class EcranResultat extends StatefulWidget {
  final XFile photo;

  // On demande uniquement la photo à la création de l'écran
  const EcranResultat({super.key, required this.photo});

  @override
  State<EcranResultat> createState() => _EcranResultatState();
}

class _EcranResultatState extends State<EcranResultat> {
  // C'est maintenant cet écran qui gère le catalogue et le choix
  String modeleSelectionne = 'Takao Plus Blanc';
  final List<String> catalogueClims = ['Takao Plus Blanc', 'Takao Plus Noir'];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Configuration du Devis'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          // Menu déroulant du catalogue
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

          // La photo prise
          Expanded(
            child: Container(
              width: double.infinity,
              margin: const EdgeInsets.all(16.0),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(15),
                // Astuce : comme "photo" appartient au Widget parent et pas au State, 
                // on doit utiliser "widget.photo.path" pour y accéder.
                child: kIsWeb 
                    ? Image.network(widget.photo.path, fit: BoxFit.cover) 
                    : Image.file(File(widget.photo.path), fit: BoxFit.cover),
              ),
            ),
          ),
          const SizedBox(height: 80),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          // C'est ici que l'on appellera l'IA et le script OpenCV
          // TODO : intégrer l'IA et OpenCV pour générer la simulation
          print("Lancement de l'IA pour le modèle : $modeleSelectionne");
        },
        label: const Text('Générer la simulation'),
        icon: const Icon(Icons.auto_fix_high), 
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}