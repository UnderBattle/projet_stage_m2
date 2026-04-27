import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import '../utils/image_utils.dart';
import '../services/ia_service.dart';
import 'ecran_resultat.dart';

/// Écran principal permettant à l'utilisateur de prendre une photo du mur ou d'en choisir une dans la galerie.
class EcranAccueil extends StatefulWidget {
  final List<CameraDescription> cameras;
  
  const EcranAccueil({super.key, required this.cameras});

  @override
  State<EcranAccueil> createState() => _EcranAccueilState();
}

class _EcranAccueilState extends State<EcranAccueil> {
  CameraController? _controller;
  final ImagePicker _picker = ImagePicker();
  
  // Indique si une image est en cours de redimensionnement pour afficher l'écran de chargement.
  bool _isOptimizing = false;

  @override
  void initState() {
    super.initState();
    
    // On lance le chargement de l'IA en tâche de fond
    IAService().initModels();

    // Initialise le contrôleur avec la première caméra disponible (généralement la caméra arrière) en haute résolution.
    if (widget.cameras.isNotEmpty) {
      _controller = CameraController(
        widget.cameras[0],
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
    // Libère les ressources de la caméra lorsque l'écran est détruit pour éviter les fuites de mémoire.
    _controller?.dispose();
    super.dispose();
  }

  /// Déclenche la prise de vue avec la caméra, optimise l'image si elle est trop lourde, puis navigue vers le résultat.
  Future<void> _prendrePhoto() async {
    if (_controller != null && _controller!.value.isInitialized) {
      try {
        setState(() => _isOptimizing = true);
        
        // Si le client a été plus vite que le chargement, on patiente pour sécuriser l'IA
        if (!IAService().isInitialized) {
          await IAService().initModels();
        }

        final rawImage = await _controller!.takePicture();
        
        // Lance le redimensionnement dans un Isolate pour éviter que l'interface ne gèle.
        String? optimizedPath = await compute(redimensionnerImageLourde, rawImage.path);
        
        if (!mounted) return;
        setState(() => _isOptimizing = false);
        _allerVersResultat(optimizedPath ?? rawImage.path);

      } catch (e) {
        print("Erreur appareil photo : $e");
        setState(() => _isOptimizing = false);
      }
    }
  }

  /// Ouvre la galerie photo, récupère l'image sélectionnée, l'optimise et navigue vers le résultat.
  Future<void> _ouvrirGalerie() async {
    try {
      final XFile? rawImage = await _picker.pickImage(source: ImageSource.gallery);
      if (rawImage != null && mounted) {
        setState(() => _isOptimizing = true);
        
        // Sécurité pour le chargement de l'IA
        if (!IAService().isInitialized) {
          await IAService().initModels();
        }
        
        // Même optimisation que pour l'appareil photo via un Isolate.
        String? optimizedPath = await compute(redimensionnerImageLourde, rawImage.path);
        
        if (!mounted) return;
        setState(() => _isOptimizing = false);
        _allerVersResultat(optimizedPath ?? rawImage.path);
      }
    } catch (e) {
      print("Erreur galerie : $e");
      setState(() => _isOptimizing = false);
    }
  }

  /// Navigue vers l'écran de résultat en lui passant le chemin de l'image finale.
  void _allerVersResultat(String imagePath) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => EcranResultat(photoPath: imagePath),
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
                        // Affiche le retour vidéo de la caméra si elle est prête.
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
                // Boutons d'actions pour choisir la source de l'image.
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
          // Affiche un indicateur de chargement par-dessus l'interface pendant l'optimisation de l'image.
          if (_isOptimizing)
            Container(
              color: Colors.black54,
              child: const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(color: Colors.white),
                    SizedBox(height: 15),
                    Text("Préparation...", style: TextStyle(color: Colors.white, fontSize: 16)),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}