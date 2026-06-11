import 'dart:async';
import 'dart:ui'; 
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:sensors_plus/sensors_plus.dart';
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

// OPTIMISATION : Ajout de WidgetsBindingObserver pour écouter les mises en pause de l'application (ex: bouton Home)
class _EcranAccueilState extends State<EcranAccueil> with WidgetsBindingObserver {
  CameraController? _controller;
  final ImagePicker _picker = ImagePicker();
  
  /// Indique si une image est en cours de redimensionnement pour afficher l'écran de chargement.
  bool _isOptimizing = false;
  
  /// Indique si les modèles d'IA sont prêts à être utilisés.
  bool _isIaReady = false;

  // =========================================================================
  // === VARIABLES POUR LE NIVEAU À BULLE ===
  // =========================================================================
  StreamSubscription<AccelerometerEvent>? _accelSubscription;
  // Utilisation d'un ValueNotifier pour mettre à jour la bulle à 60fps sans faire ramer la caméra
  final ValueNotifier<AccelerometerEvent?> _accelerometerNotifier = ValueNotifier(null);

  @override
  void initState() {
    super.initState();
    
    // Enregistrement de l'observateur de cycle de vie système
    WidgetsBinding.instance.addObserver(this);
    
    // Lance le chargement des modèles d'IA en arrière-plan.
    _preparerIA();

    // Initialise le contrôleur avec la première caméra disponible (généralement la caméra arrière) en haute résolution.
    _initialiserCamera();

    // Démarre l'écoute des capteurs d'inclinaison du téléphone
    _demarrerEcouteAccelerometre();
  }

  /// Initialise le flux d'écoute de l'accéléromètre de manière isolée
  void _demarrerEcouteAccelerometre() {
    if (_accelSubscription == null) {
      _accelSubscription = accelerometerEventStream().listen((AccelerometerEvent event) {
        if (mounted) {
          _accelerometerNotifier.value = event;
        }
      });
      print("[Optimisation] Capteur accéléromètre : ÉCOUTE ACTIVÉE");
    }
  }

  /// Arrête proprement le flux du capteur pour économiser la batterie
  void _arreterEcouteAccelerometre() {
    _accelSubscription?.cancel();
    _accelSubscription = null;
    print("[Optimisation] Capteur accéléromètre : ÉCOUTE SUSPENDUE (Économie d'énergie)");
  }

  // =========================================================================
  // === GESTION DU CYCLE DE VIE DES FLUX (ANTI-BATTERY DRAIN) ===
  // =========================================================================
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Cas 1 : L'application passe en arrière-plan (Bouton Home, appel téléphonique, etc.)
    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      _arreterEcouteAccelerometre();
    } 
    // Cas 2 : L'utilisateur revient sur l'application
    else if (state == AppLifecycleState.resumed) {
      _demarrerEcouteAccelerometre();
    }
  }

  Future<void> _initialiserCamera() async {
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

  /// Charge les modèles d'IA et met à jour l'état pour activer les boutons.
  Future<void> _preparerIA() async {
    await IAService().initModels();
    if (mounted) {
      setState(() {
        _isIaReady = true;
      });
    }
  }

  @override
  void dispose() {
    // Retrait obligatoire de l'observateur de cycle de vie
    WidgetsBinding.instance.removeObserver(this);
    
    // Libère les ressources de la caméra
    _controller?.dispose();
    
    // Sécurité maximale : on force la coupure du flux avant la destruction
    _arreterEcouteAccelerometre();
    _accelerometerNotifier.dispose();
    super.dispose();
  }

  /// Déclenche la prise de vue avec la caméra, optimise l'image si elle est trop lourde, puis navigue vers le résultat.
  Future<void> _prendrePhoto() async {
    if (_controller != null && _controller!.value.isInitialized) {
      try {
        setState(() => _isOptimizing = true);
        
        // Assure que les modèles d'IA sont initialisés avant de continuer.
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
        
        // Assure que les modèles d'IA sont initialisés avant de continuer.
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
  Future<void> _allerVersResultat(String imagePath) async {
    // OPTIMISATION : On met en pause l'accéléromètre PENDANT qu'on est sur l'écran d'édition
    _arreterEcouteAccelerometre();
    
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => EcranResultat(photoPath: imagePath),
      ),
    );
    
    // Dès que le "await" se termine (l'utilisateur a fait "Retour"), on relance le capteur !
    if (mounted) {
      _demarrerEcouteAccelerometre();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Choisir le mur'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        elevation: 0, // Design plus flat
      ),
      body: Stack(
        children: [
          Column(
            children: [
              Expanded(
                child: Container(
                  width: double.infinity,
                  margin: const EdgeInsets.all(16.0),
                  // Ajout d'une ombre douce derrière la caméra
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(color: Colors.black.withValues(alpha: 0.1), blurRadius: 20, spreadRadius: 2)
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(20),
                    child: _controller != null && _controller!.value.isInitialized
                        // Affiche le retour vidéo de la caméra si elle est prête.
                        ? Stack(
                            fit: StackFit.expand,
                            children: [
                              FittedBox(
                                fit: BoxFit.cover,
                                child: SizedBox(
                                  width: _controller!.value.previewSize?.height ?? 1,
                                  height: _controller!.value.previewSize?.width ?? 1,
                                  child: CameraPreview(_controller!),
                                ),
                              ),
                              // Calque interactif du niveau à bulle
                              ValueListenableBuilder<AccelerometerEvent?>(
                                valueListenable: _accelerometerNotifier,
                                builder: (context, event, child) {
                                  // x = inclinaison latérale (gauche/droite)
                                  // z = inclinaison d'avant en arrière (quand on tient le téléphone debout)
                                  double xTilt = event?.x ?? 0.0;
                                  double yTilt = event?.z ?? 0.0;
                                  
                                  return CustomPaint(
                                    painter: _NiveauBullePainter(xTilt: xTilt, yTilt: yTilt),
                                  );
                                },
                              ),
                            ],
                          )
                        : const Center(child: CircularProgressIndicator()),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(bottom: 40.0, left: 16.0, right: 16.0),
                // Colonne pour afficher les boutons et le message de chargement de l'IA en dessous.
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                          // Désactive le bouton si une opération est en cours ou si l'IA n'est pas prête.
                          onPressed: (_isOptimizing || !_isIaReady) ? null : _ouvrirGalerie,
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Galerie'),
                          style: ElevatedButton.styleFrom(
                            elevation: 0,
                            padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                            backgroundColor: Colors.teal.withValues(alpha: 0.1), // Subtile touche de couleur
                            foregroundColor: Colors.teal,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)), // Bouton très arrondi
                          ),
                        ),
                        ElevatedButton.icon(
                          onPressed: (_isOptimizing || !_isIaReady) ? null : _prendrePhoto,
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Photo'),
                          style: ElevatedButton.styleFrom(
                            elevation: 4,
                            shadowColor: Colors.teal.withValues(alpha: 0.4),
                            padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                            backgroundColor: Theme.of(context).colorScheme.primary,
                            foregroundColor: Theme.of(context).colorScheme.onPrimary,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)), // Bouton très arrondi
                          ),
                        ),
                      ],
                    ),
                    // Affiche un indicateur de chargement pendant que l'IA s'initialise.
                    if (!_isIaReady)
                      const Padding(
                        padding: EdgeInsets.only(top: 15.0),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(width: 15, height: 15, child: CircularProgressIndicator(strokeWidth: 2)),
                            SizedBox(width: 10),
                            Text("Chargement du moteur IA...", style: TextStyle(color: Colors.grey, fontStyle: FontStyle.italic)),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
          // Affiche un indicateur de chargement par-dessus l'interface pendant l'optimisation de l'image.
          if (_isOptimizing)
            // STYLE : Remplacement du fond noir par un bel effet de verre flouté (Glassmorphism)
            BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 5.0, sigmaY: 5.0),
              child: Container(
                color: Colors.black.withValues(alpha: 0.4),
                child: const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(color: Colors.white),
                      SizedBox(height: 20),
                      Text("Préparation de l'espace...", style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w500)),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// =========================================================================
// === CLASSE UTILITAIRE : DESSIN DU NIVEAU À BULLE GYROSCOPIQUE ===
// =========================================================================
class _NiveauBullePainter extends CustomPainter {
  final double xTilt;
  final double yTilt;

  _NiveauBullePainter({required this.xTilt, required this.yTilt});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    
    // Détermine si le téléphone est considéré comme "droit" (Tolérance de 0.5 d'accélération)
    bool isAligned = xTilt.abs() < 0.6 && yTilt.abs() < 0.6;
    
    // 1. Dessin de la cible centrale (Le viseur fixe)
    final targetColor = isAligned ? Colors.greenAccent : Colors.white.withValues(alpha: 0.6);
    final targetPaint = Paint()
      ..color = targetColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = isAligned ? 3.0 : 2.0;
    
    // Cercle central
    canvas.drawCircle(center, 40.0, targetPaint);
    
    // Croix directrice (réticule)
    canvas.drawLine(center - const Offset(55, 0), center - const Offset(15, 0), targetPaint);
    canvas.drawLine(center + const Offset(15, 0), center + const Offset(55, 0), targetPaint);
    canvas.drawLine(center - const Offset(0, 55), center - const Offset(0, 15), targetPaint);
    canvas.drawLine(center + const Offset(0, 15), center + const Offset(0, 55), targetPaint);

    // 2. Dessin de la bulle mobile (Indicateur de gravité)
    // Multiplicateur pour rendre le mouvement plus ample sur l'écran
    double sensitivity = 15.0;
    double maxRadius = 80.0; // Ne pas sortir trop loin du viseur
    
    // Calcul de la position de la bulle (Inversée pour suivre l'inclinaison physique naturelle)
    double dx = (-xTilt * sensitivity).clamp(-maxRadius, maxRadius);
    double dy = (-yTilt * sensitivity).clamp(-maxRadius, maxRadius);
    Offset bubblePos = center + Offset(dx, dy);

    final bubblePaint = Paint()
      ..color = isAligned ? Colors.greenAccent : Colors.amber
      ..style = PaintingStyle.fill;
    
    // Bulle avec un léger effet d'ombre pour ressortir
    final shadowPaint = Paint()
      ..color = Colors.black45
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3.0);
      
    canvas.drawCircle(bubblePos, 12.0, shadowPaint);
    canvas.drawCircle(bubblePos, 12.0, bubblePaint);
  }

  @override
  bool shouldRepaint(covariant _NiveauBullePainter oldDelegate) {
    return oldDelegate.xTilt != xTilt || oldDelegate.yTilt != yTilt;
  }
}