import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/ia_service.dart';
import '../services/catalogue_service.dart';
import '../traitement_image.dart';
import 'ecran_accueil.dart';

class EcranSplash extends StatefulWidget {
  const EcranSplash({super.key});

  @override
  State<EcranSplash> createState() => _EcranSplashState();
}

class _EcranSplashState extends State<EcranSplash> {
  @override
  void initState() {
    super.initState();
    _initialiserApplication();
  }

  Future<void> _initialiserApplication() async {
    try {
      // 1. Récupération des caméras en arrière-plan (Libère le main.dart)
      final cameras = await availableCameras();

      // 2. Chargement lourd des modèles IA (YOLO et LaMa)
      await IAService().initModels();

      // 3. Pré-chargement des images du catalogue dans la RAM vidéo
      if (mounted) {
        final catalogueGlobal = CatalogueService().catalogueGlobal;
        for (var listeEquipements in catalogueGlobal.values) {
          for (var equipement in listeEquipements) {
            precacheImage(AssetImage(equipement.chemin), context);
          }
        }
      }

      // 4. Initialisation du Worker OpenCV (Optionnel mais fait gagner du temps pour le premier rendu)
      await TraitementImage.initWorker();

      // 5. Navigation vers l'écran d'accueil une fois tout terminé
      if (mounted) {
        // On utilise pushReplacement pour que l'utilisateur ne puisse pas faire "Retour" vers le Splash
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(
            builder: (context) => EcranAccueil(cameras: cameras),
          ),
        );
      }
    } catch (e) {
      print("[Splash] Erreur d'initialisation : $e");
      // En cas d'erreur (ex: permissions caméra non accordées), on force quand même le passage à l'accueil
      if (mounted) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(
            builder: (context) => const EcranAccueil(cameras: []),
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: theme.colorScheme.primary, // Fond aux couleurs de la marque
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // =========================================================================
            // === LOGO DE L'APPLICATION ===
            // =========================================================================
            // Remplacé par une icône thématique parfaitement adaptée au génie climatique.
            Container(
              padding: const EdgeInsets.all(24.0),
              decoration: BoxDecoration(
                color: Colors.white,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(color: Colors.black.withValues(alpha: 0.2), blurRadius: 20, offset: const Offset(0, 10))
                ],
              ),
              // CORRECTION : Utilisation de Icons.heat_pump (Pompe à chaleur) à la place du sac de réparation
              child: Icon(Icons.heat_pump, size: 80, color: theme.colorScheme.primary),
            ),
            const SizedBox(height: 40),
            
            // TEXTE DE CHARGEMENT
            const Text(
              "Chauffage Der Energies", 
              style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold, letterSpacing: 1.2)
            ),
            const SizedBox(height: 10),
            Text(
              "Initialisation du moteur 3D...", 
              style: TextStyle(color: Colors.white70, fontSize: 16, fontStyle: FontStyle.italic)
            ),
            const SizedBox(height: 40),
            
            // INDICATEUR DE PROGRESSION BLANC
            const SizedBox(
              width: 40,
              height: 40,
              child: CircularProgressIndicator(color: Colors.white, strokeWidth: 3),
            ),
          ],
        ),
      ),
    );
  }
}