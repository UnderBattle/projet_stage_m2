import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'screens/ecran_accueil.dart';
import 'utils/image_utils.dart';

/// Liste globale contenant les caméras disponibles sur l'appareil.
List<CameraDescription> cameras = [];

/// Point d'entrée principal de l'application.
/// Initialise le framework Flutter et tente de récupérer les caméras avant de lancer l'interface.
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  nettoyerCacheImages();

  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    print('Erreur caméra : ${e.code}, ${e.description}');
  }
  runApp(const MonApplication());
}

/// Widget racine de l'application.
/// Configure le thème global et définit l'écran d'accueil.
class MonApplication extends StatelessWidget {
  const MonApplication({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Simulateur d\'Installation', // Titre généralisé
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        // Utilisation d'un Teal plus profond et moderne pour l'UI globale
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF00796B), 
          primary: const Color(0xFF00796B),
          surface: const Color(0xFFF5F7FA), // Fond très légèrement grisé pour faire ressortir le blanc
        ),
        useMaterial3: true,
      ),
      // Passe la liste des caméras à l'écran d'accueil pour initialiser l'appareil photo.
      home: EcranAccueil(cameras: cameras), 
    );
  }
}