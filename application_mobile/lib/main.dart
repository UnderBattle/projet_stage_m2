import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'screens/ecran_accueil.dart';

/// Liste globale contenant les caméras disponibles sur l'appareil.
List<CameraDescription> cameras = [];

/// Point d'entrée principal de l'application.
/// Initialise le framework Flutter et tente de récupérer les caméras avant de lancer l'interface.
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
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
      title: 'Simulateur Clim',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      // Passe la liste des caméras à l'écran d'accueil pour initialiser l'appareil photo.
      home: EcranAccueil(cameras: cameras), 
    );
  }
}