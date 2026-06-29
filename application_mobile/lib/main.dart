import 'package:flutter/material.dart';
import 'screens/ecran_splash.dart'; // NOUVEAU : Import du Splash Screen
import 'utils/image_utils.dart';
import 'theme/app_theme.dart';

/// Point d'entrée principal de l'application.
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  nettoyerCacheImages();

  // CORRECTION : L'initialisation des caméras a été déplacée dans l'EcranSplash 
  // pour que l'application s'ouvre instantanément !
  runApp(const MonApplication());
}

/// Widget racine de l'application.
/// Configure le thème global et définit l'écran d'accueil.
class MonApplication extends StatelessWidget {
  const MonApplication({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Simulateur d\'Installation', 
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.system, // S'adapte automatiquement au réglage du téléphone
      
      // L'application démarre désormais sur l'écran de chargement optimisé
      home: const EcranSplash(), 
    );
  }
}