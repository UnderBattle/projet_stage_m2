import 'package:flutter/material.dart';

// Le point d'entrée de l'application
void main() {
  runApp(const MonApplication());
}

// Le Widget racine (La configuration de l'appli)
class MonApplication extends StatelessWidget {
  const MonApplication({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mon Stage M2',
      debugShowCheckedModeBanner: false, // Enlève le petit bandeau "DEBUG" en haut à droite
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true, // Utilise le dernier design standard de Google
      ),
      home: const EcranAccueil(), // Définit l'écran qui s'affiche au lancement
    );
  }
}

// Ton premier écran visuel
class EcranAccueil extends StatelessWidget {
  const EcranAccueil({super.key});

  @override
  Widget build(BuildContext context) {
    // Scaffold est le squelette de l'écran (qui gère la barre du haut, le fond, etc.)
    return Scaffold(
      appBar: AppBar(
        title: const Text('Simulateur de Climatisation'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: const Center( // Centre son contenu au milieu de l'écran
        child: Text(
          'L\'interface Flutter est prête !',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
        ),
      ),
      // Un petit bouton flottant en bas à droite (typique des applis mobiles)
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          // ignore: avoid_print
          print("Clic ! Bientôt, ça ouvrira l'appareil photo.");
        },
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
}