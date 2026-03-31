import 'package:flutter/material.dart';

void main() {
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
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal), // Un petit vert "Atlantic"
        useMaterial3: true,
      ),
      home: const EcranAccueil(), 
    );
  }
}

class EcranAccueil extends StatefulWidget {
  const EcranAccueil({super.key});

  @override
  State<EcranAccueil> createState() => _EcranAccueilState();
}

class _EcranAccueilState extends State<EcranAccueil> {
  
  // La "mémoire" de notre écran : quelle clim est actuellement sélectionnée ?
  String modeleSelectionne = 'Takao Plus Blanc';

  // Notre catalogue de climatisations
  final List<String> catalogueClims = [
    'Takao Plus Blanc',
    'Takao Plus Noir'
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Simulation Climatisation'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      
      // On utilise une Column pour empiler les éléments de haut en bas
      body: Column(
        children: [
          
          // Le menu déroulant du catalogue
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text('Modèle : ', style: TextStyle(fontSize: 18)),
                const SizedBox(width: 10), // Un petit espace
                
                // Le widget DropdownButton (Menu déroulant)
                DropdownButton<String>(
                  value: modeleSelectionne,
                  icon: const Icon(Icons.arrow_downward),
                  elevation: 16,
                  style: const TextStyle(color: Colors.teal, fontSize: 18, fontWeight: FontWeight.bold),
                  underline: Container(height: 2, color: Colors.tealAccent),
                  
                  // Quand l'utilisateur clique sur un autre modèle...
                  onChanged: (String? nouveauChoix) {
                    // setState "rafraîchit" l'écran avec la nouvelle valeur
                    setState(() {
                      modeleSelectionne = nouveauChoix!;
                    });
                  },
                  
                  // On transforme notre liste de texte en éléments cliquables
                  items: catalogueClims.map<DropdownMenuItem<String>>((String modele) {
                    return DropdownMenuItem<String>(
                      value: modele,
                      child: Text(modele),
                    );
                  }).toList(),
                ),
              ],
            ),
          ),

          // Espace pour la photo
          Expanded(
            child: Container(
              width: double.infinity,
              margin: const EdgeInsets.all(16.0),
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(15), // Bords arrondis
                border: Border.all(color: Colors.grey.shade400, width: 2), // Petite bordure
              ),
              child: const Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.image_search, size: 100, color: Colors.grey),
                  SizedBox(height: 16),
                  Text(
                    "Aucune photo sélectionnée.\nAppuyez sur l'appareil photo.",
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey, fontSize: 16),
                  ),
                ],
              ),
            ),
          ),
          
          // Un petit espace en bas pour ne pas coller au bouton
          const SizedBox(height: 80), 
        ],
      ),

      // Bouton Camera
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          // ignore: avoid_print
          print("Lancement de la caméra pour incruster une $modeleSelectionne !");
        },
        label: const Text('Prendre une photo'),
        icon: const Icon(Icons.camera_alt),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}