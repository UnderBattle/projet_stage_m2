// Création d'un modèle de données strict
class Equipement {
  final String nom;
  final String chemin;
  final double profondeur;
  final double hauteur;
  final double largeur;

  Equipement({
    required this.nom,
    required this.chemin,
    required this.profondeur,
    required this.hauteur,
    required this.largeur,
  });
}

/// Service Singleton qui gère la base de données des équipements disponibles
class CatalogueService {
  static final CatalogueService _instance = CatalogueService._internal();
  factory CatalogueService() => _instance;
  CatalogueService._internal();

  // Le catalogue est maintenant typé et protégé
  final Map<String, List<Equipement>> catalogueGlobal = {
    'Climatisations': [
      Equipement(
        nom: 'Takao Plus Blanc',
        chemin: 'assets/installations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png',
        profondeur: 240.0,
        hauteur: 270.0, 
        largeur: 798.0,
      ),
      Equipement(
        nom: 'Takao Plus Noir',
        chemin: 'assets/installations/clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png',
        profondeur: 240.0,
        hauteur: 270.0,
        largeur: 798.0,
      )
    ],
    'Pompes à Chaleur': [], 
    'Chaudières': [],       
  };
}