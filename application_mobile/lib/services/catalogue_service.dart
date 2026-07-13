// Création d'un modèle de données strict
class Equipement {
  final String nom;
  final String chemin;
  final double profondeur;
  final double hauteur;
  final double largeur;
  
  // Champ optionnel
  final double? poids;
  final List<String>? puissances;
  final double? prixMin;
  final double? prixMax;

  Equipement({
    required this.nom,
    required this.chemin,
    required this.profondeur,
    required this.hauteur,
    required this.largeur,
    this.poids,
    this.puissances,
    this.prixMin,
    this.prixMax,
  });
}

/// Service Singleton qui gère la base de données des équipements disponibles
class CatalogueService {
  static final CatalogueService _instance = CatalogueService._internal();
  factory CatalogueService() => _instance;
  CatalogueService._internal();

  // Le catalogue est typé et protégé
  final Map<String, List<Equipement>> catalogueGlobal = {
    'Climatisations': [
      Equipement(
        nom: 'Takao Plus Blanc',
        chemin: 'assets/installations/climatisations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png',
        profondeur: 240.0,
        hauteur: 270.0, 
        largeur: 798.0,
        poids: 10.0,
        puissances: ['1500 W (Multi-Split Uniquement)', '2000 W', '2500 W', '3400 W', '4200 W'],
        prixMin: 823.0,
        prixMax: 1238.0,
      ),
      Equipement(
        nom: 'Takao Plus Noir',
        chemin: 'assets/installations/climatisations/clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png',
        profondeur: 240.0,
        hauteur: 270.0,
        largeur: 798.0,
        poids: 10.0,
        puissances: ['1500 W (Multi-Split Uniquement)', '2000 W', '2500 W', '3400 W', '4200 W'],
        prixMin: 906.0,
        prixMax: 1361.0,
      )
    ],
    'Pompes à Chaleur': [
      Equipement(
        nom: 'Takao Unité Extérieure',
        chemin: 'assets/installations/unites_exterieures/unite_exterieure_takao_plus/unite_exterieure_takao_plus.png',
        profondeur: 290.0,
        hauteur: 542.0,
        largeur: 799.0,
        prixMin: 1285.0,
        prixMax: 2496.0,
      )
    ], 
    'Gaz & Fioul': [],
    'Thermodynamique': []
  };
}