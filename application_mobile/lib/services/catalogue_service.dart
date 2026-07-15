/// Structure de données stricte pour gérer les variantes d'un équipement.
class VarianteEquipement {
  final String valeur;
  final String? chemin;
  final double profondeur;
  final double hauteur;
  final double largeur;
  final double? poids;
  final double? prix;

  VarianteEquipement({
    required this.valeur,
    this.chemin,
    required this.profondeur,
    required this.hauteur,
    required this.largeur,
    this.poids,
    this.prix,
  });
}

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

  // NOUVEAU : Liste optionnelle de variantes géométriques et techniques
  final List<VarianteEquipement>? variantes;

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
    this.variantes,
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
        nom: 'Takao Plus',
        chemin: 'assets/installations/climatisations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png',
        profondeur: 240.0,
        hauteur: 270.0, 
        largeur: 798.0,
        poids: 10.0,
        puissances: ['1500 W (Multi-Split Uniquement)', '2000 W', '2500 W', '3400 W', '4200 W'], 
        prixMin: 823.0,
        prixMax: 1361.0,
        variantes: [
          VarianteEquipement(
            valeur: 'Blanc', 
            chemin: 'assets/installations/climatisations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png',
            profondeur: 240.0,
            hauteur: 270.0,
            largeur: 798.0,
            poids: 10.0
          ),
          VarianteEquipement(
            valeur: 'Noir', 
            chemin: 'assets/installations/climatisations/clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png',
            profondeur: 240.0,
            hauteur: 270.0,
            largeur: 798.0,
            poids: 10.0,
          )
        ],
      ),
      Equipement(
        nom: 'Takao Unité Extérieure',
        chemin: 'assets/installations/climatisations/unite_exterieure_takao/unite_exterieure_takao.png',
        profondeur: 290.0,
        hauteur: 542.0,
        largeur: 799.0,
        prixMin: 1285.0,
        prixMax: 2496.0,
      )
    ],
    'Pompes à Chaleur': [
      Equipement(
        nom: 'Alféa Extensa A.I. R32',
        chemin: 'assets/installations/pompe_a_chaleur/alfea_extensa_ai_r32/unité_intérieur_r32_service.png',
        profondeur: 493.0,
        hauteur: 847.0,
        largeur: 450.0,
        puissances: ['5 KW', '6 KW', '8 KW', '10 KW'],
        prixMin: 6332.0,
        prixMax: 9349.0,
      ),
      Equipement(
        nom: 'Alféa Extensa A.I. Duo R32',
        chemin: 'assets/installations/pompe_a_chaleur/alfea_extensa_ai_r32/unité_intérieur_r32_duo.png',
        profondeur: 700.0,
        hauteur: 1863.0,
        largeur: 648.0,
        puissances: ['3 KW', '5 KW', '6 KW', '8 KW', '10 KW'],
        prixMin: 8073.0,
        prixMax: 11505.0,
      ),
      Equipement(
        nom: 'Alféa Extensa A.I. R32 Unité Extérieure',
        chemin: 'assets/installations/pompe_a_chaleur/alfea_extensa_ai_r32/unité_exterieur_r32.png',
        profondeur: 325.0,
        hauteur: 632.0,
        largeur: 886.0,
        variantes: [
          VarianteEquipement(
            valeur: '3 KW - 6 KW',
            profondeur: 325.0,
            hauteur: 632.0,
            largeur: 886.0,
          ),
          VarianteEquipement(
            valeur: '8 KW',
            profondeur: 349.0,
            hauteur: 716.0,
            largeur: 907.0,
          ),
          VarianteEquipement(
            valeur: '10 KW',
            profondeur: 372.0,
            hauteur: 830.0,
            largeur: 977.0,
          )
        ],
      )
    ], 
    'Gaz & Fioul': [],
    'Thermodynamique': [
      Equipement(
        nom: 'Calypso Mural',
        chemin: 'assets/installations/thermodynamique/calypso/calypso_mural_100l.png', 
        profondeur: 590.0,
        hauteur: 1060.0,
        largeur: 590.0,
        poids: 57.0,
        variantes: [
          VarianteEquipement(
            valeur: '100L',
            chemin: 'assets/installations/thermodynamique/calypso/calypso_mural_100l.png',
            profondeur: 590.0,
            hauteur: 1060.0,
            largeur: 590.0,
          ),
          VarianteEquipement(
            valeur: '150L',
            chemin: 'assets/installations/thermodynamique/calypso/calypso_mural_150l.png',
            profondeur: 590.0,
            hauteur: 1310.0,
            largeur: 590.0,
          )
        ]
      ),
      Equipement(
        nom: 'Calypso Vertical sur Socle',
        chemin: 'assets/installations/thermodynamique/calypso/calypso_socle_200l.png', 
        profondeur: 600.0,
        hauteur: 1710.0,
        largeur: 650.0,
        poids: 75.0,
        variantes: [
          VarianteEquipement(
            valeur: '200L',
            chemin: 'assets/installations/thermodynamique/calypso/calypso_socle_200l.png',
            profondeur: 600.0,
            hauteur: 1710.0,
            largeur: 650.0,
          ),
          VarianteEquipement(
            valeur: '240L',
            chemin: 'assets/installations/thermodynamique/calypso/calypso_socle_240l.png',
            profondeur: 600.0,
            hauteur: 1900.0,
            largeur: 650.0,
          )
        ]
      )
    ]
  };
}