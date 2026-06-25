import 'dart:io';
import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:image_gallery_saver_plus/image_gallery_saver_plus.dart';
import 'package:flutter/services.dart';

import '../traitement_image.dart';
import '../services/ia_service.dart';
import '../services/catalogue_service.dart';
import '../models/devis_models.dart';
import '../utils/painters_resultat.dart';
import '../utils/image_utils.dart';
import '../widgets/catalogue_devis.dart';
import '../widgets/boutons_action_devis.dart';

/// Écran affichant l'image capturée, exécutant la détection de l'IA, 
/// et permettant à l'utilisateur d'incruster et de manipuler des modèles d'équipement.
class EcranResultat extends StatefulWidget {
  final String photoPath;

  const EcranResultat({super.key, required this.photoPath});

  @override
  State<EcranResultat> createState() => _EcranResultatState();
}

class _EcranResultatState extends State<EcranResultat> {
  // =========================================================================
  // === VARIABLES D'ÉTAT ===
  // =========================================================================
  
  // NOUVEAU : On ne force plus la catégorie "Climatisations", on prend dynamiquement la 1ère dispo
  late String _categorieSelectionnee;
  Equipement? _modeleSelectionne;
  
  bool _isProcessing = true;
  String _loadingMessage = "Analyse en cours...";
  
  Uint8List? _imageResultatBytes; // L'image finale (avec goulotte + equipement)
  
  // =========================================================================
  // === SYSTÈME DE CACHE INDÉPENDANT ===
  // =========================================================================
  // En séparant l'équipement et la goulotte dans 2 variables différentes, on évite les recalculs inutiles !
  Uint8List? _imageFondPropreBytes;
  Uint8List? _imageFondAvecGoulotteBytes; 
  Uint8List? _calqueEquipementPngBytes; // Le calque de l'équipement transparent (PNG)
  
  // NOUVEAU: Mémorise le dernier rendu complet (non rogné) pour le drag hors de l'écran
  Uint8List? _dernierCalqueEquipementCompletBytes;
  Offset _decalageDuCalqueComplet = Offset.zero;

  int? _imageWidth;
  int? _imageHeight;
  List<Map<String, double>>? _pointsCibles;
  bool _isManualPlacementMode = false;
  
  // NOUVEAU : Gère l'affichage de la carte de confirmation en bas de l'écran
  bool _attenteConfirmationIA = false;

  // Sécurité pour ne charger le catalogue en RAM qu'une seule fois
  bool _isCatalogPrecached = false;

  // Contrôleur pour gérer programmatiquement le zoom et le déplacement de l'image
  final TransformationController _transformationController = TransformationController();

  // =========================================================================
  // GESTION D'ÉTAT OPTIMISÉE (VALUENOTIFIERS)
  // =========================================================================
  final ValueNotifier<Offset> _decalageNotifier = ValueNotifier(Offset.zero);
  final ValueNotifier<double> _splitNotifier = ValueNotifier(1.0);
  final ValueNotifier<bool> _isDraggingEquipementNotifier = ValueNotifier(false);

  // Pile d'historique pour annuler les déplacements successifs de l'équipement
  final List<Offset> _historiqueDecalages = [Offset.zero];
  
  // OPTIMISATION : Notifier ultra-léger pour dire au bouton Undo de s'afficher SANS recalculer à chaque frame du glissement
  final ValueNotifier<int> _historiqueLengthNotifier = ValueNotifier(1);

  // Variables d'état pour la Goulotte interactive
  bool _isDrawGoulotteMode = false;
  final ValueNotifier<LigneGoulotte?> _goulotteNotifier = ValueNotifier(null);
  LigneGoulotte? _goulotteInitiale; // Mémoire de la position de départ de la goulotte pour le bouton Undo
  
  final ValueNotifier<bool> _isDraggingGoulotteNotifier = ValueNotifier(false);

  // Variables temporaires pour ne pas casser le drag du tout premier tracé
  Offset? _goulotteStartOrig;
  final ValueNotifier<Offset?> _goulotteCurrentEndOrigNotifier = ValueNotifier(null);

  // Notifie pour la position de la loupe de précision
  final ValueNotifier<Offset?> _magnifierPositionNotifier = ValueNotifier(null);

  // Compteur de doigts sur l'écran pour empêcher les erreurs lors du zoom
  int _activePointers = 0;

  @override
  void initState() {
    super.initState();
    
    // Initialisation de l'Isolate persistant pour OpenCV
    TraitementImage.initWorker();
    
    // Dynamisme parfait basé sur le catalogue existant
    _categorieSelectionnee = CatalogueService().catalogueGlobal.keys.first;
    _analyserImage();
  }

  // =========================================================================
  // === OPTIMISATION : MISE EN CACHE RAM DU CATALOGUE ===
  // =========================================================================
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // didChangeDependencies nous donne accès au 'context' nécessaire pour precacheImage
    if (!_isCatalogPrecached) {
      _precacherCatalogue();
      _isCatalogPrecached = true;
    }
  }

  /// Parcourt le catalogue et met les images en cache RAM pour un affichage instantané
  void _precacherCatalogue() {
    final catalogueGlobal = CatalogueService().catalogueGlobal;
    for (var listeEquipements in catalogueGlobal.values) {
      for (var equipement in listeEquipements) {
        precacheImage(AssetImage(equipement.chemin), context);
      }
    }
    print("[Optimisation] Images du catalogue préchargées en RAM/VRAM avec succès !");
  }

  @override
  void dispose() {
    _transformationController.dispose();
    _decalageNotifier.dispose();
    _splitNotifier.dispose();
    _isDraggingEquipementNotifier.dispose();
    _historiqueLengthNotifier.dispose(); // OPTIMISATION: Nettoyage
    _goulotteNotifier.dispose();
    _isDraggingGoulotteNotifier.dispose();
    _goulotteCurrentEndOrigNotifier.dispose();
    _magnifierPositionNotifier.dispose(); // Libération de la mémoire de la loupe
    
    // Fermeture propre de l'Isolate persistant
    TraitementImage.disposeWorker();
    
    super.dispose();
  }

  /// Affiche une bannière d'erreur visible pour l'utilisateur
  void _montrerErreur(String message) {
    if (!mounted) return;
    final theme = Theme.of(context);
    
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message, style: TextStyle(color: theme.colorScheme.onError)),
        backgroundColor: theme.colorScheme.error,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 4),
        action: SnackBarAction(
          label: 'OK',
          textColor: theme.colorScheme.onError,
          onPressed: () {},
        ),
      ),
    );
  }
  
  // Fonction pour empêcher un point de sortir des limites strictes de l'image
  Offset _clampToImageBounds(Offset point) {
    if (_imageWidth == null || _imageHeight == null) return point;
    return Offset(
      point.dx.clamp(0.0, _imageWidth!.toDouble()),
      point.dy.clamp(0.0, _imageHeight!.toDouble()),
    );
  }

  // Fonction pour forcer la goulotte à être parfaitement droite (horizontale ou verticale)
  Offset _snapToOrthogonal(Offset reference, Offset target) {
    double dx = (target.dx - reference.dx).abs();
    double dy = (target.dy - reference.dy).abs();
    
    if (dx > dy) {
      // Mouvement majoritairement horizontal -> On force l'alignement sur l'axe Y de référence
      return Offset(target.dx, reference.dy);
    } else {
      // Mouvement majoritairement vertical -> On force l'alignement sur l'axe X de référence
      return Offset(reference.dx, target.dy);
    }
  }

  // =========================================================================
  // === LOGIQUE MÉTIER ET IA ===
  // =========================================================================
  Future<void> _analyserImage() async {
    final yoloModel = IAService().yoloModel;
    if (yoloModel == null) {
      _montrerErreur("Le modèle IA (YOLO) n'a pas pu être chargé.");
      return;
    }

    try {
      setState(() => _loadingMessage = "Détection de l'autocollant...");

      final imageBytes = await File(widget.photoPath).readAsBytes();
      var inputShape = yoloModel.getInputTensor(0).shape;
      bool isNHWC = inputShape[3] == 3;
      
      final resultMatrixPrep = await compute(prepareImageMatrixForIA, {
        'bytes': imageBytes,
        'isNHWC': isNHWC
      });

      if (resultMatrixPrep == null) throw Exception("Impossible de lire l'image.");

      _imageWidth = resultMatrixPrep['width'];
      _imageHeight = resultMatrixPrep['height'];
      var inputMatrix = resultMatrixPrep['matrix'];

      var outputShape = yoloModel.getOutputTensor(0).shape;
      var outputMatrix = List.generate(outputShape[0], (i) => 
        List.generate(outputShape[1], (j) => 
          List<double>.filled(outputShape[2], 0.0)
        )
      );

      yoloModel.run(inputMatrix, outputMatrix);

      double maxConfiance = 0;
      int meilleurIndex = 0;
      bool isTransposed = outputShape[1] == 21504;
      int nbColonnes = isTransposed ? 21504 : outputShape[2];

      for (int i = 0; i < nbColonnes; i++) {
        double confiance = isTransposed ? outputMatrix[0][i][4] : outputMatrix[0][4][i];
        if (confiance > maxConfiance) {
          maxConfiance = confiance;
          meilleurIndex = i;
        }
      }

      if (maxConfiance > 0.5) {
        double boxX = isTransposed ? outputMatrix[0][meilleurIndex][0] : outputMatrix[0][0][meilleurIndex];
        double boxY = isTransposed ? outputMatrix[0][meilleurIndex][1] : outputMatrix[0][1][meilleurIndex];
        double boxW = isTransposed ? outputMatrix[0][meilleurIndex][2] : outputMatrix[0][2][meilleurIndex];
        double boxH = isTransposed ? outputMatrix[0][meilleurIndex][3] : outputMatrix[0][3][meilleurIndex];
        
        double scale = (boxW < 2.0 && boxH < 2.0) ? 1024.0 : 1.0;
        List<Map<String, double>> rawPoints = [];
        double confMoyennePoints = 0;

        for(int point = 0; point < 4; point++) {
           int idxX = 5 + (point * 3);
           int idxY = idxX + 1;
           int idxConf = idxX + 2;
           
           double px = isTransposed ? outputMatrix[0][meilleurIndex][idxX] : outputMatrix[0][idxX][meilleurIndex];
           double py = isTransposed ? outputMatrix[0][meilleurIndex][idxY] : outputMatrix[0][idxY][meilleurIndex];
           double pConf = isTransposed ? outputMatrix[0][meilleurIndex][idxConf] : outputMatrix[0][idxConf][meilleurIndex];
           
           confMoyennePoints += pConf;
           rawPoints.add({'x': px * scale, 'y': py * scale});
        }
        confMoyennePoints = confMoyennePoints / 4.0;

        print("[IA] Confiance moyenne des points : $confMoyennePoints");

        if (confMoyennePoints >= 0.9875) {
          _pointsCibles = TraitementImage.trierPoints(rawPoints);
        } else {
          double xMin = (boxX * scale) - (boxW * scale) / 2;
          double yMin = (boxY * scale) - (boxH * scale) / 2;
          double xMax = (boxX * scale) + (boxW * scale) / 2;
          double yMax = (boxY * scale) + (boxH * scale) / 2;
          
          _pointsCibles = [
            {'x': xMin, 'y': yMin},
            {'x': xMax, 'y': yMin},
            {'x': xMax, 'y': yMax},
            {'x': xMin, 'y': yMax} 
          ];
        }

        if (_pointsCibles != null) {
          // L'IA a trouvé la zone : On affiche la bounding box et on montre le panel en bas
          setState(() {
            _isProcessing = false;
            _isManualPlacementMode = true; 
            _attenteConfirmationIA = true; // Déclenche l'affichage du menu "Oui / Non"
          });
        }
      } else {
        setState(() {
          _pointsCibles = null;
          _isProcessing = false;
        });
        
        WidgetsBinding.instance.addPostFrameCallback((_) {
          _demanderPlacementManuel();
        });
      }
    } catch (e) {
      print("[IA - ERREUR] Exception : $e");
      if (mounted) {
        setState(() {
          _pointsCibles = null;
          _isProcessing = false;
        });
        _montrerErreur("Une erreur est survenue lors de l'analyse de l'image par l'IA.");
      }
    }
  }

  void _demanderPlacementManuel() {
    final theme = Theme.of(context);
    
    // On conserve la Pop-up ici car l'IA a complètement échoué, on oblige l'utilisateur à choisir.
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        backgroundColor: theme.cardColor,
        title: Text("Autocollant introuvable", style: TextStyle(color: theme.colorScheme.onSurface)),
        content: Text("L'IA n'a pas pu détecter l'autocollant avec certitude.\nVoulez-vous placer la zone manuellement ?", style: TextStyle(color: theme.colorScheme.onSurface)),
        actions: [
          ElevatedButton.icon(
            icon: const Icon(Icons.touch_app),
            label: const Text("Placer manuellement"),
            style: ElevatedButton.styleFrom(backgroundColor: theme.colorScheme.primary, foregroundColor: theme.colorScheme.onPrimary),
            onPressed: () {
              Navigator.pop(context);
              _activerModeManuel();
            },
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context); 
              Navigator.pop(context); 
            },
            child: Text("Annuler", style: TextStyle(color: theme.colorScheme.onSurface.withValues(alpha: 0.6))),
          ),
        ],
      ),
    );
  }

  void _activerModeManuel() {
    setState(() {
      _isManualPlacementMode = true;
      _attenteConfirmationIA = false; // On n'a pas besoin de confirmer l'IA puisqu'elle a échoué
      _pointsCibles = [
        {'x': 512.0 - 75.0, 'y': 512.0 - 150.0}, // Haut Gauche
        {'x': 512.0 + 75.0, 'y': 512.0 - 150.0}, // Haut Droit
        {'x': 512.0 + 75.0, 'y': 512.0 + 150.0}, // Bas Droit
        {'x': 512.0 - 75.0, 'y': 512.0 + 150.0}, // Bas Gauche
      ];
    });
  }

  Future<void> _validerPlacementManuel() async {
    _transformationController.value = Matrix4.identity();
    setState(() {
      _isManualPlacementMode = false;
      _isProcessing = true;
      _loadingMessage = "Nettoyage de la zone sélectionnée..."; 
      _pointsCibles = TraitementImage.trierPoints(_pointsCibles!); 
    });

    try {
      _imageFondPropreBytes = await TraitementImage.effacerAutocollantWorker({
        'photoPath': widget.photoPath,
        'pointsIA': _pointsCibles!,
        'lamaBytes': IAService().lamaBytes,
      });
      _imageFondAvecGoulotteBytes = null; // On nettoie les caches car la base a changé
      _calqueEquipementPngBytes = null;
      _dernierCalqueEquipementCompletBytes = null; // On nettoie aussi le cache du drag
    } catch (e) {
      print("Erreur inpainting manuel : $e");
      if (mounted) {
         _montrerErreur("Impossible de nettoyer le mur (Erreur OpenCV).");
      }
    }

    setState(() {
      _isProcessing = false;
    });
  }

  // NOUVELLE ARCHITECTURE : On choisit ce qu'on recalcule !
  Future<void> _genererIncrustation({bool recomputeGoulotte = false, bool recomputeEquipement = false}) async {
    if (_pointsCibles == null || _modeleSelectionne == null || _imageFondPropreBytes == null) return;
    
    setState(() {
      _isProcessing = true;
      _loadingMessage = "Préparation du rendu...";
    });

    try {
      String equipementPath = _modeleSelectionne!.chemin;
      final ByteData data = await DefaultAssetBundle.of(context).load(equipementPath);
      Uint8List equipementBytes = data.buffer.asUint8List();
      
      double profondeur = _modeleSelectionne!.profondeur;
      double hauteur = _modeleSelectionne!.hauteur;
      double largeur = _modeleSelectionne!.largeur;

      double ptHgXOrig = _pointsCibles![0]['x']! * (_imageWidth! / 1024.0);
      double ptHgYOrig = _pointsCibles![0]['y']! * (_imageHeight! / 1024.0);
      double ptHdXOrig = _pointsCibles![1]['x']! * (_imageWidth! / 1024.0);
      double ptHdYOrig = _pointsCibles![1]['y']! * (_imageHeight! / 1024.0);
      double dx = ptHdXOrig - ptHgXOrig;
      double dy = ptHdYOrig - ptHgYOrig;
      
      double autoWPxOrig = math.sqrt(dx * dx + dy * dy);
      double ratioPxParMm = autoWPxOrig / 50.0;
      double largeurGoulotteOrig = 80.0 * ratioPxParMm; 

      // 1. GÉNÉRATION DE LA GOULOTTE (Seulement si demandé ou pas en cache)
      if (recomputeGoulotte || (_goulotteNotifier.value != null && _imageFondAvecGoulotteBytes == null)) {
        if (_goulotteNotifier.value != null) {
          setState(() => _loadingMessage = "Incrustation de la goulotte...");
          
          _imageFondAvecGoulotteBytes = await TraitementImage.incrusterGoulotteWorker({
            'imageDeFondBytes': _imageFondPropreBytes!, // Toujours dessiné sur le mur propre !
            'ptDepartX': _goulotteNotifier.value!.start.dx,
            'ptDepartY': _goulotteNotifier.value!.start.dy,
            'ptArriveeX': _goulotteNotifier.value!.end.dx,
            'ptArriveeY': _goulotteNotifier.value!.end.dy,
            'largeurPx': largeurGoulotteOrig, 
          });
        } else {
          _imageFondAvecGoulotteBytes = null;
        }
      }

      // 2. GÉNÉRATION DU CALQUE DE L'ÉQUIPEMENT EN PNG (Seulement si demandé ou pas en cache)
      if (recomputeEquipement || _calqueEquipementPngBytes == null) {
        setState(() => _loadingMessage = "Calcul des ombres de l'équipement...");
        
        _calqueEquipementPngBytes = await TraitementImage.genererCalqueEquipementWorker({
          'fondPropreBytes': _imageFondPropreBytes!, // Toujours calculée d'après le mur propre !
          'equipementBytes': equipementBytes,
          'pointsIA': _pointsCibles!,
          'decalageX': _decalageNotifier.value.dx,
          'decalageY': _decalageNotifier.value.dy,
          'equipementAssetPath': equipementPath,
          'profondeurMm': profondeur,
          'hauteurMm': hauteur,
          'largeurMm': largeur,
        });

        // --- NOUVEAU : SAUVEGARDE DU DERNIER RENDU COMPLET (NON ROGNÉ) ---
        // On calcule la taille et la position de la clim pour vérifier si elle sort de l'écran
        double equipementWPxOrig = (largeur / 50.0) * autoWPxOrig;
        double equipementHPxOrig = equipementWPxOrig * (hauteur / largeur);
        
        double eqX = ptHgXOrig + _decalageNotifier.value.dx;
        double eqY = ptHgYOrig + _decalageNotifier.value.dy;
        
        // On utilise une marge de 50 pixels pour être sûr que l'ombre et l'extrusion soient entièrement sur la photo
        bool isCropped = (eqX < 50 || eqY < 50 || 
                         (eqX + equipementWPxOrig + 100) > _imageWidth! || 
                         (eqY + equipementHPxOrig + 100) > _imageHeight!);

        // Si la machine n'est PAS rognée, on la sauvegarde en tant qu'image "parfaite" pour le glissement !
        // Si elle l'est, on garde notre ancienne image parfaite en mémoire pour éviter le trou visuel.
        if (!isCropped || _dernierCalqueEquipementCompletBytes == null) {
           _dernierCalqueEquipementCompletBytes = _calqueEquipementPngBytes;
           _decalageDuCalqueComplet = _decalageNotifier.value;
        }
      }

      // 3. FUSION INSTANTANÉE DES DEUX CALQUES
      setState(() => _loadingMessage = "Assemblage final...");
      Uint8List fondBase = _imageFondAvecGoulotteBytes ?? _imageFondPropreBytes!;

      Uint8List? resultImage = await TraitementImage.fusionnerCalqueWorker({
        'fondBytes': fondBase,
        'calquePngBytes': _calqueEquipementPngBytes!
      });

      if (resultImage != null) {
        setState(() {
          _imageResultatBytes = resultImage;
          _splitNotifier.value = 1.0;
        });
      } else {
        throw Exception("L'image fusionnée est nulle.");
      }
    } catch (e) {
      print("[UI/OpenCV - ERREUR] Exception : $e");
      if (mounted) {
        _montrerErreur("Erreur lors de la génération 3D de l'équipement.");
      }
    } finally {
      if (mounted) {
        setState(() => _isProcessing = false);
      }
    }
  }

  // Méthode globale qui annule intelligemment le dernier mouvement selon l'outil actif.
  void _reinitialiserPosition() {
    if (_isProcessing) return;

    if (_isDrawGoulotteMode) {
      // Mode Goulotte : On annule le mouvement et on revient à la ligne d'origine
      if (_goulotteNotifier.value != null && _goulotteInitiale != null) {
        _goulotteNotifier.value = _goulotteInitiale;
        _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
      }
    } else {
      // CORRECTION : Mode Equipement -> On dépile le dernier élément pour faire un Undo étape par étape
      if (_historiqueDecalages.length > 1) {
        _historiqueDecalages.removeLast(); // Retire la position courante
        _decalageNotifier.value = _historiqueDecalages.last; // Applique la précédente
        _historiqueLengthNotifier.value = _historiqueDecalages.length; // OPTIMISATION : Déclenche la MAJ UI de manière ciblée
        _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true);
      }
    }
  }

  // NOUVEAU : Méthode pour tout remettre à zéro (Position initiale calculée par l'IA)
  void _resetPositionEquipement() {
    if (_isProcessing) return;

    setState(() {
      _decalageNotifier.value = Offset.zero;
      _historiqueDecalages.clear();
      _historiqueDecalages.add(Offset.zero);
      _historiqueLengthNotifier.value = 1; // Cache les boutons undo/reset
    });

    // On recalcule le PNG à la position zéro
    _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true);
  }

  Future<void> _sauvegarderImage() async {
    if (_imageResultatBytes == null) return;

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Sauvegarde en cours...'), duration: Duration(milliseconds: 500)),
    );

    try {
      final result = await ImageGallerySaverPlus.saveImage(
        _imageResultatBytes!,
        quality: 100,
        name: "Devis_Simulation_${DateTime.now().millisecondsSinceEpoch}", 
      );
      
      if (!mounted) return;
      
      if (result['isSuccess'] == true) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Simulation sauvegardée dans la galerie'), backgroundColor: Colors.green, duration: Duration(seconds: 3)),
        );
      } else {
        throw Exception("Échec de la sauvegarde.");
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Erreur lors de la sauvegarde.'), backgroundColor: Colors.red));
    }
  }

  // =========================================================================
  // === UI : DÉCOUPAGE EN WIDGETS (POUR ALLÉGER LE BUILD) ===
  // =========================================================================

  Widget _buildCalquePlacementManuel(double scale, double offsetX, double offsetY, ThemeData theme) {
    List<Offset> screenPoints = _pointsCibles!.map((p) {
      double pxOrig = p['x']! * (_imageWidth! / 1024.0);
      double pyOrig = p['y']! * (_imageHeight! / 1024.0);
      return Offset(pxOrig * scale + offsetX, pyOrig * scale + offsetY);
    }).toList();

    double minX = screenPoints.map((p) => p.dx).reduce(math.min);
    double maxX = screenPoints.map((p) => p.dx).reduce(math.max);
    double minY = screenPoints.map((p) => p.dy).reduce(math.min);
    double maxY = screenPoints.map((p) => p.dy).reduce(math.max);

    // COMPENSATON DYNAMIQUE DU ZOOM : On écoute le contrôleur d'InteractiveViewer
    return ValueListenableBuilder<Matrix4>(
      valueListenable: _transformationController,
      builder: (context, matrix, _) {
        
        // Calcule à quel point l'utilisateur a zoomé (ex: 2.0 pour x2)
        double currentZoom = matrix.getMaxScaleOnAxis();
        if (currentZoom <= 0) currentZoom = 1.0;
        
        // Le ratio inverse : si on zoome x4, on doit dessiner x0.25 pour que ça reste de la même taille physique !
        double invZoom = 1.0 / currentZoom;

        return Stack(
          children: [
            Positioned.fill(
              child: Hero(
                tag: 'image_mur',
                child: Image.file(File(widget.photoPath), fit: BoxFit.contain),
              )
            ),
            
            // Le dessinateur va utiliser currentZoom pour diviser l'épaisseur du pinceau
            Positioned.fill(
              child: CustomPaint(
                painter: BoundingBoxPainter(
                  points: screenPoints, 
                  primaryColor: Colors.blueAccent, // FORCE LE BLEU (Meilleur contraste)
                  zoomScale: currentZoom, // On passe le zoom !
                )
              )
            ),
            
            // DÉPLACEMENT CENTRAL DU RECTANGLE (Garde toujours sa forme 90°)
            Positioned(
              left: minX,
              top: minY,
              width: maxX - minX,
              height: maxY - minY,
              child: GestureDetector(
                behavior: HitTestBehavior.opaque,
                onPanStart: (_) {
                  // Si l'utilisateur commence à bouger la boîte, on cache la question de l'IA
                  if (_attenteConfirmationIA) {
                    setState(() {
                      _attenteConfirmationIA = false;
                      
                      // ALIGNEMENT STRICT : On convertit la forme de l'IA en VRAI RECTANGLE
                      double minX = _pointsCibles!.map((p) => p['x']!).reduce(math.min);
                      double maxX = _pointsCibles!.map((p) => p['x']!).reduce(math.max);
                      double minY = _pointsCibles!.map((p) => p['y']!).reduce(math.min);
                      double maxY = _pointsCibles!.map((p) => p['y']!).reduce(math.max);
                      
                      _pointsCibles = [
                        {'x': minX, 'y': minY}, // Haut Gauche
                        {'x': maxX, 'y': minY}, // Haut Droit
                        {'x': maxX, 'y': maxY}, // Bas Droit
                        {'x': minX, 'y': maxY}, // Bas Gauche
                      ];
                    });
                  }
                },
                onPanUpdate: (details) {
                  // Sécurité : on ignore le déplacement si plusieurs doigts sont détectés
                  if (_activePointers > 1) return;
                  
                  setState(() {
                    double dxOrig = details.delta.dx / scale;
                    double dyOrig = details.delta.dy / scale;
                    double dx1024 = dxOrig * (1024.0 / _imageWidth!);
                    double dy1024 = dyOrig * (1024.0 / _imageHeight!);
                    
                    bool canMove = true;
                    for (var p in _pointsCibles!) {
                      double newX = p['x']! + dx1024;
                      double newY = p['y']! + dy1024;
                      if (newX < 0 || newX > 1024 || newY < 0 || newY > 1024) {
                        canMove = false;
                        break;
                      }
                    }
                    
                    if (canMove) {
                      for (int i = 0; i < 4; i++) {
                        _pointsCibles![i]['x'] = _pointsCibles![i]['x']! + dx1024;
                        _pointsCibles![i]['y'] = _pointsCibles![i]['y']! + dy1024;
                      }
                    }
                  });
                },
                child: Container(color: Colors.transparent),
              ),
            ),

            ...screenPoints.asMap().entries.map((entry) {
              int idx = entry.key;
              Offset pt = entry.value;
              
              // On garde la taille physique constante à l'écran, peu importe le zoom !
              double hitBoxSize = 48.0 * invZoom; // Zone tactile généreuse mais invisible (Avant: 40)
              double visualNodeSize = 16.0 * invZoom; // GROS NOEUD BLEU BIEN VISIBLE (Avant: 6.0)
              double offsetCenter = hitBoxSize / 2.0;

              return Positioned(
                left: pt.dx - offsetCenter, 
                top: pt.dy - offsetCenter,  
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque, 
                  onPanStart: (_) {
                    // Si l'utilisateur attrape un point, on cache la question de l'IA
                    if (_attenteConfirmationIA) {
                      setState(() {
                        _attenteConfirmationIA = false;
                        
                        // ALIGNEMENT STRICT : Dès qu'on touche un coin, ça devient un rectangle parfait
                        double minX = _pointsCibles!.map((p) => p['x']!).reduce(math.min);
                        double maxX = _pointsCibles!.map((p) => p['x']!).reduce(math.max);
                        double minY = _pointsCibles!.map((p) => p['y']!).reduce(math.min);
                        double maxY = _pointsCibles!.map((p) => p['y']!).reduce(math.max);
                        
                        _pointsCibles = [
                          {'x': minX, 'y': minY}, // Haut Gauche
                          {'x': maxX, 'y': minY}, // Haut Droit
                          {'x': maxX, 'y': maxY}, // Bas Droit
                          {'x': minX, 'y': maxY}, // Bas Gauche
                        ];
                      });
                    }
                  },
                  onPanUpdate: (details) {
                    // Sécurité multi-touch
                    if (_activePointers > 1) return;
                    
                    setState(() {
                      double dxOrig = details.delta.dx / scale;
                      double dyOrig = details.delta.dy / scale;
                      double dx1024 = dxOrig * (1024.0 / _imageWidth!);
                      double dy1024 = dyOrig * (1024.0 / _imageHeight!);
                      
                      double newX = (_pointsCibles![idx]['x']! + dx1024).clamp(0.0, 1024.0);
                      double newY = (_pointsCibles![idx]['y']! + dy1024).clamp(0.0, 1024.0);

                      // CORRECTION MAJEURE : COMPORTEMENT "RECTANGLE CLASSIQUE PAINT"
                      // Tirer un coin modifie automatiquement ses voisins pour garder les angles à 90° !
                      if (idx == 0) { // Haut-Gauche
                        if (newX >= _pointsCibles![1]['x']! - 10) newX = _pointsCibles![1]['x']! - 10;
                        if (newY >= _pointsCibles![3]['y']! - 10) newY = _pointsCibles![3]['y']! - 10;
                        _pointsCibles![0]['x'] = newX;
                        _pointsCibles![0]['y'] = newY;
                        _pointsCibles![1]['y'] = newY; // Aligne Haut-Droit
                        _pointsCibles![3]['x'] = newX; // Aligne Bas-Gauche
                      } else if (idx == 1) { // Haut-Droit
                        if (newX <= _pointsCibles![0]['x']! + 10) newX = _pointsCibles![0]['x']! + 10;
                        if (newY >= _pointsCibles![2]['y']! - 10) newY = _pointsCibles![2]['y']! - 10;
                        _pointsCibles![1]['x'] = newX;
                        _pointsCibles![1]['y'] = newY;
                        _pointsCibles![0]['y'] = newY; // Aligne Haut-Gauche
                        _pointsCibles![2]['x'] = newX; // Aligne Bas-Droit
                      } else if (idx == 2) { // Bas-Droit
                        if (newX <= _pointsCibles![3]['x']! + 10) newX = _pointsCibles![3]['x']! + 10;
                        if (newY <= _pointsCibles![1]['y']! + 10) newY = _pointsCibles![1]['y']! + 10;
                        _pointsCibles![2]['x'] = newX;
                        _pointsCibles![2]['y'] = newY;
                        _pointsCibles![3]['y'] = newY; // Aligne Bas-Gauche
                        _pointsCibles![1]['x'] = newX; // Aligne Haut-Droit
                      } else if (idx == 3) { // Bas-Gauche
                        if (newX >= _pointsCibles![2]['x']! - 10) newX = _pointsCibles![2]['x']! - 10;
                        if (newY <= _pointsCibles![0]['y']! + 10) newY = _pointsCibles![0]['y']! + 10;
                        _pointsCibles![3]['x'] = newX;
                        _pointsCibles![3]['y'] = newY;
                        _pointsCibles![2]['y'] = newY; // Aligne Bas-Droit
                        _pointsCibles![0]['x'] = newX; // Aligne Haut-Gauche
                      }
                    });
                  },
                  child: Container(
                    width: hitBoxSize, // Zone tactile invisible plus grande
                    height: hitBoxSize, 
                    color: Colors.transparent, 
                    alignment: Alignment.center,
                    child: Container(
                      width: visualNodeSize, // Le carré est plus gros 
                      height: visualNodeSize, 
                      decoration: BoxDecoration(
                        color: Colors.blueAccent.withValues(alpha: 0.6), // FORCE LE BLEU
                        shape: BoxShape.rectangle, // CORRECTION : LOOK CARRE CLASSIQUE DE PAINT !
                        border: Border.all(color: Colors.blueAccent, width: 2.0 * invZoom), 
                      ),
                    ),
                  ),
                ),
              );
            }),
          ],
        );
      }
    );
  }

  // Construit les zones tactiles (nœuds) pour éditer une goulotte existante sans bloquer le zoom
  List<Widget> _buildGoulotteDraggers(LigneGoulotte goulotte, double scale, double offsetX, double offsetY) {
    Offset p1 = Offset(goulotte.start.dx * scale + offsetX, goulotte.start.dy * scale + offsetY);
    Offset p2 = Offset(goulotte.end.dx * scale + offsetX, goulotte.end.dy * scale + offsetY);
    double len = (p2 - p1).distance;
    double angle = math.atan2(p2.dy - p1.dy, p2.dx - p1.dx);

    return [
      // 1. Noeud de déplacement complet (Corps de la goulotte)
      Positioned(
        left: p1.dx,
        top: p1.dy - 20, // Décale pour centrer le hitbox sur la ligne
        child: Transform.rotate(
          angle: angle,
          alignment: Alignment.centerLeft,
          child: GestureDetector(
            behavior: HitTestBehavior.translucent,
            onPanStart: (_) {
              if (_activePointers > 1) return;
              _isDraggingGoulotteNotifier.value = true;
              // On ne déclenche pas la loupe pour le déplacement du corps entier
            },
            onPanUpdate: (details) {
              if (_activePointers > 1 || !_isDraggingGoulotteNotifier.value) return;
              
              // CORRECTION DU BUG DE ROTATION : 
              // Transform.rotate altère le repère local du GestureDetector.
              // On ré-oriente le "delta" pour retrouver les vraies coordonnées globales de l'écran.
              double cosA = math.cos(angle);
              double sinA = math.sin(angle);
              double globalDx = details.delta.dx * cosA - details.delta.dy * sinA;
              double globalDy = details.delta.dx * sinA + details.delta.dy * cosA;
              Offset deltaOrig = Offset(globalDx, globalDy) / scale;
              
              var g = _goulotteNotifier.value!;
              Offset newStart = g.start + deltaOrig;
              Offset newEnd = g.end + deltaOrig;

              // Sécurité pour empêcher le corps entier de sortir de l'image
              double minX = math.min(newStart.dx, newEnd.dx);
              double maxX = math.max(newStart.dx, newEnd.dx);
              double minY = math.min(newStart.dy, newEnd.dy);
              double maxY = math.max(newStart.dy, newEnd.dy);
              
              double adjustDx = 0;
              double adjustDy = 0;
              
              if (minX < 0) adjustDx = -minX;
              if (maxX > _imageWidth!) adjustDx = _imageWidth! - maxX;
              if (minY < 0) adjustDy = -minY;
              if (maxY > _imageHeight!) adjustDy = _imageHeight! - maxY;
              
              newStart = Offset(newStart.dx + adjustDx, newStart.dy + adjustDy);
              newEnd = Offset(newEnd.dx + adjustDx, newEnd.dy + adjustDy);
              
              _goulotteNotifier.value = LigneGoulotte(newStart, newEnd);
            },
            onPanEnd: (_) {
              if (!_isDraggingGoulotteNotifier.value) return;
              _isDraggingGoulotteNotifier.value = false;
              // Seule la goulotte a bougé !
              _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
            },
            onPanCancel: () {
              if (!_isDraggingGoulotteNotifier.value) return;
              _isDraggingGoulotteNotifier.value = false;
              _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
            },
            child: Container(width: len, height: 40, color: Colors.transparent),
          ),
        ),
      ),
      // 2. Noeud de redimensionnement de Départ
      Positioned(
        left: p1.dx - 25,
        top: p1.dy - 25,
        child: GestureDetector(
          behavior: HitTestBehavior.translucent,
          onPanStart: (_) {
            if (_activePointers > 1) return;
            _isDraggingGoulotteNotifier.value = true;
            
            // Déclenche la loupe sur ce nœud
            _magnifierPositionNotifier.value = Offset(goulotte.start.dx * scale + offsetX, goulotte.start.dy * scale + offsetY);
          },
          onPanUpdate: (details) {
            if (_activePointers > 1 || !_isDraggingGoulotteNotifier.value) return;
            Offset deltaOrig = details.delta / scale;
            var g = _goulotteNotifier.value!;
            Offset rawStart = g.start + deltaOrig;
            
            // Clamper pour ne pas sortir des limites de l'image
            rawStart = _clampToImageBounds(rawStart);
            
            // On force l'alignement rectiligne parfait (horizontal ou vertical)
            Offset snappedStart = _snapToOrthogonal(g.end, rawStart);
            
            // Re-clamp par sécurité au cas où le snap pousserait le point dehors
            _goulotteNotifier.value = LigneGoulotte(_clampToImageBounds(snappedStart), g.end);
            
            // Mise à jour de la loupe
            _magnifierPositionNotifier.value = Offset(snappedStart.dx * scale + offsetX, snappedStart.dy * scale + offsetY);
          },
          onPanEnd: (_) {
            _magnifierPositionNotifier.value = null; // Cache la loupe
            if (!_isDraggingGoulotteNotifier.value) return;
            _isDraggingGoulotteNotifier.value = false;
            _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
          },
          onPanCancel: () {
            _magnifierPositionNotifier.value = null; // Cache la loupe
            if (!_isDraggingGoulotteNotifier.value) return;
            _isDraggingGoulotteNotifier.value = false;
            _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
          },
          child: Container(width: 50, height: 50, color: Colors.transparent),
        ),
      ),
      // 3. Noeud de redimensionnement de Fin
      Positioned(
        left: p2.dx - 25,
        top: p2.dy - 25,
        child: GestureDetector(
          behavior: HitTestBehavior.translucent,
          onPanStart: (_) {
            if (_activePointers > 1) return;
            _isDraggingGoulotteNotifier.value = true;
            
            // Déclenche la loupe sur ce nœud
            _magnifierPositionNotifier.value = Offset(goulotte.end.dx * scale + offsetX, goulotte.end.dy * scale + offsetY);
          },
          onPanUpdate: (details) {
            if (_activePointers > 1 || !_isDraggingGoulotteNotifier.value) return;
            Offset deltaOrig = details.delta / scale;
            var g = _goulotteNotifier.value!;
            Offset rawEnd = g.end + deltaOrig;
            
            // Clamper pour ne pas sortir des limites de l'image
            rawEnd = _clampToImageBounds(rawEnd);
            
            // On force l'alignement rectiligne parfait (horizontal ou vertical)
            Offset snappedEnd = _snapToOrthogonal(g.start, rawEnd);
            
            // Re-clamp par sécurité
            _goulotteNotifier.value = LigneGoulotte(g.start, _clampToImageBounds(snappedEnd));
            
            // Mise à jour de la loupe
            _magnifierPositionNotifier.value = Offset(snappedEnd.dx * scale + offsetX, snappedEnd.dy * scale + offsetY);
          },
          onPanEnd: (_) {
            _magnifierPositionNotifier.value = null; // Cache la loupe
            if (!_isDraggingGoulotteNotifier.value) return;
            _isDraggingGoulotteNotifier.value = false;
            _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
          },
          onPanCancel: () {
            _magnifierPositionNotifier.value = null; // Cache la loupe
            if (!_isDraggingGoulotteNotifier.value) return;
            _isDraggingGoulotteNotifier.value = false;
            _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
          },
          child: Container(width: 50, height: 50, color: Colors.transparent),
        ),
      ),
    ];
  }

  Widget _buildCalqueResultat(double scale, double offsetX, double offsetY, BoxConstraints constraints, ThemeData theme) {
    double ptHgXOrig = _pointsCibles![0]['x']! * (_imageWidth! / 1024.0);
    double ptHgYOrig = _pointsCibles![0]['y']! * (_imageHeight! / 1024.0);
    double ptHdXOrig = _pointsCibles![1]['x']! * (_imageWidth! / 1024.0);
    double ptHdYOrig = _pointsCibles![1]['y']! * (_imageHeight! / 1024.0);

    double dx = ptHdXOrig - ptHgXOrig;
    double dy = ptHdYOrig - ptHgYOrig;
    double autoWPxOrig = math.sqrt(dx * dx + dy * dy);
    
    // On récupère les dimensions depuis le catalogue pour le calcul d'affichage UI
    double largeurMm = 798.0; 
    double hauteurMm = 270.0;
    if (_modeleSelectionne != null) {
      largeurMm = _modeleSelectionne!.largeur;
      hauteurMm = _modeleSelectionne!.hauteur;
    }

    double equipementWPxOrig = (largeurMm / 50.0) * autoWPxOrig;
    double equipementHPxOrig = equipementWPxOrig * (hauteurMm / largeurMm);

    double equipementScreenW = equipementWPxOrig * scale;
    double equipementScreenH = equipementHPxOrig * scale;

    double angleRad = math.atan2(dy, dx);
    
    // Calcul de l'épaisseur pour le Painter
    double ratioPxParMm = autoWPxOrig / 50.0;
    double largeurGoulotteOrig = 80.0 * ratioPxParMm; 

    return Stack(
      children: [
        // 1. Couche de Fond : Mur Propre OU Mur avec OpenCV Goulotte
        Positioned.fill(
          child: ValueListenableBuilder<bool>( // Écoute le drag de l'équipement
            valueListenable: _isDraggingEquipementNotifier,
            builder: (context, isDraggingEquipement, _) {
              return ValueListenableBuilder<bool>( // Écoute le drag de la goulotte
                valueListenable: _isDraggingGoulotteNotifier,
                builder: (context, isDraggingGoulotte, _) {
                  Uint8List? imgToShow;
                  if (isDraggingGoulotte) {
                    // Quand on déplace la goulotte, on affiche le mur propre en fond.
                    imgToShow = _imageFondPropreBytes;
                  } else {
                    // Sinon (déplacement equipement ou repos), le fond est le mur avec la goulotte (si tracée).
                    imgToShow = _imageFondAvecGoulotteBytes ?? _imageFondPropreBytes;
                  }
                  if (imgToShow == null) {
                    return Hero(
                      tag: 'image_mur',
                      child: Image.file(File(widget.photoPath), fit: BoxFit.contain)
                    );
                  }
                  return Hero(
                    tag: 'image_mur',
                    child: Image.memory(imgToShow, fit: BoxFit.contain, gaplessPlayback: true)
                  );
                }
              );
            }
          ),
        ),

        // 2. Couche OpenCV Résultat Complet (Cachée pendant TOUT glissement)
        if (_imageResultatBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: _isDraggingEquipementNotifier,
            builder: (context, isDraggingEquipement, _) {
              return ValueListenableBuilder<bool>(
                valueListenable: _isDraggingGoulotteNotifier,
                builder: (context, isDraggingGoulotte, _) {
                  // Si l'utilisateur touche une pièce, on cache le rendu final
                  if (isDraggingEquipement || isDraggingGoulotte) return const SizedBox.shrink(); 
                  
                  return ValueListenableBuilder<double>(
                    valueListenable: _splitNotifier,
                    builder: (context, splitVal, _) {
                      double currentSplit = _isDrawGoulotteMode ? 1.0 : splitVal;
                      return Positioned.fill(
                        child: ClipRect(
                          clipper: SplitClipper(currentSplit),
                          child: Image.memory(_imageResultatBytes!, fit: BoxFit.contain, gaplessPlayback: true), 
                        ),
                      );
                    }
                  );
                }
              );
            }
          ),

        // =========================================================================
        // === CORRECTION : AFFICHE L'ÉQUIPEMENT RENDU LORS DU DRAG DE LA GOULOTTE =
        // =========================================================================
        if (_calqueEquipementPngBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: _isDraggingGoulotteNotifier,
            builder: (context, isDraggingGoulotte, _) {
              if (isDraggingGoulotte) {
                // Quand on déplace la goulotte, on montre le calque PNG parfait de la clim sur le mur
                return Positioned.fill(
                  child: Image.memory(_calqueEquipementPngBytes!, fit: BoxFit.contain, gaplessPlayback: true),
                );
              }
              return const SizedBox.shrink();
            }
          ),

        // Le painter global de la goulotte
        Positioned.fill(
          child: IgnorePointer(
            child: ValueListenableBuilder<LigneGoulotte?>(
              valueListenable: _goulotteNotifier,
              builder: (context, goulotte, _) {
                return ValueListenableBuilder<bool>(
                  valueListenable: _isDraggingEquipementNotifier, // Etat du drag équipement
                  builder: (context, isDraggingEquipement, _) {
                    return ValueListenableBuilder<bool>(
                      valueListenable: _isDraggingGoulotteNotifier, // Etat du drag goulotte
                      builder: (context, isDraggingGoulotte, _) {
                        
                        // CORRECTION : On n'affiche le trait vectoriel QUE lorsqu'on déplace la goulotte !
                        // Sinon (si on déplace la clim), on verra la magnifique goulotte rendue par OpenCV en fond.
                        bool showLine = isDraggingGoulotte;
                        // On affiche les ronds bleus (nœuds) uniquement si l'outil pinceau est activé
                        bool showNodes = _isDrawGoulotteMode;

                        return CustomPaint(
                          painter: GoulottePainter(
                            goulotte: goulotte,
                            scale: scale,
                            offsetX: offsetX,
                            offsetY: offsetY,
                            thicknessOrig: largeurGoulotteOrig,
                            showLine: showLine,
                            showNodes: showNodes,
                            primaryColor: theme.colorScheme.primary, // Injecte la couleur de la marque
                          ),
                        );
                      }
                    );
                  }
                );
              }
            )
          )
        ),

        // 3. Curseur du slider avant/après
        if (_imageResultatBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: _isDraggingEquipementNotifier,
            builder: (context, isDragging, _) {
              // On masque la barre de slider si l'utilisateur trace une goulotte ou drag
              if (isDragging || _isDrawGoulotteMode) return const SizedBox.shrink();
              return ValueListenableBuilder<double>(
                valueListenable: _splitNotifier,
                builder: (context, splitVal, _) {
                  return Positioned(
                    key: const ValueKey('slider_interactif'),
                    top: 0,
                    bottom: 0,
                    left: (constraints.maxWidth * splitVal) - 20, 
                    child: GestureDetector(
                      behavior: HitTestBehavior.opaque,
                      onHorizontalDragUpdate: (details) {
                        // Anti-Missclick pendant le zoom
                        if (_activePointers > 1) return;
                        _splitNotifier.value = (_splitNotifier.value + details.delta.dx / constraints.maxWidth).clamp(0.0, 1.0);
                      },
                      child: SizedBox(
                        width: 40, 
                        child: Stack(
                          alignment: Alignment.center, 
                          children: [
                            // STYLE : La ligne blanche a un effet Glow subtil
                            Container(
                              width: 4, 
                              decoration: BoxDecoration(
                                color: Colors.white,
                                boxShadow: [BoxShadow(color: theme.colorScheme.primary.withValues(alpha: 0.5), blurRadius: 8, spreadRadius: 1)]
                              ),
                            ),
                            Positioned(
                              bottom: 20, 
                              child: Container(
                                height: 50, // Forme de pilule verticale
                                width: 30,
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(20),
                                  boxShadow: const [BoxShadow(color: Colors.black38, blurRadius: 8, spreadRadius: 1)]
                                ),
                                child: Icon(Icons.compare_arrows, size: 20, color: theme.colorScheme.primary), // Adapté au thème
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  );
                }
              );
            }
          ),

        // 4. L'image brute de l'Équipement (qui s'affiche en transparence UNIQUEMENT pendant un drag)
        if (_modeleSelectionne != null)
          ValueListenableBuilder<Offset>(
            valueListenable: _decalageNotifier,
            builder: (context, decalage, _) {
              return ValueListenableBuilder<bool>(
                valueListenable: _isDraggingEquipementNotifier,
                builder: (context, isDraggingEquipement, _) {
                  double equipementScreenX = (ptHgXOrig + decalage.dx) * scale + offsetX;
                  double equipementScreenY = (ptHgYOrig + decalage.dy) * scale + offsetY;
                  
                  // OPTIMISATION DRAG : On calcule le vecteur de translation par rapport au dernier rendu complet !
                  Uint8List? dragImage = _dernierCalqueEquipementCompletBytes ?? _calqueEquipementPngBytes;
                  Offset refDecalage = _dernierCalqueEquipementCompletBytes != null 
                      ? _decalageDuCalqueComplet 
                      : (_historiqueDecalages.isNotEmpty ? _historiqueDecalages.last : Offset.zero);
                      
                  double diffX = (decalage.dx - refDecalage.dx) * scale;
                  double diffY = (decalage.dy - refDecalage.dy) * scale;

                  return Stack(
                    children: [
                      // NOUVEAU : Rendu complet 3D en mouvement au lieu du PNG plat !
                      if (isDraggingEquipement && dragImage != null)
                        Positioned.fill(
                          child: IgnorePointer(
                            child: Transform.translate(
                              offset: Offset(diffX, diffY),
                              child: Opacity(
                                opacity: 0.85, // Légèrement transparent pour voir où on le pose
                                child: Image.memory(dragImage, fit: BoxFit.contain, gaplessPlayback: true),
                              ),
                            ),
                          ),
                        ),

                      // Zone tactile transparente (Hitbox de déplacement invisible)
                      Positioned(
                        key: const ValueKey('equipement_draggable'),
                        left: equipementScreenX,
                        top: equipementScreenY,
                        width: equipementScreenW,
                        height: equipementScreenH,
                        child: IgnorePointer(
                          ignoring: _isDrawGoulotteMode,
                          child: GestureDetector(
                            behavior: HitTestBehavior.translucent,
                            onPanStart: (_) {
                              if (_activePointers > 1) return;
                              _isDraggingEquipementNotifier.value = true;
                            },
                            onPanUpdate: (details) { 
                               if (_activePointers > 1 || !_isDraggingEquipementNotifier.value) return;
                               
                               double cosA = math.cos(angleRad);
                               double sinA = math.sin(angleRad);
                               double globalDx = details.delta.dx * cosA - details.delta.dy * sinA;
                               double globalDy = details.delta.dx * sinA + details.delta.dy * cosA;

                               _decalageNotifier.value = Offset(
                                 _decalageNotifier.value.dx + globalDx / scale,
                                 _decalageNotifier.value.dy + globalDy / scale
                               );
                            },
                            onPanEnd: (_) { 
                               if (!_isDraggingEquipementNotifier.value) return;
                               _isDraggingEquipementNotifier.value = false;

                               // CORRECTION : Enregistrement de la nouvelle coordonnée validée dans la pile d'historique
                               if (_historiqueDecalages.isEmpty || _historiqueDecalages.last != _decalageNotifier.value) {
                                 _historiqueDecalages.add(_decalageNotifier.value);
                                 _historiqueLengthNotifier.value = _historiqueDecalages.length; // OPTIMISATION : Demande la MAJ du bouton Undo
                               }

                               // On ne re-calcule que l'Equipement ! C'est ce qui fait gagner du temps.
                               _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true); 
                             },
                            onPanCancel: () { 
                               if (!_isDraggingEquipementNotifier.value) return;
                               _isDraggingEquipementNotifier.value = false;
                               _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true); 
                             },
                            child: Transform.rotate(
                              angle: angleRad,
                              alignment: Alignment.topLeft, 
                              // CORRECTION : Le PNG brut plat est maintenant à 0.0 en permanence.
                              // Il sert juste de "boîte de collision" pour les gestes tactiles !
                              child: Opacity(
                                opacity: 0.0, 
                                child: Image.asset(_modeleSelectionne!.chemin, fit: BoxFit.fill),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ]
                  );
                }
              );
            }
          ),

        // Calque interactif invisible avec système Nodal pour la Goulotte
        if (_isDrawGoulotteMode)
          Positioned.fill(
            child: ValueListenableBuilder<LigneGoulotte?>(
              valueListenable: _goulotteNotifier,
              builder: (context, goulotte, _) {
                // La couche d'interaction (Création OU Édition)
                if (goulotte == null) {
                  // A - Mode plein écran pour tracer la toute première goulotte
                  return GestureDetector(
                    behavior: HitTestBehavior.translucent,
                    onPanStart: (details) {
                      if (_activePointers > 1) return;
                      // On garde la trace initiale sans provoquer la disparition de la zone tactile
                      Offset touchOrig = (details.localPosition - Offset(offsetX, offsetY)) / scale;
                      
                      // Clamper le point de départ pour ne pas commencer en dehors
                      _goulotteStartOrig = _clampToImageBounds(touchOrig);
                      _goulotteCurrentEndOrigNotifier.value = _goulotteStartOrig;
                      _isDraggingGoulotteNotifier.value = true;
                      
                      // Affiche la loupe
                      _magnifierPositionNotifier.value = Offset(_goulotteStartOrig!.dx * scale + offsetX, _goulotteStartOrig!.dy * scale + offsetY);
                    },
                    onPanUpdate: (details) {
                      if (_activePointers > 1 || !_isDraggingGoulotteNotifier.value) return;
                      if (_goulotteStartOrig != null) {
                        Offset rawEnd = (details.localPosition - Offset(offsetX, offsetY)) / scale;
                        
                        // On clamp pour ne pas sortir de l'image
                        rawEnd = _clampToImageBounds(rawEnd);
                        
                        // Force le tracé à être parfaitement droit
                        Offset snappedEnd = _snapToOrthogonal(_goulotteStartOrig!, rawEnd);
                        
                        // Re-clamp au cas où le snap aurait poussé la ligne en dehors
                        _goulotteCurrentEndOrigNotifier.value = _clampToImageBounds(snappedEnd);
                        
                        // Met à jour la position de la loupe
                        _magnifierPositionNotifier.value = Offset(snappedEnd.dx * scale + offsetX, snappedEnd.dy * scale + offsetY);
                      }
                    },
                    onPanEnd: (_) {
                      _magnifierPositionNotifier.value = null; // Cache la loupe
                      if (!_isDraggingGoulotteNotifier.value) return;
                      if (_goulotteStartOrig != null && _goulotteCurrentEndOrigNotifier.value != null) {
                        _isDraggingGoulotteNotifier.value = false;
                        
                        // Sauvegarde de la goulotte pour le Reset et affectation finale
                        LigneGoulotte newGoulotte = LigneGoulotte(_goulotteStartOrig!, _goulotteCurrentEndOrigNotifier.value!);
                        _goulotteInitiale = newGoulotte; 
                        _goulotteNotifier.value = newGoulotte;
                        
                        _goulotteStartOrig = null;
                        _goulotteCurrentEndOrigNotifier.value = null;
                        // On trace la toute première goulotte
                        _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false);
                      }
                    },
                    onPanCancel: () {
                      _magnifierPositionNotifier.value = null; // Cache la loupe
                      if (!_isDraggingGoulotteNotifier.value) return;
                      _isDraggingGoulotteNotifier.value = false;
                      _goulotteStartOrig = null;
                      _goulotteCurrentEndOrigNotifier.value = null;
                    },
                    // Le painter temporaire pour afficher le trait PENDANT sa création
                    child: ValueListenableBuilder<Offset?>(
                      valueListenable: _goulotteCurrentEndOrigNotifier,
                      builder: (context, currentEnd, _) {
                        if (_goulotteStartOrig == null || currentEnd == null) return const SizedBox.shrink();
                        return CustomPaint(
                          painter: GoulottePainter(
                            goulotte: LigneGoulotte(_goulotteStartOrig!, currentEnd),
                            scale: scale,
                            offsetX: offsetX,
                            offsetY: offsetY,
                            thicknessOrig: largeurGoulotteOrig,
                            showLine: true, 
                            showNodes: true, // Affiche les nœuds pendant la création
                            primaryColor: theme.colorScheme.primary, // Injecte la couleur du thème
                          ),
                        );
                      },
                    ),
                  );
                } else {
                  // B - Mode interactif nodal (laisse 90% de l'écran libre pour le zoom !)
                  return Stack(
                    children: _buildGoulotteDraggers(goulotte, scale, offsetX, offsetY),
                  );
                }
              }
            ),
          ),

        // Le calque ultime au-dessus de tout : La Loupe de Précision (Magnifier)
        ValueListenableBuilder<Offset?>(
          valueListenable: _magnifierPositionNotifier,
          builder: (context, magPos, _) {
            // Si aucune position n'est définie (pas de drag), on ne montre rien
            if (magPos == null) return const SizedBox.shrink();
            
            // On centre la loupe horizontalement (magPos.dx - 60)
            // On positionne la loupe 130 pixels AU-DESSUS du doigt (pour ne pas être cachée)
            return Positioned(
              left: magPos.dx - 60, // 60 = moitié de la largeur de la loupe (120/2)
              top: magPos.dy - 130, // Décale vers le haut
              child: RawMagnifier(
                decoration: MagnifierDecoration(
                  shape: CircleBorder(
                    side: BorderSide(color: theme.colorScheme.primary, width: 2), // Bordure colorée avec le thème
                  ),
                  shadows: const [
                    BoxShadow(color: Colors.black26, blurRadius: 8, spreadRadius: 2)
                  ],
                ),
                size: const Size(120, 120),
                magnificationScale: 2.0,
                // Le point focal regarde 70 pixels plus bas que le centre de la loupe
                // Ce qui pointe EXACTEMENT sous le doigt de l'utilisateur !
                focalPointOffset: const Offset(0, 70),
              ),
            );
          }
        ),
      ],
    );
  }

  // =========================================================================
  // === BUILD PRINCIPAL ===
  // =========================================================================
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context); // NOUVEAU : Récupération du thème actif

    return Scaffold(
      appBar: AppBar(
        title: const Text('Configuration du Devis'),
        // L'AppBar utilise automatiquement les couleurs du thème global
      ),
      // Permet de compter le nombre de doigts à l'écran
      body: Listener(
        onPointerDown: (_) {
          _activePointers++;
          if (_activePointers > 1) {
            // Sécurité Anti-Missclick pendant le Zoom
            // On annule silencieusement tous les glissements en cours (equipement ou goulotte)
            _isDraggingEquipementNotifier.value = false;
            _isDraggingGoulotteNotifier.value = false;
            _goulotteStartOrig = null;
            _goulotteCurrentEndOrigNotifier.value = null;
            _magnifierPositionNotifier.value = null; // Cache la loupe
          }
        },
        onPointerUp: (_) {
          _activePointers = math.max(0, _activePointers - 1);
        },
        onPointerCancel: (_) {
          _activePointers = math.max(0, _activePointers - 1);
        },
        child: Column(
          children: [
            Expanded(
              child: Container(
                width: double.infinity,
                margin: const EdgeInsets.only(left: 16.0, right: 16.0, top: 16.0, bottom: 8.0),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [BoxShadow(color: theme.shadowColor.withValues(alpha: 0.1), blurRadius: 15, spreadRadius: 1)], // Ombre thématique
                ),
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(20),
                        // Écoute l'état de la goulotte pour réactiver intelligemment le Pan/Zoom
                        child: ValueListenableBuilder<LigneGoulotte?>(
                          valueListenable: _goulotteNotifier,
                          builder: (context, goulotte, _) {
                            // Si le mode Goulotte n'est pas actif, OU si une goulotte existe déjà (et utilise le drag nodale) : on autorise le zoom !
                            bool isPanZoomEnabled = !_isDrawGoulotteMode || goulotte != null;
                            
                            return InteractiveViewer(
                              transformationController: _transformationController, 
                              panEnabled: isPanZoomEnabled,
                              scaleEnabled: isPanZoomEnabled,
                              minScale: 1.0,
                              maxScale: 15.0, // Mis à 9.0 selon ta demande
                              child: LayoutBuilder(
                                builder: (context, constraints) {
                                  if (_imageWidth == null || _imageHeight == null) {
                                     return Hero(
                                       tag: 'image_mur',
                                       child: Image.file(File(widget.photoPath), fit: BoxFit.contain)
                                     );
                                  }

                                  if (_pointsCibles == null && !_isManualPlacementMode) {
                                     return Hero(
                                       tag: 'image_mur',
                                       child: Image.file(File(widget.photoPath), fit: BoxFit.contain)
                                     );
                                  }

                                  double scale = math.min(constraints.maxWidth / _imageWidth!, constraints.maxHeight / _imageHeight!);
                                  double offsetX = (constraints.maxWidth - (_imageWidth! * scale)) / 2;
                                  double offsetY = (constraints.maxHeight - (_imageHeight! * scale)) / 2;

                                  if (_isManualPlacementMode) {
                                    return _buildCalquePlacementManuel(scale, offsetX, offsetY, theme); // Injection du thème
                                  } else {
                                    return _buildCalqueResultat(scale, offsetX, offsetY, constraints, theme); // Injection du thème
                                  }
                                },
                              ),
                            );
                          }
                        ),
                      ),
                    ),
                    
                    // Colonne de boutons d'action au dessus de l'image
                    if (_modeleSelectionne != null && !_isManualPlacementMode)
                      Positioned(
                        top: 10,
                        right: 10,
                        child: BoutonsActionDevis(
                          isDraggingEquipementNotifier: _isDraggingEquipementNotifier,
                          isDraggingGoulotteNotifier: _isDraggingGoulotteNotifier,
                          historiqueLengthNotifier: _historiqueLengthNotifier,
                          goulotteNotifier: _goulotteNotifier,
                          goulotteInitiale: _goulotteInitiale,
                          isDrawGoulotteMode: _isDrawGoulotteMode,
                          isProcessing: _isProcessing,
                          onUndo: _reinitialiserPosition,
                          onResetPosition: _resetPositionEquipement, // NOUVEAU
                          onToggleGoulotteMode: () {
                            setState(() {
                              _isDrawGoulotteMode = !_isDrawGoulotteMode;
                              _isDraggingEquipementNotifier.value = false; 
                            });
                          },
                          onDeleteConfirmed: () {
                            _goulotteNotifier.value = null; // Vide la goulotte unique
                            _goulotteInitiale = null; // On vide aussi l'historique
                            // On force le recalcul uniquement de la goulotte (qui disparaît)
                            _genererIncrustation(recomputeGoulotte: true, recomputeEquipement: false); 
                          },
                        ),
                      ),

                    // Affichage du chargement par dessus tout (hors de l'InteractiveViewer pour ne pas être zoomé)
                    if (_isProcessing)
                      Positioned.fill(
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(20),
                          child: ValueListenableBuilder<bool>(
                            valueListenable: _isDraggingEquipementNotifier,
                            builder: (context, isDragging, _) {
                               if (isDragging) return const SizedBox.shrink(); 
                               return BackdropFilter(
                                filter: ImageFilter.blur(sigmaX: 5.0, sigmaY: 5.0),
                                child: Container(
                                  color: Colors.black.withValues(alpha: 0.4), // On garde le noir pour l'effet de verre fumé (même en light mode)
                                  child: Center(
                                    child: Column(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        const CircularProgressIndicator(color: Colors.white),
                                        const SizedBox(height: 20),
                                        Text(_loadingMessage, style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w500), textAlign: TextAlign.center),
                                      ],
                                    )
                                  )
                                ),
                              );
                            }
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),

            if (_pointsCibles != null && !_isManualPlacementMode)
              CatalogueDevis(
                categorieSelectionnee: _categorieSelectionnee,
                modeleSelectionne: _modeleSelectionne,
                isProcessing: _isProcessing,
                onCategorieChanged: (catName) {
                  setState(() => _categorieSelectionnee = catName);
                },
                onModeleSelected: (equipement) {
                  setState(() {
                    _modeleSelectionne = equipement;
                    
                    // NOUVEAU COMPORTEMENT : On ne reset plus la position !
                    // La nouvelle clim apparaîtra exactement là où on avait glissé l'ancienne.
                    // On vide juste le cache des calques pour obliger OpenCV à recalculer le PNG
                    _calqueEquipementPngBytes = null;
                    _dernierCalqueEquipementCompletBytes = null;

                    // En cas de changement de modèle, on ne recalcule QUE l'équipement !
                    _genererIncrustation(recomputeGoulotte: false, recomputeEquipement: true);
                  });
                },
              ),

            const SizedBox(height: 80), // Espace pour le FAB
          ],
        ),
      ),
      
      // =========================================================================
      // === LOGIQUE DES BOUTONS FLOTTANTS (EN BAS DE L'ÉCRAN) ===
      // =========================================================================
      floatingActionButton: _attenteConfirmationIA 
          // 1. CARTE DE CONFIRMATION DE L'IA (Remplaçant l'ancien AlertDialog bloquant)
          ? Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Card(
                elevation: 8,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                color: theme.cardColor,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text("Détection automatique", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: theme.colorScheme.onSurface)),
                      const SizedBox(height: 8),
                      Text("L'IA a détecté l'autocollant. Cette sélection vous convient-elle ?", style: TextStyle(color: theme.colorScheme.onSurface.withValues(alpha: 0.8)), textAlign: TextAlign.center),
                      const SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          TextButton(
                            onPressed: () {
                              setState(() {
                                _attenteConfirmationIA = false;
                                
                                // COMPORTEMENT RECTANGLE CLASSIQUE : On snap les points de l'IA pour 
                                // former un rectangle parfait avant l'édition manuelle !
                                double minX = _pointsCibles!.map((p) => p['x']!).reduce(math.min);
                                double maxX = _pointsCibles!.map((p) => p['x']!).reduce(math.max);
                                double minY = _pointsCibles!.map((p) => p['y']!).reduce(math.min);
                                double maxY = _pointsCibles!.map((p) => p['y']!).reduce(math.max);
                                
                                _pointsCibles = [
                                  {'x': minX, 'y': minY}, // Haut Gauche
                                  {'x': maxX, 'y': minY}, // Haut Droit
                                  {'x': maxX, 'y': maxY}, // Bas Droit
                                  {'x': minX, 'y': maxY}, // Bas Gauche
                                ];
                              });
                            },
                            child: Text("Ajuster", style: TextStyle(color: theme.colorScheme.onSurface.withValues(alpha: 0.6))),
                          ),
                          ElevatedButton.icon(
                            icon: const Icon(Icons.check, size: 18),
                            label: const Text("Oui, valider"),
                            style: ElevatedButton.styleFrom(backgroundColor: theme.colorScheme.primary, foregroundColor: theme.colorScheme.onPrimary),
                            onPressed: () {
                              setState(() {
                                _attenteConfirmationIA = false;
                              });
                              _validerPlacementManuel(); // IA validée, on passe à l'inpainting
                            },
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            )
          // 2. BOUTON DE VALIDATION DU MODE MANUEL (Ajustement humain)
          : _isManualPlacementMode
              ? FloatingActionButton.extended(
                  onPressed: _validerPlacementManuel,
                  label: const Text("Valider la position", style: TextStyle(fontWeight: FontWeight.bold)),
                  icon: const Icon(Icons.check),
                  // La couleur provient automatiquement de floatingActionButtonTheme
                )
          // 3. BOUTON DE SAUVEGARDE FINALE
              : (_imageResultatBytes != null && !_isProcessing)
                  ? FloatingActionButton.extended(
                      onPressed: _sauvegarderImage,
                      label: const Text("Sauvegarder", style: TextStyle(fontWeight: FontWeight.bold)),
                      icon: const Icon(Icons.download),
                      // La couleur provient automatiquement de floatingActionButtonTheme
                    )
                  : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}