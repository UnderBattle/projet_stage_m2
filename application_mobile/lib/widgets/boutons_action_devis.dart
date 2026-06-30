import 'dart:ui';
import 'package:flutter/material.dart';
import '../models/devis_models.dart';

class BoutonsActionDevis extends StatelessWidget {
  final ValueNotifier<bool> isDraggingEquipementNotifier;
  final ValueNotifier<bool> isDraggingGoulotteNotifier;
  final ValueNotifier<int> historiqueLengthNotifier;
  final ValueNotifier<int> historiqueRedoLengthNotifier; 
  final ValueNotifier<LigneGoulotte?> goulotteNotifier;
  final LigneGoulotte? goulotteInitiale;
  final bool isDrawGoulotteMode;
  final bool isProcessing;
  final VoidCallback onUndo;
  final VoidCallback onRedo; 
  final VoidCallback onResetPosition; 
  final VoidCallback onToggleGoulotteMode;
  final VoidCallback onDeleteConfirmed;

  const BoutonsActionDevis({
    super.key,
    required this.isDraggingEquipementNotifier,
    required this.isDraggingGoulotteNotifier,
    required this.historiqueLengthNotifier,
    required this.historiqueRedoLengthNotifier, 
    required this.goulotteNotifier,
    required this.goulotteInitiale,
    required this.isDrawGoulotteMode,
    required this.isProcessing,
    required this.onUndo,
    required this.onRedo, 
    required this.onResetPosition,
    required this.onToggleGoulotteMode,
    required this.onDeleteConfirmed,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context); // Récupération du thème

    return Column(
      // On aligne la colonne à droite pour éviter que l'apparition des boutons ne décale le reste
      crossAxisAlignment: CrossAxisAlignment.end, 
      children: [
        // Les boutons Historique (Undo et Reset) intelligents
        ValueListenableBuilder<bool>(
          valueListenable: isDraggingEquipementNotifier,
          builder: (context, isDraggingEquipement, _) {
            return ValueListenableBuilder<bool>(
              valueListenable: isDraggingGoulotteNotifier,
              builder: (context, isDraggingGoulotte, _) {
                
                // OPTIMISATION : On écoute uniquement la LONGUEUR de l'historique 
                // Les boutons ne se reconstruiront plus jamais pendant que le doigt bouge l'équipement !
                return ValueListenableBuilder<int>(
                  valueListenable: historiqueLengthNotifier,
                  builder: (context, historyLength, _) {
                    return ValueListenableBuilder<int>( 
                      valueListenable: historiqueRedoLengthNotifier,
                      builder: (context, historyRedoLength, _) {
                        return ValueListenableBuilder<LigneGoulotte?>(
                          valueListenable: goulotteNotifier,
                          builder: (context, goulotteActuelle, _) {
                            
                            // 1. Détermination de l'état "Actif/Inactif" pour chaque bouton
                            bool isUndoActive = (!isDrawGoulotteMode && historyLength > 1) || 
                                                (isDrawGoulotteMode && goulotteActuelle != null && goulotteInitiale != null && goulotteActuelle != goulotteInitiale);
                                                
                            // Le redo fonctionne de la même manière pour les 2 modes grâce à sa variable de longueur
                            bool isRedoActive = historyRedoLength > 0; 
                            
                            // Le reset global de l'équipement apparaît s'il y a un passé ou un futur d'enregistré
                            bool showResetEquipement = !isDrawGoulotteMode && (historyLength > 1 || historyRedoLength > 0);

                            // Si on n'a strictement rien bougé (position initiale), on cache le bloc d'historique entier
                            if (!isUndoActive && !isRedoActive && !showResetEquipement) {
                              return const SizedBox.shrink();
                            }

                            // 2. Vérification des actions en cours (Désactive tout temporairement pendant le calcul)
                            bool isGlobalDisabled = isProcessing || isDraggingEquipement || isDraggingGoulotte;

                            // CORRECTION : On ne cache plus les boutons, on les grise (disabled) s'ils sont inactifs !
                            // Cela empêche l'un de prendre la place de l'autre visuellement.
                            bool disableUndo = isGlobalDisabled || !isUndoActive;
                            bool disableRedo = isGlobalDisabled || !isRedoActive;
                            bool disableReset = isGlobalDisabled || historyLength <= 1; // Grisé si on est déjà au point de départ

                            return Column(
                              crossAxisAlignment: CrossAxisAlignment.end, // Aligné à droite
                              children: [
                                // LIGNE UNDO / REDO
                                // Affiche TOUJOURS les 2 boutons espacés fixement s'il y a au moins 1 action possible
                                if (isUndoActive || isRedoActive)
                                  Padding(
                                    padding: const EdgeInsets.only(bottom: 12.0),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min, // La Row prend juste la taille nécessaire
                                      children: [
                                        // BOUTON UNDO (Annuler)
                                        Container(
                                          decoration: BoxDecoration(
                                            shape: BoxShape.circle,
                                            boxShadow: [if (!disableUndo) BoxShadow(color: theme.shadowColor.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
                                          ),
                                          child: ClipOval(
                                            child: BackdropFilter(
                                              filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                                              child: Container(
                                                color: theme.cardColor.withValues(alpha: disableUndo ? 0.4 : 0.85), // S'adapte au mode sombre et se grise si inactif
                                                child: IconButton(
                                                  icon: const Icon(Icons.undo), 
                                                  color: disableUndo ? theme.disabledColor : theme.colorScheme.primary, 
                                                  tooltip: isDrawGoulotteMode ? 'Annuler la goulotte' : 'Annuler le déplacement',
                                                  onPressed: disableUndo ? null : onUndo, // Désactive l'action au clic
                                                ),
                                              ),
                                            ),
                                          ),
                                        ),

                                        const SizedBox(width: 8), // Espace physique strictement fixe entre les deux boutons

                                        // BOUTON REDO (Rétablir)
                                        Container(
                                          decoration: BoxDecoration(
                                            shape: BoxShape.circle,
                                            boxShadow: [if (!disableRedo) BoxShadow(color: theme.shadowColor.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
                                          ),
                                          child: ClipOval(
                                            child: BackdropFilter(
                                              filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                                              child: Container(
                                                color: theme.cardColor.withValues(alpha: disableRedo ? 0.4 : 0.85),
                                                child: IconButton(
                                                  icon: const Icon(Icons.redo), 
                                                  color: disableRedo ? theme.disabledColor : theme.colorScheme.primary,
                                                  tooltip: isDrawGoulotteMode ? 'Rétablir la goulotte' : 'Rétablir le déplacement',
                                                  onPressed: disableRedo ? null : onRedo,
                                                ),
                                              ),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),

                                // BOUTON RESET (Retour immédiat à la case départ)
                                if (showResetEquipement)
                                  Padding(
                                    padding: const EdgeInsets.only(bottom: 12.0),
                                    child: Container(
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        boxShadow: [if (!disableReset) BoxShadow(color: theme.shadowColor.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
                                      ),
                                      child: ClipOval(
                                        child: BackdropFilter(
                                          filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                                          child: Container(
                                            color: theme.cardColor.withValues(alpha: disableReset ? 0.4 : 0.85),
                                            child: IconButton(
                                              icon: const Icon(Icons.restart_alt), 
                                              color: disableReset ? theme.disabledColor : theme.colorScheme.secondary, 
                                              tooltip: 'Remettre à la position d\'origine',
                                              onPressed: disableReset ? null : onResetPosition,
                                            ),
                                          ),
                                        ),
                                      ),
                                    ),
                                  ),
                              ],
                            );
                          }
                        );
                      }
                    );
                  }
                );
              }
            );
          }
        ),
        
        // Bouton d'activation du Mode Goulotte
        Padding(
          padding: const EdgeInsets.only(bottom: 12.0),
          child: Container(
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              boxShadow: [BoxShadow(color: theme.shadowColor.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
            ),
            child: ClipOval(
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                child: Container(
                  color: isDrawGoulotteMode ? theme.colorScheme.primary.withValues(alpha: 0.9) : theme.cardColor.withValues(alpha: 0.85),
                  child: IconButton(
                    icon: Icon(Icons.format_paint, color: isDrawGoulotteMode ? theme.colorScheme.onPrimary : theme.colorScheme.primary),
                    tooltip: 'Tracer une goulotte',
                    onPressed: isProcessing ? null : onToggleGoulotteMode,
                  ),
                ),
              ),
            ),
          ),
        ),
        
        // Bouton pour ENLEVER la goulotte (Unique goulotte)
        if (isDrawGoulotteMode) // Apparaît uniquement en mode goulotte
          ValueListenableBuilder<LigneGoulotte?>(
            valueListenable: goulotteNotifier,
            builder: (context, goulotteActuelle, _) {
              if (goulotteActuelle == null) return const SizedBox.shrink();
              return Padding(
                padding: const EdgeInsets.only(bottom: 12.0),
                child: Container(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    boxShadow: [BoxShadow(color: theme.shadowColor.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
                  ),
                  child: ClipOval(
                    child: BackdropFilter(
                      filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                      child: Container(
                        color: theme.cardColor.withValues(alpha: 0.85),
                        child: IconButton(
                          icon: const Icon(Icons.delete_outline),
                          color: theme.colorScheme.error, // S'adapte au mode sombre
                          tooltip: 'Supprimer la goulotte',
                          onPressed: isProcessing ? null : () {
                            // Sécurité pour éviter le missclick
                            showDialog(
                              context: context,
                              builder: (BuildContext context) {
                                return AlertDialog(
                                  title: const Text("Supprimer la goulotte"),
                                  content: const Text("Êtes-vous sûr de vouloir effacer cette goulotte ?"),
                                  actions: [
                                    TextButton(
                                      onPressed: () => Navigator.of(context).pop(),
                                      child: Text("Annuler", style: TextStyle(color: theme.colorScheme.onSurface.withValues(alpha: 0.6))),
                                    ),
                                    ElevatedButton(
                                      style: ElevatedButton.styleFrom(backgroundColor: theme.colorScheme.error, foregroundColor: Colors.white),
                                      onPressed: () {
                                        Navigator.of(context).pop();
                                        onDeleteConfirmed();
                                      },
                                      child: const Text("Supprimer"),
                                    ),
                                  ],
                                );
                              },
                            );
                          },
                        ),
                      ),
                    ),
                  ),
                ),
              );
            }
          ),
      ],
    );
  }
}