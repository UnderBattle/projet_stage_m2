import 'dart:ui';
import 'package:flutter/material.dart';
import '../models/devis_models.dart';

class BoutonsActionDevis extends StatelessWidget {
  final ValueNotifier<bool> isDraggingEquipementNotifier;
  final ValueNotifier<bool> isDraggingGoulotteNotifier;
  final ValueNotifier<int> historiqueLengthNotifier;
  final ValueNotifier<LigneGoulotte?> goulotteNotifier;
  final LigneGoulotte? goulotteInitiale;
  final bool isDrawGoulotteMode;
  final bool isProcessing;
  final VoidCallback onUndo;
  final VoidCallback onResetPosition; // Permet de remettre l'équipement au centre
  final VoidCallback onToggleGoulotteMode;
  final VoidCallback onDeleteConfirmed;

  const BoutonsActionDevis({
    super.key,
    required this.isDraggingEquipementNotifier,
    required this.isDraggingGoulotteNotifier,
    required this.historiqueLengthNotifier,
    required this.goulotteNotifier,
    required this.goulotteInitiale,
    required this.isDrawGoulotteMode,
    required this.isProcessing,
    required this.onUndo,
    required this.onResetPosition,
    required this.onToggleGoulotteMode,
    required this.onDeleteConfirmed,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context); // Récupération du thème

    return Column(
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
                    return ValueListenableBuilder<LigneGoulotte?>(
                      valueListenable: goulotteNotifier,
                      builder: (context, goulotteActuelle, _) {
                        
                        // 1. Est-ce que les boutons ont une raison d'être affichés ?
                        bool showHistoriqueEquipement = !isDrawGoulotteMode && historyLength > 1;
                        bool showResetGoulotte = isDrawGoulotteMode && 
                                                 goulotteActuelle != null && 
                                                 goulotteInitiale != null && 
                                                 goulotteActuelle != goulotteInitiale;

                        // Si on n'a rien bougé, on ne montre pas le bloc d'historique
                        if (!showHistoriqueEquipement && !showResetGoulotte) return const SizedBox.shrink();

                        // 2. Est-ce que le bouton doit être désactivé (grisé) ?
                        bool isDisabled = isProcessing || isDraggingEquipement || isDraggingGoulotte;

                        return Column(
                          children: [
                            // BOUTON UNDO (Annuler le dernier mouvement pas à pas)
                            Padding(
                              padding: const EdgeInsets.only(bottom: 12.0),
                              // STYLE : Bouton flottant Glassmorphism
                              child: Container(
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  boxShadow: [if (!isDisabled) BoxShadow(color: theme.shadowColor.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
                                ),
                                child: ClipOval(
                                  child: BackdropFilter(
                                    filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                                    child: Container(
                                      color: theme.cardColor.withValues(alpha: isDisabled ? 0.4 : 0.85), // S'adapte au mode sombre
                                      child: IconButton(
                                        icon: const Icon(Icons.undo), 
                                        color: isDisabled ? theme.disabledColor : theme.colorScheme.primary, // Grise l'icône si inactif
                                        tooltip: isDrawGoulotteMode ? 'Réinitialiser la goulotte' : 'Annuler le dernier déplacement',
                                        onPressed: isDisabled ? null : onUndo, // Désactive l'action
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ),

                            // NOUVEAU : BOUTON RESET (Retour immédiat à la case départ)
                            // Apparaît uniquement pour l'équipement si on l'a bougé
                            if (showHistoriqueEquipement)
                              Padding(
                                padding: const EdgeInsets.only(bottom: 12.0),
                                child: Container(
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    boxShadow: [if (!isDisabled) BoxShadow(color: theme.shadowColor.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
                                  ),
                                  child: ClipOval(
                                    child: BackdropFilter(
                                      filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                                      child: Container(
                                        color: theme.cardColor.withValues(alpha: isDisabled ? 0.4 : 0.85),
                                        child: IconButton(
                                          // CORRECTION : Icône universelle de réinitialisation ("Reset")
                                          icon: const Icon(Icons.restart_alt), 
                                          color: isDisabled ? theme.disabledColor : theme.colorScheme.secondary, // Utilise la couleur d'accentuation
                                          tooltip: 'Remettre à la position d\'origine',
                                          onPressed: isDisabled ? null : onResetPosition,
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