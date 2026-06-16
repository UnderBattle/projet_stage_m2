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
    required this.onToggleGoulotteMode,
    required this.onDeleteConfirmed,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Le bouton Réinitialiser intelligent (grisé pendant le drag ou le calcul)
        ValueListenableBuilder<bool>(
          valueListenable: isDraggingEquipementNotifier,
          builder: (context, isDraggingEquipement, _) {
            return ValueListenableBuilder<bool>(
              valueListenable: isDraggingGoulotteNotifier,
              builder: (context, isDraggingGoulotte, _) {
                
                // OPTIMISATION : On écoute uniquement la LONGUEUR de l'historique 
                // Le bouton ne se reconstruira plus jamais pendant que le doigt bouge l'équipement !
                return ValueListenableBuilder<int>(
                  valueListenable: historiqueLengthNotifier,
                  builder: (context, historyLength, _) {
                    return ValueListenableBuilder<LigneGoulotte?>(
                      valueListenable: goulotteNotifier,
                      builder: (context, goulotteActuelle, _) {
                        
                        // 1. Est-ce que le bouton a une raison d'être affiché ?
                        bool showResetEquipement = !isDrawGoulotteMode && historyLength > 1;
                        bool showResetGoulotte = isDrawGoulotteMode && 
                                                 goulotteActuelle != null && 
                                                 goulotteInitiale != null && 
                                                 goulotteActuelle != goulotteInitiale;

                        // Si on n'a rien bougé, on ne montre pas le bouton
                        if (!showResetEquipement && !showResetGoulotte) return const SizedBox.shrink();

                        // 2. Est-ce que le bouton doit être désactivé (grisé) ?
                        bool isDisabled = isProcessing || isDraggingEquipement || isDraggingGoulotte;

                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12.0),
                          // STYLE : Bouton flottant Glassmorphism
                          child: Container(
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              boxShadow: [if (!isDisabled) BoxShadow(color: Colors.black.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
                            ),
                            child: ClipOval(
                              child: BackdropFilter(
                                filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                                child: Container(
                                  color: Colors.white.withValues(alpha: isDisabled ? 0.4 : 0.85),
                                  child: IconButton(
                                    icon: const Icon(Icons.undo), 
                                    color: isDisabled ? Colors.grey : Colors.teal, // Grise l'icône si inactif
                                    tooltip: isDrawGoulotteMode ? 'Réinitialiser la goulotte' : 'Annuler le déplacement',
                                    onPressed: isDisabled ? null : onUndo, // Désactive l'action
                                  ),
                                ),
                              ),
                            ),
                          ),
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
              boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
            ),
            child: ClipOval(
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                child: Container(
                  color: isDrawGoulotteMode ? Colors.teal.withValues(alpha: 0.9) : Colors.white.withValues(alpha: 0.85),
                  child: IconButton(
                    icon: Icon(Icons.format_paint, color: isDrawGoulotteMode ? Colors.white : Colors.teal),
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
                    boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.15), blurRadius: 10, offset: const Offset(0, 4))],
                  ),
                  child: ClipOval(
                    child: BackdropFilter(
                      filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                      child: Container(
                        color: Colors.white.withValues(alpha: 0.85),
                        child: IconButton(
                          icon: const Icon(Icons.delete_outline),
                          color: Colors.red,
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
                                      child: const Text("Annuler", style: TextStyle(color: Colors.grey)),
                                    ),
                                    ElevatedButton(
                                      style: ElevatedButton.styleFrom(backgroundColor: Colors.red, foregroundColor: Colors.white),
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