import 'package:flutter/material.dart';

class CarteConfirmationIA extends StatelessWidget {
  final ThemeData theme;
  final VoidCallback onAjuster;
  final VoidCallback onValider;

  const CarteConfirmationIA({
    super.key,
    required this.theme,
    required this.onAjuster,
    required this.onValider,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
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
                    onPressed: onAjuster,
                    child: Text("Ajuster", style: TextStyle(color: theme.colorScheme.secondary)),
                  ),
                  ElevatedButton.icon(
                    icon: const Icon(Icons.check, size: 18),
                    label: const Text("Oui, valider"),
                    style: ElevatedButton.styleFrom(backgroundColor: theme.colorScheme.primary, foregroundColor: theme.colorScheme.onPrimary),
                    onPressed: onValider,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}