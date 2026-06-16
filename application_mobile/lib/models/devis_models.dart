import 'package:flutter/material.dart';

// Structure de données pour mémoriser la goulotte unique
class LigneGoulotte {
  final Offset start;
  final Offset end;
  LigneGoulotte(this.start, this.end);

  // AJOUT : Surcharge des opérateurs pour comparer facilement deux goulottes (Utile pour le Undo)
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is LigneGoulotte &&
          runtimeType == other.runtimeType &&
          start == other.start &&
          end == other.end;

  @override
  int get hashCode => start.hashCode ^ end.hashCode;
}

// Les différents états d'interaction avec la goulotte
enum DragMode { none, start, end, body, drawingNew }