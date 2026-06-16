import 'package:flutter/material.dart';
import '../models/devis_models.dart';

/// =========================================================================
/// === CLASSES UTILITAIRES (DESSIN ET DÉCOUPAGE) ===
/// =========================================================================

/// CustomPainter responsable de dessiner la goulotte vectorielle et ses points nodaux.
/// L'affichage est conditionnel pour optimiser les performances.
class GoulottePainter extends CustomPainter {
  final LigneGoulotte? goulotte;
  final double scale;
  final double offsetX;
  final double offsetY;
  final double thicknessOrig; 
  final bool showLine; // Gère l'affichage du corps de la goulotte
  final bool showNodes; // Gère l'affichage des nœuds bleus (ronds)

  GoulottePainter({
    required this.goulotte, 
    required this.scale, 
    required this.offsetX, 
    required this.offsetY, 
    required this.thicknessOrig, 
    required this.showLine,
    required this.showNodes,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (goulotte == null) return;

    final p1 = Offset(goulotte!.start.dx * scale + offsetX, goulotte!.start.dy * scale + offsetY);
    final p2 = Offset(goulotte!.end.dx * scale + offsetX, goulotte!.end.dy * scale + offsetY);
    
    // On dessine le trait vectoriel si demandé (quand l'image cuite d'OpenCV est masquée par un déplacement)
    if (showLine) {
      final paintLine = Paint()
        ..color = Colors.white70
        ..strokeWidth = thicknessOrig * scale 
        ..strokeCap = StrokeCap.butt; // Bout parfaitement plat
        
      canvas.drawLine(p1, p2, paintLine);
    }

    // Indicateurs nodaux (les ronds) pour montrer à l'utilisateur où grab/tirer la ligne
    if (showNodes) {
      final paintHandle = Paint()..color = Colors.white; // Anneaux blancs au lieu de ronds bleus pleins
      final paintHandleBorder = Paint()
        ..color = Colors.teal
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5; 
      
      List<Offset> nodes = [p1, p2];
      for (var p in nodes) {
        // Ombre portée sous le nœud
        canvas.drawCircle(p, 9.0, Paint()..color = Colors.black26..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3));
        canvas.drawCircle(p, 8.0, paintHandle);
        canvas.drawCircle(p, 8.0, paintHandleBorder);
      }
    }
  }

  @override
  bool shouldRepaint(covariant GoulottePainter oldDelegate) => true;
}

/// CustomClipper utilisé pour créer l'effet de séparation (Slider Split Screen).
/// Permet de comparer le mur original avec le mur traité par l'IA.
class SplitClipper extends CustomClipper<Rect> {
  final double percentage;
  SplitClipper(this.percentage);

  @override
  Rect getClip(Size size) {
    return Rect.fromLTRB(0, 0, size.width * percentage, size.height);
  }

  @override
  bool shouldReclip(SplitClipper oldClipper) => percentage != oldClipper.percentage;
}

/// CustomPainter utilisé pour dessiner la zone de sélection manuelle (Bounding Box) et sa surface bleutée.
class BoundingBoxPainter extends CustomPainter {
  final List<Offset> points;
  BoundingBoxPainter({required this.points});

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length != 4) return;
    
    final paint = Paint()
      ..color = Colors.teal // Adapté à la charte graphique
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;
      
    final path = Path()
      ..moveTo(points[0].dx, points[0].dy)
      ..lineTo(points[1].dx, points[1].dy)
      ..lineTo(points[2].dx, points[2].dy)
      ..lineTo(points[3].dx, points[3].dy)
      ..close();
      
    canvas.drawPath(path, paint);

    // Dessine de légers crochets dans les angles pour un aspect viseur d'appareil photo
    double cornerLength = 20.0;
    final cornerPaint = Paint()..color = Colors.teal..strokeWidth = 5.0..style = PaintingStyle.stroke;
    
    for (int i = 0; i < 4; i++) {
      Offset p = points[i];
      Offset pNext = points[(i + 1) % 4];
      Offset pPrev = points[(i + 3) % 4];
      
      Offset dNext = (pNext - p) / (pNext - p).distance;
      Offset dPrev = (pPrev - p) / (pPrev - p).distance;
      
      Path cornerPath = Path()
        ..moveTo(p.dx + dPrev.dx * cornerLength, p.dy + dPrev.dy * cornerLength)
        ..lineTo(p.dx, p.dy)
        ..lineTo(p.dx + dNext.dx * cornerLength, p.dy + dNext.dy * cornerLength);
      canvas.drawPath(cornerPath, cornerPaint);
    }

    final fillPaint = Paint()
      ..color = Colors.teal.withValues(alpha: 0.15)
      ..style = PaintingStyle.fill;
    canvas.drawPath(path, fillPaint);
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) => true;
}