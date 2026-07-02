import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../../utils/painters_resultat.dart';

class CalquePlacementManuel extends StatelessWidget {
  final String photoPath;
  final List<Map<String, double>> pointsCibles;
  final int imageWidth;
  final int imageHeight;
  final double scale;
  final double offsetX;
  final double offsetY;
  final TransformationController transformationController;
  final bool attenteConfirmationIA;
  final int activePointers;
  final ThemeData theme;
  
  final VoidCallback onAlignerZone;
  final Function(double dx1024, double dy1024) onDeplacerZone;
  final Function(int idx, double newX, double newY) onRedimensionnerZone;

  const CalquePlacementManuel({
    super.key,
    required this.photoPath,
    required this.pointsCibles,
    required this.imageWidth,
    required this.imageHeight,
    required this.scale,
    required this.offsetX,
    required this.offsetY,
    required this.transformationController,
    required this.attenteConfirmationIA,
    required this.activePointers,
    required this.theme,
    required this.onAlignerZone,
    required this.onDeplacerZone,
    required this.onRedimensionnerZone,
  });

  @override
  Widget build(BuildContext context) {
    List<Offset> screenPoints = pointsCibles.map((p) {
      double pxOrig = p['x']! * (imageWidth / 1024.0);
      double pyOrig = p['y']! * (imageHeight / 1024.0);
      return Offset(pxOrig * scale + offsetX, pyOrig * scale + offsetY);
    }).toList();

    double minX = screenPoints.map((p) => p.dx).reduce(math.min);
    double maxX = screenPoints.map((p) => p.dx).reduce(math.max);
    double minY = screenPoints.map((p) => p.dy).reduce(math.min);
    double maxY = screenPoints.map((p) => p.dy).reduce(math.max);

    // COMPENSATON DYNAMIQUE DU ZOOM : On écoute le contrôleur d'InteractiveViewer
    return ValueListenableBuilder<Matrix4>(
      valueListenable: transformationController,
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
                child: Image.file(File(photoPath), fit: BoxFit.contain),
              )
            ),
            
            // Le dessinateur va utiliser currentZoom pour diviser l'épaisseur du pinceau
            Positioned.fill(
              child: CustomPaint(
                painter: BoundingBoxPainter(
                  points: screenPoints, 
                  primaryColor: theme.colorScheme.secondary, // RECTANGLE : Utilise le nouveau bleu clair de la marque
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
                  if (attenteConfirmationIA) {
                    onAlignerZone(); // Utilisation de la méthode extraite
                  }
                },
                onPanUpdate: (details) {
                  // Sécurité : on ignore le déplacement si plusieurs doigts sont détectés
                  if (activePointers > 1) return;
                  
                  double dxOrig = details.delta.dx / scale;
                  double dyOrig = details.delta.dy / scale;
                  double dx1024 = dxOrig * (1024.0 / imageWidth);
                  double dy1024 = dyOrig * (1024.0 / imageHeight);
                  
                  onDeplacerZone(dx1024, dy1024); // Utilisation de la méthode extraite
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
                    if (attenteConfirmationIA) {
                      onAlignerZone(); // Utilisation de la méthode extraite
                    }
                  },
                  onPanUpdate: (details) {
                    // Sécurité multi-touch
                    if (activePointers > 1) return;
                    
                    double dxOrig = details.delta.dx / scale;
                    double dyOrig = details.delta.dy / scale;
                    double dx1024 = dxOrig * (1024.0 / imageWidth);
                    double dy1024 = dyOrig * (1024.0 / imageHeight);
                    
                    double newX = (pointsCibles[idx]['x']! + dx1024).clamp(0.0, 1024.0);
                    double newY = (pointsCibles[idx]['y']! + dy1024).clamp(0.0, 1024.0);

                    onRedimensionnerZone(idx, newX, newY); // Utilisation de la méthode extraite
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
                        color: theme.colorScheme.secondary.withValues(alpha: 0.6),
                        shape: BoxShape.rectangle,
                        border: Border.all(color: theme.colorScheme.secondary, width: 2.0 * invZoom), 
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
}