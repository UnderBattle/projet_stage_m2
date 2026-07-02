import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../../models/devis_models.dart';
import '../../utils/painters_resultat.dart';
import '../../services/catalogue_service.dart';

class CalqueSimulation extends StatefulWidget {
  final String photoPath;
  final double scale;
  final double offsetX;
  final double offsetY;
  final BoxConstraints constraints;
  final ThemeData theme;
  
  final int imageWidth;
  final int imageHeight;
  final List<Map<String, double>> pointsCibles;
  final Equipement? modeleSelectionne;
  
  final Uint8List? imageFondPropreBytes;
  final Uint8List? imageFondAvecGoulotteBytes;
  final Uint8List? imageResultatBytes;
  final Uint8List? calqueEquipementPngBytes;
  final Uint8List? dernierCalqueEquipementCompletBytes;
  final Offset decalageDuCalqueComplet;
  
  final bool isDrawGoulotteMode;
  final int activePointers;
  
  final ValueNotifier<Offset> decalageNotifier;
  final ValueNotifier<double> splitNotifier;
  final ValueNotifier<bool> isDraggingEquipementNotifier;
  final ValueNotifier<LigneGoulotte?> goulotteNotifier;
  final ValueNotifier<bool> isDraggingGoulotteNotifier;
  final ValueNotifier<Offset?> goulotteCurrentEndOrigNotifier;
  final ValueNotifier<Offset?> magnifierPositionNotifier;
  
  final Offset lastHistoriqueDecalage;
  
  final VoidCallback onEquipementDropped;
  final Function(LigneGoulotte) onGoulotteCreated;
  final VoidCallback onGoulotteEdited;

  const CalqueSimulation({
    super.key,
    required this.photoPath,
    required this.scale,
    required this.offsetX,
    required this.offsetY,
    required this.constraints,
    required this.theme,
    required this.imageWidth,
    required this.imageHeight,
    required this.pointsCibles,
    required this.modeleSelectionne,
    required this.imageFondPropreBytes,
    required this.imageFondAvecGoulotteBytes,
    required this.imageResultatBytes,
    required this.calqueEquipementPngBytes,
    required this.dernierCalqueEquipementCompletBytes,
    required this.decalageDuCalqueComplet,
    required this.isDrawGoulotteMode,
    required this.activePointers,
    required this.decalageNotifier,
    required this.splitNotifier,
    required this.isDraggingEquipementNotifier,
    required this.goulotteNotifier,
    required this.isDraggingGoulotteNotifier,
    required this.goulotteCurrentEndOrigNotifier,
    required this.magnifierPositionNotifier,
    required this.lastHistoriqueDecalage,
    required this.onEquipementDropped,
    required this.onGoulotteCreated,
    required this.onGoulotteEdited,
  });

  @override
  State<CalqueSimulation> createState() => _CalqueSimulationState();
}

class _CalqueSimulationState extends State<CalqueSimulation> {
  Offset? _goulotteStartOrig;

  // Fonction pour empêcher un point de sortir des limites strictes de l'image
  Offset _clampToImageBounds(Offset point) {
    return Offset(
      point.dx.clamp(0.0, widget.imageWidth.toDouble()),
      point.dy.clamp(0.0, widget.imageHeight.toDouble()),
    );
  }

  // Fonction pour forcer la goulotte à être parfaitement droite (horizontale ou verticale)
  Offset _snapToOrthogonal(Offset reference, Offset target) {
    double dx = (target.dx - reference.dx).abs();
    double dy = (target.dy - reference.dy).abs();
    if (dx > dy) {
      return Offset(target.dx, reference.dy);
    } else {
      return Offset(reference.dx, target.dy);
    }
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
              if (widget.activePointers > 1) return;
              widget.isDraggingGoulotteNotifier.value = true;
              // On ne déclenche pas la loupe pour le déplacement du corps entier
            },
            onPanUpdate: (details) {
              if (widget.activePointers > 1 || !widget.isDraggingGoulotteNotifier.value) return;

              double cosA = math.cos(angle);
              double sinA = math.sin(angle);
              double globalDx = details.delta.dx * cosA - details.delta.dy * sinA;
              double globalDy = details.delta.dx * sinA + details.delta.dy * cosA;
              Offset deltaOrig = Offset(globalDx, globalDy) / scale;
              
              var g = widget.goulotteNotifier.value!;
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
              if (maxX > widget.imageWidth) adjustDx = widget.imageWidth - maxX;
              if (minY < 0) adjustDy = -minY;
              if (maxY > widget.imageHeight) adjustDy = widget.imageHeight - maxY;
              
              newStart = Offset(newStart.dx + adjustDx, newStart.dy + adjustDy);
              newEnd = Offset(newEnd.dx + adjustDx, newEnd.dy + adjustDy);
              
              widget.goulotteNotifier.value = LigneGoulotte(newStart, newEnd);
            },
            onPanEnd: (_) {
              if (!widget.isDraggingGoulotteNotifier.value) return;
              widget.isDraggingGoulotteNotifier.value = false;
              widget.onGoulotteEdited();
            },
            onPanCancel: () {
              if (!widget.isDraggingGoulotteNotifier.value) return;
              widget.isDraggingGoulotteNotifier.value = false;
              widget.onGoulotteEdited();
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
            if (widget.activePointers > 1) return;
            widget.isDraggingGoulotteNotifier.value = true;
            
            // Déclenche la loupe sur ce nœud
            widget.magnifierPositionNotifier.value = Offset(goulotte.start.dx * scale + offsetX, goulotte.start.dy * scale + offsetY);
          },
          onPanUpdate: (details) {
            if (widget.activePointers > 1 || !widget.isDraggingGoulotteNotifier.value) return;
            Offset deltaOrig = details.delta / scale;
            var g = widget.goulotteNotifier.value!;
            Offset rawStart = g.start + deltaOrig;
            
            // Clamper pour ne pas sortir des limites de l'image
            rawStart = _clampToImageBounds(rawStart);
            
            // On force l'alignement rectiligne parfait (horizontal ou vertical)
            Offset snappedStart = _snapToOrthogonal(g.end, rawStart);
            
            // Re-clamp par sécurité au cas où le snap pousserait le point dehors
            widget.goulotteNotifier.value = LigneGoulotte(_clampToImageBounds(snappedStart), g.end);
            
            // Mise à jour de la loupe
            widget.magnifierPositionNotifier.value = Offset(snappedStart.dx * scale + offsetX, snappedStart.dy * scale + offsetY);
          },
          onPanEnd: (_) {
            widget.magnifierPositionNotifier.value = null; // Cache la loupe
            if (!widget.isDraggingGoulotteNotifier.value) return;
            widget.isDraggingGoulotteNotifier.value = false;
            widget.onGoulotteEdited();
          },
          onPanCancel: () {
            widget.magnifierPositionNotifier.value = null; // Cache la loupe
            if (!widget.isDraggingGoulotteNotifier.value) return;
            widget.isDraggingGoulotteNotifier.value = false;
            widget.onGoulotteEdited();
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
            if (widget.activePointers > 1) return;
            widget.isDraggingGoulotteNotifier.value = true;
            
            // Déclenche la loupe sur ce nœud
            widget.magnifierPositionNotifier.value = Offset(goulotte.end.dx * scale + offsetX, goulotte.end.dy * scale + offsetY);
          },
          onPanUpdate: (details) {
            if (widget.activePointers > 1 || !widget.isDraggingGoulotteNotifier.value) return;
            Offset deltaOrig = details.delta / scale;
            var g = widget.goulotteNotifier.value!;
            Offset rawEnd = g.end + deltaOrig;
            
            // Clamper pour ne pas sortir des limites de l'image
            rawEnd = _clampToImageBounds(rawEnd);
            
            // On force l'alignement rectiligne parfait (horizontal ou vertical)
            Offset snappedEnd = _snapToOrthogonal(g.start, rawEnd);
            
            // Re-clamp par sécurité
            widget.goulotteNotifier.value = LigneGoulotte(g.start, _clampToImageBounds(snappedEnd));
            
            // Mise à jour de la loupe
            widget.magnifierPositionNotifier.value = Offset(snappedEnd.dx * scale + offsetX, snappedEnd.dy * scale + offsetY);
          },
          onPanEnd: (_) {
            widget.magnifierPositionNotifier.value = null; // Cache la loupe
            if (!widget.isDraggingGoulotteNotifier.value) return;
            widget.isDraggingGoulotteNotifier.value = false;
            widget.onGoulotteEdited();
          },
          onPanCancel: () {
            widget.magnifierPositionNotifier.value = null; // Cache la loupe
            if (!widget.isDraggingGoulotteNotifier.value) return;
            widget.isDraggingGoulotteNotifier.value = false;
            widget.onGoulotteEdited();
          },
          child: Container(width: 50, height: 50, color: Colors.transparent),
        ),
      ),
    ];
  }

  @override
  Widget build(BuildContext context) {
    double ptHgXOrig = widget.pointsCibles[0]['x']! * (widget.imageWidth / 1024.0);
    double ptHgYOrig = widget.pointsCibles[0]['y']! * (widget.imageHeight / 1024.0);
    double ptHdXOrig = widget.pointsCibles[1]['x']! * (widget.imageWidth / 1024.0);
    double ptHdYOrig = widget.pointsCibles[1]['y']! * (widget.imageHeight / 1024.0);

    double dx = ptHdXOrig - ptHgXOrig;
    double dy = ptHdYOrig - ptHgYOrig;
    double autoWPxOrig = math.sqrt(dx * dx + dy * dy);
    
    // On récupère les dimensions depuis le catalogue pour le calcul d'affichage UI
    double largeurMm = 798.0; 
    double hauteurMm = 270.0;
    if (widget.modeleSelectionne != null) {
      largeurMm = widget.modeleSelectionne!.largeur;
      hauteurMm = widget.modeleSelectionne!.hauteur;
    }

    double equipementWPxOrig = (largeurMm / 50.0) * autoWPxOrig;
    double equipementHPxOrig = equipementWPxOrig * (hauteurMm / largeurMm);

    double equipementScreenW = equipementWPxOrig * widget.scale;
    double equipementScreenH = equipementHPxOrig * widget.scale;

    double angleRad = 0.0; 
    
    // Calcul de l'épaisseur pour le Painter
    double ratioPxParMm = autoWPxOrig / 50.0;
    double largeurGoulotteOrig = 80.0 * ratioPxParMm; 

    return Stack(
      children: [
        // 1. Couche de Fond : Mur Propre OU Mur avec OpenCV Goulotte
        Positioned.fill(
          child: ValueListenableBuilder<bool>( // Écoute le drag de l'équipement
            valueListenable: widget.isDraggingEquipementNotifier,
            builder: (context, isDraggingEquipement, _) {
              return ValueListenableBuilder<bool>( // Écoute le drag de la goulotte
                valueListenable: widget.isDraggingGoulotteNotifier,
                builder: (context, isDraggingGoulotte, _) {
                  Uint8List? imgToShow;
                  if (isDraggingGoulotte) {
                    // Quand on déplace la goulotte, on affiche le mur propre en fond.
                    imgToShow = widget.imageFondPropreBytes;
                  } else {
                    // Sinon (déplacement equipement ou repos), le fond est le mur avec la goulotte (si tracée).
                    imgToShow = widget.imageFondAvecGoulotteBytes ?? widget.imageFondPropreBytes;
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
        if (widget.imageResultatBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: widget.isDraggingEquipementNotifier,
            builder: (context, isDraggingEquipement, _) {
              return ValueListenableBuilder<bool>(
                valueListenable: widget.isDraggingGoulotteNotifier,
                builder: (context, isDraggingGoulotte, _) {
                  // Si l'utilisateur touche une pièce, on cache le rendu final
                  if (isDraggingEquipement || isDraggingGoulotte) return const SizedBox.shrink(); 
                  
                  return ValueListenableBuilder<double>(
                    valueListenable: widget.splitNotifier,
                    builder: (context, splitVal, _) {
                      double currentSplit = widget.isDrawGoulotteMode ? 1.0 : splitVal;
                      return Positioned.fill(
                        child: ClipRect(
                          clipper: SplitClipper(currentSplit),
                          child: Image.memory(widget.imageResultatBytes!, fit: BoxFit.contain, gaplessPlayback: true), 
                        ),
                      );
                    }
                  );
                }
              );
            }
          ),

        if (widget.calqueEquipementPngBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: widget.isDraggingGoulotteNotifier,
            builder: (context, isDraggingGoulotte, _) {
              if (isDraggingGoulotte) {
                // Quand on déplace la goulotte, on montre le calque PNG parfait de la clim sur le mur
                return Positioned.fill(
                  child: Image.memory(widget.calqueEquipementPngBytes!, fit: BoxFit.contain, gaplessPlayback: true),
                );
              }
              return const SizedBox.shrink();
            }
          ),

        // Le painter global de la goulotte
        Positioned.fill(
          child: IgnorePointer(
            child: ValueListenableBuilder<LigneGoulotte?>(
              valueListenable: widget.goulotteNotifier,
              builder: (context, goulotte, _) {
                return ValueListenableBuilder<bool>(
                  valueListenable: widget.isDraggingEquipementNotifier, // Etat du drag équipement
                  builder: (context, isDraggingEquipement, _) {
                    return ValueListenableBuilder<bool>(
                      valueListenable: widget.isDraggingGoulotteNotifier, // Etat du drag goulotte
                      builder: (context, isDraggingGoulotte, _) {
                        
                        // On n'affiche le trait vectoriel QUE lorsqu'on déplace la goulotte
                        // Sinon (si on déplace la clim), on verra la magnifique goulotte rendue par OpenCV en fond.
                        bool showLine = isDraggingGoulotte;
                        // On affiche les ronds bleus (nœuds) uniquement si l'outil pinceau est activé
                        bool showNodes = widget.isDrawGoulotteMode;

                        return CustomPaint(
                          painter: GoulottePainter(
                            goulotte: goulotte,
                            scale: widget.scale,
                            offsetX: widget.offsetX,
                            offsetY: widget.offsetY,
                            thicknessOrig: largeurGoulotteOrig,
                            showLine: showLine,
                            showNodes: showNodes,
                            primaryColor: widget.theme.colorScheme.primary, // Injecte la couleur de la marque
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
        if (widget.imageResultatBytes != null)
          ValueListenableBuilder<bool>(
            valueListenable: widget.isDraggingEquipementNotifier,
            builder: (context, isDragging, _) {
              // On masque la barre de slider si l'utilisateur trace une goulotte ou drag
              if (isDragging || widget.isDrawGoulotteMode) return const SizedBox.shrink();
              return ValueListenableBuilder<double>(
                valueListenable: widget.splitNotifier,
                builder: (context, splitVal, _) {
                  return Positioned(
                    key: const ValueKey('slider_interactif'),
                    top: 0,
                    bottom: 0,
                    left: (widget.constraints.maxWidth * splitVal) - 20, 
                    child: GestureDetector(
                      behavior: HitTestBehavior.opaque,
                      onHorizontalDragUpdate: (details) {
                        // Anti-Missclick pendant le zoom
                        if (widget.activePointers > 1) return;
                        widget.splitNotifier.value = (widget.splitNotifier.value + details.delta.dx / widget.constraints.maxWidth).clamp(0.0, 1.0);
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
                                boxShadow: [BoxShadow(color: widget.theme.colorScheme.primary.withValues(alpha: 0.5), blurRadius: 8, spreadRadius: 1)]
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
                                child: Icon(Icons.compare_arrows, size: 20, color: widget.theme.colorScheme.primary), // Adapté au thème
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
        if (widget.modeleSelectionne != null)
          ValueListenableBuilder<Offset>(
            valueListenable: widget.decalageNotifier,
            builder: (context, decalage, _) {
              return ValueListenableBuilder<bool>(
                valueListenable: widget.isDraggingEquipementNotifier,
                builder: (context, isDraggingEquipement, _) {
                  double equipementScreenX = (ptHgXOrig + decalage.dx) * widget.scale + widget.offsetX;
                  double equipementScreenY = (ptHgYOrig + decalage.dy) * widget.scale + widget.offsetY;
                  
                  // OPTIMISATION DRAG : On calcule le vecteur de translation par rapport au dernier rendu complet !
                  Uint8List? dragImage = widget.dernierCalqueEquipementCompletBytes ?? widget.calqueEquipementPngBytes;
                  Offset refDecalage = widget.dernierCalqueEquipementCompletBytes != null 
                      ? widget.decalageDuCalqueComplet 
                      : widget.lastHistoriqueDecalage;
                      
                  double diffX = (decalage.dx - refDecalage.dx) * widget.scale;
                  double diffY = (decalage.dy - refDecalage.dy) * widget.scale;

                  return Stack(
                    children: [
                      // Rendu complet 3D en mouvement
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
                          ignoring: widget.isDrawGoulotteMode,
                          child: GestureDetector(
                            behavior: HitTestBehavior.translucent,
                            onPanStart: (_) {
                              if (widget.activePointers > 1) return;
                              widget.isDraggingEquipementNotifier.value = true;
                            },
                            onPanUpdate: (details) { 
                               if (widget.activePointers > 1 || !widget.isDraggingEquipementNotifier.value) return;
                               
                               double cosA = math.cos(angleRad);
                               double sinA = math.sin(angleRad);
                               double globalDx = details.delta.dx * cosA - details.delta.dy * sinA;
                               double globalDy = details.delta.dx * sinA + details.delta.dy * cosA;

                               widget.decalageNotifier.value = Offset(
                                 widget.decalageNotifier.value.dx + globalDx / widget.scale,
                                 widget.decalageNotifier.value.dy + globalDy / widget.scale
                               );
                            },
                            onPanEnd: (_) { 
                               if (!widget.isDraggingEquipementNotifier.value) return;
                               widget.isDraggingEquipementNotifier.value = false;
                               widget.onEquipementDropped(); 
                             },
                            onPanCancel: () { 
                               if (!widget.isDraggingEquipementNotifier.value) return;
                               widget.isDraggingEquipementNotifier.value = false;
                               widget.onEquipementDropped();
                             },
                            child: Transform.rotate(
                              angle: angleRad,
                              alignment: Alignment.topLeft, 
                              child: Opacity(
                                opacity: 0.0, 
                                child: Image.asset(widget.modeleSelectionne!.chemin, fit: BoxFit.fill),
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
        if (widget.isDrawGoulotteMode)
          Positioned.fill(
            child: ValueListenableBuilder<LigneGoulotte?>(
              valueListenable: widget.goulotteNotifier,
              builder: (context, goulotte, _) {
                // La couche d'interaction (Création OU Édition)
                if (goulotte == null) {
                  // A - Mode plein écran pour tracer la toute première goulotte
                  return GestureDetector(
                    behavior: HitTestBehavior.translucent,
                    onPanStart: (details) {
                      if (widget.activePointers > 1) return;
                      // On garde la trace initiale sans provoquer la disparition de la zone tactile
                      Offset touchOrig = (details.localPosition - Offset(widget.offsetX, widget.offsetY)) / widget.scale;
                      
                      // Clamper le point de départ pour ne pas commencer en dehors
                      _goulotteStartOrig = _clampToImageBounds(touchOrig);
                      widget.goulotteCurrentEndOrigNotifier.value = _goulotteStartOrig;
                      widget.isDraggingGoulotteNotifier.value = true;
                      
                      // Affiche la loupe
                      widget.magnifierPositionNotifier.value = Offset(_goulotteStartOrig!.dx * widget.scale + widget.offsetX, _goulotteStartOrig!.dy * widget.scale + widget.offsetY);
                    },
                    onPanUpdate: (details) {
                      if (widget.activePointers > 1 || !widget.isDraggingGoulotteNotifier.value) return;
                      if (_goulotteStartOrig != null) {
                        Offset rawEnd = (details.localPosition - Offset(widget.offsetX, widget.offsetY)) / widget.scale;
                        
                        // On clamp pour ne pas sortir de l'image
                        rawEnd = _clampToImageBounds(rawEnd);
                        
                        // Force le tracé à être parfaitement droit
                        Offset snappedEnd = _snapToOrthogonal(_goulotteStartOrig!, rawEnd);
                        
                        // Re-clamp au cas où le snap aurait poussé la ligne en dehors
                        widget.goulotteCurrentEndOrigNotifier.value = _clampToImageBounds(snappedEnd);
                        
                        // Met à jour la position de la loupe
                        widget.magnifierPositionNotifier.value = Offset(snappedEnd.dx * widget.scale + widget.offsetX, snappedEnd.dy * widget.scale + widget.offsetY);
                      }
                    },
                    onPanEnd: (_) {
                      widget.magnifierPositionNotifier.value = null; // Cache la loupe
                      if (!widget.isDraggingGoulotteNotifier.value) return;
                      if (_goulotteStartOrig != null && widget.goulotteCurrentEndOrigNotifier.value != null) {
                        widget.isDraggingGoulotteNotifier.value = false;
                        
                        LigneGoulotte newGoulotte = LigneGoulotte(_goulotteStartOrig!, widget.goulotteCurrentEndOrigNotifier.value!);
                        
                        _goulotteStartOrig = null;
                        widget.goulotteCurrentEndOrigNotifier.value = null;

                        widget.onGoulotteCreated(newGoulotte);
                      }
                    },
                    onPanCancel: () {
                      widget.magnifierPositionNotifier.value = null; // Cache la loupe
                      if (!widget.isDraggingGoulotteNotifier.value) return;
                      widget.isDraggingGoulotteNotifier.value = false;
                      _goulotteStartOrig = null;
                      widget.goulotteCurrentEndOrigNotifier.value = null;
                    },
                    // Le painter temporaire pour afficher le trait PENDANT sa création
                    child: ValueListenableBuilder<Offset?>(
                      valueListenable: widget.goulotteCurrentEndOrigNotifier,
                      builder: (context, currentEnd, _) {
                        if (_goulotteStartOrig == null || currentEnd == null) return const SizedBox.shrink();
                        return CustomPaint(
                          painter: GoulottePainter(
                            goulotte: LigneGoulotte(_goulotteStartOrig!, currentEnd),
                            scale: widget.scale,
                            offsetX: widget.offsetX,
                            offsetY: widget.offsetY,
                            thicknessOrig: largeurGoulotteOrig,
                            showLine: true, 
                            showNodes: true, // Affiche les nœuds pendant la création
                            primaryColor: widget.theme.colorScheme.primary, // Injecte la couleur du thème
                          ),
                        );
                      },
                    ),
                  );
                } else {
                  // B - Mode interactif nodal (laisse 90% de l'écran libre pour le zoom !)
                  return Stack(
                    children: _buildGoulotteDraggers(goulotte, widget.scale, widget.offsetX, widget.offsetY),
                  );
                }
              }
            ),
          ),

        // Le calque ultime au-dessus de tout : La Loupe de Précision (Magnifier)
        ValueListenableBuilder<Offset?>(
          valueListenable: widget.magnifierPositionNotifier,
          builder: (context, magPos, _) {
            // Si aucune position n'est définie (pas de drag), on ne montre rien
            if (magPos == null) return const SizedBox.shrink();
            
            return Positioned(
              left: magPos.dx - 60, // 60 = moitié de la largeur de la loupe (120/2)
              top: magPos.dy - 130, // Décale vers le haut
              child: RawMagnifier(
                decoration: MagnifierDecoration(
                  shape: CircleBorder(
                    side: BorderSide(color: widget.theme.colorScheme.primary, width: 2), // Bordure colorée avec le thème
                  ),
                  shadows: const [
                    BoxShadow(color: Colors.black26, blurRadius: 8, spreadRadius: 2)
                  ],
                ),
                size: const Size(120, 120),
                magnificationScale: 2.0,
                // Le point focal regarde 70 pixels plus bas que le centre de la loupe
                focalPointOffset: const Offset(0, 70),
              ),
            );
          }
        ),
      ],
    );
  }
}