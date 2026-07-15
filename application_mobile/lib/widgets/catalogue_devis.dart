import 'package:flutter/material.dart';
import '../services/catalogue_service.dart';

class CatalogueDevis extends StatelessWidget {
  final String categorieSelectionnee;
  final Equipement? modeleSelectionne;
  final bool isProcessing;
  final ValueChanged<String> onCategorieChanged;

  final Function(Equipement, VarianteEquipement?) onModeleSelected;

  const CatalogueDevis({
    super.key,
    required this.categorieSelectionnee,
    required this.modeleSelectionne,
    required this.isProcessing,
    required this.onCategorieChanged,
    required this.onModeleSelected,
  });

  void _afficherPopupVariantes(BuildContext context, Equipement equipement, ThemeData theme) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: theme.cardColor,
          title: Text("Choisir une variante", style: TextStyle(color: theme.colorScheme.onSurface, fontWeight: FontWeight.bold)),
          content: SizedBox(
            width: double.maxFinite,
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: equipement.variantes!.length,
              itemBuilder: (context, index) {
                final variante = equipement.variantes![index];
                return ListTile(
                  contentPadding: EdgeInsets.zero,
                  title: Text(variante.valeur, style: TextStyle(fontWeight: FontWeight.bold, color: theme.colorScheme.primary)),
                  subtitle: Text("${variante.hauteur} x ${variante.largeur} x ${variante.profondeur} mm", style: TextStyle(color: theme.colorScheme.onSurface.withValues(alpha: 0.6))),
                  trailing: Icon(Icons.chevron_right, color: theme.colorScheme.primary),
                  onTap: () {
                    Navigator.pop(context);
                    onModeleSelected(equipement, variante);
                  },
                );
              },
            ),
          ),
        );
      }
    );
  }

  // Méthode pour afficher le panneau d'informations techniques
  void _afficherInfosEquipement(BuildContext context, Equipement equipement, ThemeData theme) {
    showModalBottomSheet(
      context: context,
      backgroundColor: theme.cardColor,
      isScrollControlled: true, // Permet au panneau de prendre la hauteur nécessaire
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(25))),
      builder: (context) {
        return Padding(
          padding: EdgeInsets.only(
            left: 24.0, 
            right: 24.0, 
            top: 24.0, 
            bottom: MediaQuery.of(context).padding.bottom + 24.0
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Titre et bouton de fermeture
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      equipement.nom,
                      style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: theme.colorScheme.onSurface),
                    ),
                  ),
                  IconButton(
                    icon: Icon(Icons.close, color: theme.colorScheme.onSurface.withValues(alpha: 0.5)), 
                    onPressed: () => Navigator.pop(context)
                  ),
                ],
              ),
              Divider(color: theme.dividerColor, thickness: 1.5),
              const SizedBox(height: 16),
              
              // Informations techniques adaptatives (ne s'affichent que si elles existent)
              
              // Affichage intelligent des dimensions (Globale OU par variante)
              if (equipement.variantes != null && equipement.variantes!.isNotEmpty) ...[
                Padding(
                  padding: const EdgeInsets.only(bottom: 8.0),
                  child: Text("Dimensions par variante :", style: TextStyle(fontWeight: FontWeight.w600, color: theme.colorScheme.onSurface.withValues(alpha: 0.6))),
                ),
                ...equipement.variantes!.map((v) => _buildInfoRow(
                  Icons.straighten, 
                  v.valeur, 
                  "${v.hauteur} x ${v.largeur} x ${v.profondeur} mm", 
                  theme
                )),
              ] else ...[
                _buildInfoRow(Icons.straighten, "Dimensions (H x L x P)", "${equipement.hauteur} x ${equipement.largeur} x ${equipement.profondeur} mm", theme),
              ],
              
              if (equipement.poids != null)
                _buildInfoRow(Icons.scale, "Poids", "${equipement.poids} kg", theme),
                
              if (equipement.puissances != null && equipement.puissances!.isNotEmpty)
                _buildInfoRow(Icons.bolt, "Puissances disponibles", equipement.puissances!.join('\n'), theme),
                
              if (equipement.prixMin != null && equipement.prixMax != null)
                _buildInfoRow(Icons.euro, "Fourchette de prix public", "${equipement.prixMin!.toStringAsFixed(0)} €  —  ${equipement.prixMax!.toStringAsFixed(0)} €", theme),
                
              const SizedBox(height: 10),
            ],
          ),
        );
      },
    );
  }

  // Widget utilitaire pour dessiner une belle ligne d'information avec icône
  Widget _buildInfoRow(IconData icon, String titre, String valeur, ThemeData theme) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: theme.colorScheme.primary.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, size: 24, color: theme.colorScheme.primary),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(titre, style: TextStyle(fontWeight: FontWeight.w600, color: theme.colorScheme.onSurface.withValues(alpha: 0.6), fontSize: 14)),
                const SizedBox(height: 4),
                Text(valeur, style: TextStyle(color: theme.colorScheme.onSurface, fontSize: 16, height: 1.4)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final catalogueGlobal = CatalogueService().catalogueGlobal;
    final theme = Theme.of(context); // NOUVEAU : Récupération du thème actif

    return Container(
      height: 190,
      padding: const EdgeInsets.only(top: 15, bottom: 10),
      // STYLE : Bottom sheet moderne avec coins arrondis et ombre douce
      decoration: BoxDecoration(
        color: theme.cardColor, // Utilise la couleur de carte du thème
        boxShadow: [BoxShadow(color: theme.shadowColor.withValues(alpha: 0.08), blurRadius: 20, offset: const Offset(0, -5))],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            height: 40,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              itemCount: catalogueGlobal.keys.length,
              itemBuilder: (context, index) {
                String catName = catalogueGlobal.keys.elementAt(index);
                bool isSelected = categorieSelectionnee == catName;
                
                return Padding(
                  padding: const EdgeInsets.only(right: 8.0),
                  child: ChoiceChip(
                    label: Text(
                      catName, 
                      style: TextStyle(
                        fontWeight: isSelected ? FontWeight.bold : FontWeight.w500, 
                        color: isSelected ? theme.colorScheme.primary : theme.colorScheme.onSurface.withValues(alpha: 0.7)
                      )
                    ),
                    selected: isSelected,
                    selectedColor: theme.colorScheme.primary.withValues(alpha: 0.15),
                    backgroundColor: theme.scaffoldBackgroundColor, // S'adapte au mode sombre
                    side: BorderSide.none, // Retrait des bordures pour un effet "pilule" moderne
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                    showCheckmark: false, // Plus propre sans le V de validation
                    onSelected: (bool selected) {
                      if (selected && !isProcessing) {
                        // =========================================================================
                        // === OPTIMISATION RAM : Nettoyage dynamique du cache vidéo             ===
                        // =========================================================================
                        for (var entry in catalogueGlobal.entries) {
                          if (entry.key != catName) {
                            // On libère la RAM pour les équipements des catégories non affichées
                            for (var equipement in entry.value) {
                              AssetImage(equipement.chemin).evict();
                            }
                          } else {
                            // On s'assure que la nouvelle catégorie est bien pré-chargée en RAM
                            for (var equipement in entry.value) {
                              precacheImage(AssetImage(equipement.chemin), context);
                            }
                          }
                        }
                        
                        onCategorieChanged(catName);
                      }
                    },
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 15),
          
          Expanded(
            child: catalogueGlobal[categorieSelectionnee]!.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.build_circle_outlined, size: 40, color: theme.colorScheme.onSurface.withValues(alpha: 0.3)),
                        const SizedBox(height: 8),
                        Text("Cette catégorie sera ajoutée prochainement", style: TextStyle(color: theme.colorScheme.onSurface.withValues(alpha: 0.5), fontStyle: FontStyle.italic, fontWeight: FontWeight.w500)),
                      ],
                    ),
                  )
                : ListView.builder(
                    scrollDirection: Axis.horizontal,
                    itemCount: catalogueGlobal[categorieSelectionnee]!.length,
                    itemBuilder: (context, index) {
                      final equipement = catalogueGlobal[categorieSelectionnee]![index];
                      final bool isSelected = modeleSelectionne == equipement;

                      return GestureDetector(
                        onTap: () {
                          if (isProcessing) return;
                          
                          // VÉRIFICATION DES VARIANTES AVANT L'INCRUSTATION
                          if (equipement.variantes != null && equipement.variantes!.isNotEmpty) {
                            _afficherPopupVariantes(context, equipement, theme);
                          } else {
                            onModeleSelected(equipement, null);
                          }
                        },
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 200),
                          width: 120,
                          margin: EdgeInsets.only(left: 16.0, bottom: 8.0, top: 4.0, right: index == catalogueGlobal[categorieSelectionnee]!.length - 1 ? 16.0 : 0.0),
                          // STYLE : Carte équipement sublimée avec ombres douces et bordures fines
                          decoration: BoxDecoration(
                            color: isSelected ? theme.colorScheme.primary.withValues(alpha : 0.05) : theme.cardColor,
                            border: Border.all(color: isSelected ? theme.colorScheme.primary : theme.dividerColor, width: 2),
                            borderRadius: BorderRadius.circular(15),
                            boxShadow: [
                              BoxShadow(
                                color: isSelected ? theme.colorScheme.primary.withValues(alpha : 0.15) : theme.shadowColor.withValues(alpha: 0.04), 
                                blurRadius: isSelected ? 12 : 8, 
                                offset: const Offset(0, 4)
                              )
                            ],
                          ),
                          // Stack utilisé pour superposer le bouton 'i' sur la carte
                          child: Stack(
                            children: [
                              Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  crossAxisAlignment: CrossAxisAlignment.center,
                                  children: [
                                    Expanded(child: Padding(padding: const EdgeInsets.all(8.0), child: Image.asset(equipement.chemin, fit: BoxFit.contain))),
                                    Padding(
                                      padding: const EdgeInsets.symmetric(horizontal: 4.0, vertical: 8.0),
                                      child: Text(
                                        equipement.nom, 
                                        style: TextStyle(
                                          fontSize: 12, 
                                          fontWeight: isSelected ? FontWeight.bold : FontWeight.w500, 
                                          color: isSelected ? theme.colorScheme.primary : theme.colorScheme.onSurface
                                        ), 
                                        textAlign: TextAlign.center, 
                                        maxLines: 2,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              
                              // Bouton 'Info' (i) en haut à droite
                              Positioned(
                                top: 2,
                                right: 2,
                                child: GestureDetector(
                                  onTap: () {
                                    if (isProcessing) return;
                                    _afficherInfosEquipement(context, equipement, theme);
                                  },
                                  child: Container(
                                    padding: const EdgeInsets.all(4.0),
                                    // Petit fond translucide pour que l'icône soit toujours lisible par-dessus l'image de la clim
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: theme.cardColor.withValues(alpha: 0.8),
                                    ),
                                    child: Icon(
                                      Icons.info_outline,
                                      size: 20,
                                      color: isSelected ? theme.colorScheme.primary : theme.colorScheme.secondary,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}