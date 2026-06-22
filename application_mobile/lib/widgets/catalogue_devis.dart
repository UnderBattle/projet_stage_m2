import 'package:flutter/material.dart';
import '../services/catalogue_service.dart';

class CatalogueDevis extends StatelessWidget {
  final String categorieSelectionnee;
  final Equipement? modeleSelectionne;
  final bool isProcessing;
  final ValueChanged<String> onCategorieChanged;
  final ValueChanged<Equipement> onModeleSelected;

  const CatalogueDevis({
    super.key,
    required this.categorieSelectionnee,
    required this.modeleSelectionne,
    required this.isProcessing,
    required this.onCategorieChanged,
    required this.onModeleSelected,
  });

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
                          onModeleSelected(equipement);
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
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
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
                                  maxLines: 2
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