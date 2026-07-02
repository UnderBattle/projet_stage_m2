import 'package:flutter/material.dart';

class AppTheme {
  static const Color primaryColor = Color(0xFFFF7A45); // Orange vif mais doux (Chauffage)
  static const Color primaryLight = Color(0xFFFFAB91); // Orange très clair
  static const Color primaryDark = Color(0xFFE64A19);  // Orange soutenu (Bordures/Ombres)
  static const Color accentColor = Color(0xFF5C85C8);  // Bleu clair acier du logo (Energies)

  // =========================================================================
  // === THÈME CLAIR (Light Mode)                                          ===
  // =========================================================================
  static final ThemeData lightTheme = ThemeData(
    brightness: Brightness.light,
    useMaterial3: true,
    colorScheme: ColorScheme.light(
      primary: primaryColor,
      secondary: accentColor,
      surface: const Color(0xFFF9FAFB), // Blanc très légèrement grisé pour le fond
      onSurface: Colors.black87,        // Texte principal
      error: Colors.red.shade800,
    ),
    scaffoldBackgroundColor: const Color(0xFFF9FAFB),
    cardColor: Colors.white, // Fond des bottom sheets et cartes
    dividerColor: Colors.grey.shade200,
    shadowColor: Colors.black,
    appBarTheme: const AppBarTheme(
      backgroundColor: primaryColor,
      foregroundColor: Colors.white,
      elevation: 0,
      centerTitle: true,
    ),
    floatingActionButtonTheme: const FloatingActionButtonThemeData(
      backgroundColor: primaryColor,
      foregroundColor: Colors.white,
    ),
  );

  // =========================================================================
  // === THÈME SOMBRE (Dark Mode)                                          ===
  // =========================================================================
  static final ThemeData darkTheme = ThemeData(
    brightness: Brightness.dark,
    useMaterial3: true,
    colorScheme: ColorScheme.dark(
      primary: primaryColor, // L'orange clair en dark mode
      secondary: accentColor, // Le bleu doux aussi
      surface: const Color(0xFF1E1E1E), // Gris très foncé (Surface des cartes)
      onSurface: Colors.white,          // Texte principal
      error: Colors.redAccent,
    ),
    scaffoldBackgroundColor: const Color(0xFF121212), // Fond de l'app noir profond
    cardColor: const Color(0xFF242424), // Fond des bottom sheets
    dividerColor: Colors.grey.shade800,
    shadowColor: Colors.black, // L'ombre reste noire
    appBarTheme: const AppBarTheme(
      backgroundColor: Color(0xFF1E1E1E),
      foregroundColor: primaryLight, // Texte orange clair sur fond noir
      elevation: 0,
      centerTitle: true,
    ),
    floatingActionButtonTheme: const FloatingActionButtonThemeData(
      backgroundColor: primaryColor,
      foregroundColor: Colors.white, 
    ),
  );
}