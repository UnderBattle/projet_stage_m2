import 'package:flutter/material.dart';

class AppTheme {
  // =========================================================================
  // === COULEURS DE LA MARQUE (Chauffage Der Energies) ===
  // =========================================================================
  static const Color primaryColor = Color(0xFF00796B); // Teal profond
  static const Color primaryLight = Color(0xFF48A999);
  static const Color primaryDark = Color(0xFF004C40);
  static const Color accentColor = Color(0xFF00BFA5); // Utilisé pour le Dark Mode

  // =========================================================================
  // === THÈME CLAIR (Light Mode) ===
  // =========================================================================
  static final ThemeData lightTheme = ThemeData(
    brightness: Brightness.light,
    useMaterial3: true,
    colorScheme: ColorScheme.light(
      primary: primaryColor,
      secondary: accentColor,
      surface: const Color(0xFFF5F7FA), // Fond de l'app très légèrement grisé
      onSurface: Colors.black87,        // Texte principal
      error: Colors.red.shade800,
    ),
    scaffoldBackgroundColor: const Color(0xFFF5F7FA),
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
  // === THÈME SOMBRE (Dark Mode) ===
  // =========================================================================
  static final ThemeData darkTheme = ThemeData(
    brightness: Brightness.dark,
    useMaterial3: true,
    colorScheme: ColorScheme.dark(
      primary: accentColor, // En Dark Mode, on utilise une couleur plus vive pour contraster
      secondary: primaryLight,
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
      foregroundColor: Colors.white,
      elevation: 0,
      centerTitle: true,
    ),
    floatingActionButtonTheme: const FloatingActionButtonThemeData(
      backgroundColor: accentColor,
      foregroundColor: Colors.black, // Texte en noir sur fond clair pour la lisibilité
    ),
  );
}