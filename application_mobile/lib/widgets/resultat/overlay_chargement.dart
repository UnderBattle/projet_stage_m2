import 'dart:ui';
import 'package:flutter/material.dart';

class OverlayChargement extends StatelessWidget {
  final String message;

  const OverlayChargement({
    super.key,
    required this.message,
  });

  @override
  Widget build(BuildContext context) {
    return BackdropFilter(
      filter: ImageFilter.blur(sigmaX: 5.0, sigmaY: 5.0),
      child: Container(
        color: Colors.black.withValues(alpha: 0.4),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const CircularProgressIndicator(color: Colors.white),
              const SizedBox(height: 20),
              Text(
                message, 
                style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w500), 
                textAlign: TextAlign.center
              ),
            ],
          )
        )
      ),
    );
  }
}