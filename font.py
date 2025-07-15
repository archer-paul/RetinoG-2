import tkinter.font as tkFont
import urllib.request
import os

def setup_google_fonts():
    """Configure les polices Google si disponibles"""
    try:
        # Vérifier si Google Sans est disponible
        available_fonts = tkFont.families()
        if 'Google Sans' not in available_fonts:
            # Fallback vers des polices similaires
            return {
                'title_large': ('Segoe UI', 32, 'normal'),
                'title': ('Segoe UI', 24, 'normal'),
                'subtitle': ('Segoe UI', 18, 'normal'),
                'body': ('Segoe UI', 14, 'normal'),
                'small': ('Segoe UI', 12, 'normal'),
                'tiny': ('Segoe UI', 10, 'normal'),
                'mono': ('Consolas', 11, 'normal')
            }
    except:
        pass
    
    return None