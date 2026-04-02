import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os

from src.traitement_image import incruster_climatisation

# ==========================================
# CONSTANTES DU PROJET
# ==========================================
AUTOCOLLANT_TAILLE_REELLE = (100, 50) # Hauteur, Largeur en mm
TAKAO_PLUS_DIMENSION_REELLE = (270, 798, 240) # Hauteur, Largeur, Profondeur en mm

CATALOGUE_CLIMS = {
    "Takao Plus Blanc": "./installations/clim_takao_plus/8e74c5374539-takao-plus-blanc-face-atlantic.png",
    "Takao Plus Noir": "./installations/clim_takao_plus/baae79054b9d-takao-plus-noir-face-atlantic.png",
}

PTS_AUTOCOLLANT_DEFAUT = np.float32([[1128, 1128], [1208, 1129], [1206, 1288], [1128, 1287]]) 
# Dimensions maximales de la fenêtre sur ton écran PC
MAX_WIDTH = 1200
MAX_HEIGHT = 700

# ==========================================
# L'INTERFACE GRAPHIQUE
# ==========================================
class SimulateurApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Labo de test : Incrustation Interactive")

        # Variables de mémoire
        self.mur_img_cv2 = None
        self.clim_img_cv2 = None
        self.clim_resized_pil = None

        # Facteur d'échelle pour l'affichage écran
        self.ratio_affichage = 1.0

        # Dimensions réelles (pour OpenCV)
        self.clim_w_reel = 0
        self.clim_h_reel = 0

        # Dimensions et Position à l'écran (pour Tkinter)
        self.clim_w_disp = 0
        self.clim_h_disp = 0
        self.clim_x_disp = 0
        self.clim_y_disp = 0

        # Mécanique de Drag & Drop
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.setup_ui()

        # Image par défaut
        default_wall = './img_test/IMG_20260401_090116.jpg'
        if os.path.exists(default_wall):
            self.charger_mur(default_wall)

        self.charger_clim(list(CATALOGUE_CLIMS.keys())[0])

    def setup_ui(self):
        frame_controls = tk.Frame(self.root, padx=10, pady=10)
        frame_controls.pack(side=tk.TOP, fill=tk.X)

        btn_charger = tk.Button(frame_controls, text="Choisir une image de mur", command=self.choisir_mur, bg="lightblue")
        btn_charger.pack(side=tk.LEFT, padx=5)

        tk.Label(frame_controls, text="Sélectionner la clim :").pack(side=tk.LEFT, padx=15)

        self.combo_clim = ttk.Combobox(frame_controls, values=list(CATALOGUE_CLIMS.keys()), state="readonly", width=25)
        self.combo_clim.current(0)
        self.combo_clim.pack(side=tk.LEFT, padx=5)
        self.combo_clim.bind("<<ComboboxSelected>>", self.on_clim_changed)

        self.canvas = tk.Canvas(self.root, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.canvas_clim_id = self.canvas.create_image(0, 0, anchor=tk.NW)

    def choisir_mur(self):
        chemin = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if chemin:
            self.charger_mur(chemin)

    def charger_mur(self, chemin):
        img = cv2.imread(chemin, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Erreur", f"Impossible de charger l'image : {chemin}")
            return
        self.mur_img_cv2 = img

        h, w = img.shape[:2]
        ratio_w = MAX_WIDTH / w
        ratio_h = MAX_HEIGHT / h
        # On prend le plus petit ratio pour s'assurer que l'image rentre entièrement
        self.ratio_affichage = min(ratio_w, ratio_h, 1.0) # Ne pas dépasser 1.0

        disp_w = int(w * self.ratio_affichage)
        disp_h = int(h * self.ratio_affichage)
        self.canvas.config(width=disp_w, height=disp_h)
        
        # On réinitialise la position pour la nouvelle image
        self.clim_x_disp = 0
        self.clim_y_disp = 0

        self.afficher_fond(img)
        self.calculer_dimensions_clim() # Recalcule la clim pour la nouvelle échelle
        self.mettre_a_jour_rendu()

    def on_clim_changed(self, event):
        self.charger_clim(self.combo_clim.get())

    def charger_clim(self, nom_modele):
        chemin = CATALOGUE_CLIMS[nom_modele]
        img = cv2.imread(chemin, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Erreur", f"Impossible de charger la clim : {chemin}")
            return
        self.clim_img_cv2 = img
        self.calculer_dimensions_clim()
        self.mettre_a_jour_rendu()

    def calculer_dimensions_clim(self):
        if self.mur_img_cv2 is None or self.clim_img_cv2 is None:
            return

        # Taille Réelle
        pt_hg = PTS_AUTOCOLLANT_DEFAUT[0]
        pt_hd = PTS_AUTOCOLLANT_DEFAUT[1]
        dist_px = np.sqrt((pt_hd[0] - pt_hg[0])**2 + (pt_hd[1] - pt_hg[1])**2)
        ratio = dist_px / AUTOCOLLANT_TAILLE_REELLE[1]

        self.clim_w_reel = int(TAKAO_PLUS_DIMENSION_REELLE[1] * ratio)
        self.clim_h_reel = int(TAKAO_PLUS_DIMENSION_REELLE[0] * ratio)

        # Taille d'affichage
        self.clim_w_disp = int(self.clim_w_reel * self.ratio_affichage)
        self.clim_h_disp = int(self.clim_h_reel * self.ratio_affichage)

        # Placement initial par défaut (converti pour l'écran)
        if self.clim_x_disp == 0 and self.clim_y_disp == 0:
            self.clim_x_disp = int(pt_hg[0] * self.ratio_affichage)
            self.clim_y_disp = int(pt_hg[1] * self.ratio_affichage)

        # On crée l'image légère pour la souris
        clim_resized = cv2.resize(self.clim_img_cv2, (self.clim_w_disp, self.clim_h_disp), interpolation=cv2.INTER_AREA)
        clim_rgba = cv2.cvtColor(clim_resized, cv2.COLOR_BGRA2RGBA)
        self.clim_resized_pil = ImageTk.PhotoImage(image=Image.fromarray(clim_rgba))
        self.canvas.itemconfig(self.canvas_clim_id, image=self.clim_resized_pil)

    def afficher_fond(self, img_cv2):
        """ Redimensionne n'importe quelle image OpenCV pour l'affichage écran """
        disp_w = int(img_cv2.shape[1] * self.ratio_affichage)
        disp_h = int(img_cv2.shape[0] * self.ratio_affichage)
        img_resized = cv2.resize(img_cv2, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        self.img_tk_fond = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.canvas.itemconfig(self.canvas_img_id, image=self.img_tk_fond)

    def mettre_a_jour_rendu(self):
        if self.mur_img_cv2 is None or self.clim_img_cv2 is None:
            return

        self.canvas.itemconfig(self.canvas_clim_id, state="hidden")

        real_x = int(self.clim_x_disp / self.ratio_affichage)
        real_y = int(self.clim_y_disp / self.ratio_affichage)

        try:
            resultat = incruster_climatisation(
                mur_img=self.mur_img_cv2,
                clim_img=self.clim_img_cv2,
                pts_autocollant=PTS_AUTOCOLLANT_DEFAUT,
                dim_clim_mm=TAKAO_PLUS_DIMENSION_REELLE,
                dim_autocollant_mm=AUTOCOLLANT_TAILLE_REELLE,
                position_cible=(real_x, real_y)
            )
            self.afficher_fond(resultat)
            
        except ValueError as e:
            print("Action annulée :", e)
            self.afficher_fond(self.mur_img_cv2)
            self.canvas.itemconfig(self.canvas_clim_id, image=self.clim_resized_pil, state="normal")
            self.canvas.coords(self.canvas_clim_id, self.clim_x_disp, self.clim_y_disp)

    # ==========================================
    # ÉVÈNEMENTS SOURIS
    # ==========================================
    def on_press(self, event):
        if self.clim_x_disp <= event.x <= self.clim_x_disp + self.clim_w_disp and \
           self.clim_y_disp <= event.y <= self.clim_y_disp + self.clim_h_disp:
            self.is_dragging = True
            self.drag_start_x = event.x - self.clim_x_disp
            self.drag_start_y = event.y - self.clim_y_disp

            self.afficher_fond(self.mur_img_cv2)
            self.canvas.itemconfig(self.canvas_clim_id, image=self.clim_resized_pil, state="normal")
            self.canvas.coords(self.canvas_clim_id, self.clim_x_disp, self.clim_y_disp)

    def on_drag(self, event):
        if self.is_dragging:
            self.clim_x_disp = event.x - self.drag_start_x
            self.clim_y_disp = event.y - self.drag_start_y
            self.canvas.coords(self.canvas_clim_id, self.clim_x_disp, self.clim_y_disp)

    def on_release(self, event):
        if self.is_dragging:
            self.is_dragging = False
            self.mettre_a_jour_rendu()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulateurApp(root)
    root.mainloop()