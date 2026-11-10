"""
main.py : application finale (assemblage des modules du groupe).

Pipeline : charger l'image -> détecter les visages -> flouter -> afficher.

Utilisation :
    python main.py chemin_image.jpg      (une image)
    python main.py dossier_images        (toutes les images d'un dossier)
"""
import os
import sys

import cv2

import filtres                                                   # Mohamed
from face_detection import initialiser_modele_dnn, detecter_visages   # Abdellah
from privacy_filter import flouter_visages                       # Fatimata

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DOSSIER_RESULTATS = "resultats"


def reduire_pour_affichage(image, hauteur_max=700):
    """Réduit l'image pour qu'elle tienne à l'écran (affichage seulement)."""
    h, w = image.shape[:2]
    if h <= hauteur_max:
        return image
    echelle = hauteur_max / h
    return cv2.resize(image, (int(w * echelle), int(h * echelle)))


def traiter_image(chemin, modele):
    """Traite une image. Retourne (originale, floutee) ou None en cas d'erreur."""
    # 1. Charger l'image (load_img retourne None si le fichier est introuvable)
    image = filtres.load_img(chemin)
    if image is None:
        print(f"Erreur : image introuvable ou illisible : {chemin}")
        return None

    # 2. Détecter les visages (liste de boîtes : startX, startY, endX, endY)
    boites = detecter_visages(image, modele)
    print(f"{os.path.basename(chemin)} : {len(boites)} visage(s) détecté(s)")

    # 3. Flouter les régions détectées
    floutee = flouter_visages(image, boites)

    # 4. Sauvegarder le résultat (utile pour l'évaluation des résultats)
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    nom = os.path.splitext(os.path.basename(chemin))[0]
    cv2.imwrite(os.path.join(DOSSIER_RESULTATS, nom + "_floute.jpg"), floutee)

    return image, floutee


def afficher(originale, floutee):
    """Affiche l'image originale et l'image modifiée (touche = suivant)."""
    cv2.imshow("Image originale", reduire_pour_affichage(originale))
    cv2.imshow("Image floutee", reduire_pour_affichage(floutee))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(chemin):
    # Charger le modèle DNN une seule fois (téléchargé au premier lancement)
    try:
        modele = initialiser_modele_dnn()
    except Exception as erreur:
        print(f"Erreur : impossible de charger le modèle DNN : {erreur}")
        return

    # Liste des images à traiter : un fichier ou tout un dossier
    if os.path.isdir(chemin):
        fichiers = [os.path.join(chemin, f) for f in sorted(os.listdir(chemin))
                    if f.lower().endswith(EXTENSIONS)]
        if not fichiers:
            print(f"Erreur : aucune image trouvée dans {chemin}")
            return
    else:
        fichiers = [chemin]

    for fichier in fichiers:
        resultat = traiter_image(fichier, modele)
        if resultat is not None:
            afficher(*resultat)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python main.py chemin_image.jpg  (ou un dossier)")
    else:
        main(sys.argv[1])


