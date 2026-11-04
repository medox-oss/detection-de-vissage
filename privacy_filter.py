import cv2

def flouter_visages(image, coordonnees):
    image_floutee = image.copy()

    hauteur_image, largeur_image = image.shape[:2]

    for (startX, startY, endX, endY) in coordonnees:

        # Vérifier que les coordonnées restent dans l'image
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(largeur_image, endX)
        endY = min(hauteur_image, endY)

        # Extraire la région du visage
        visage = image_floutee[startY:endY, startX:endX]

        hauteur, largeur = visage.shape[:2]

        # Choisir une taille de flou adaptée au visage
        taille = min(51, hauteur, largeur)

        if taille % 2 == 0:
            taille -= 1

        if taille < 3:
            continue

        # Appliquer le flou gaussien
        visage_floute = cv2.GaussianBlur(
            visage, (taille, taille), 0
        )

        # Remplacer le visage original par sa version floutée
        image_floutee[startY:endY, startX:endX] = visage_floute

    return image_floutee
