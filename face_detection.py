import cv2
import numpy as np
import os
import urllib.request

# INTÉGRATION : Importation du module de ton collègue Mohamed
import filtres 

def initialiser_modele_dnn():
    """
    Télécharge et charge le modèle DNN en mémoire.
    """
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"

    if not os.path.exists(prototxt_path):
        print("[IA] Téléchargement de l'architecture du réseau...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", prototxt_path)
    if not os.path.exists(model_path):
        print("[IA] Téléchargement des poids du modèle...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", model_path)

    net = cv2.dnn.readNet(model_path, prototxt_path)
    return net

def detecter_visages(image, net, seuil_confiance=0.5):
    """
    Reçoit l'image brute, utilise le prétraitement de Mohamed, et retourne les boîtes des visages.
    """
    # 1. On sauvegarde les dimensions originales AVANT la déformation par Mohamed
    (h, w) = image.shape[:2]
    liste_coordonnees = []

    # 2. COLLABORATION : Utilisation de la fonction développée par Mohamed pour le prétraitement
    blob = filtres.creat_blob(image)
    
    # 3. Exécution de l'intelligence artificielle (Inférence)
    net.setInput(blob)
    detections = net.forward()

    # 4. Extraction et filtrage des résultats
    for i in range(0, detections.shape[2]):
        confiance = detections[0, 0, i, 2]

        if confiance > seuil_confiance:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            liste_coordonnees.append((startX, startY, endX, endY))

    return liste_coordonnees


# BLOC DE TEST INDIVIDUEL 
if __name__ == "__main__":
    print("--- Test du module IA intégrant le prétraitement de Mohamed ---")
    
    modele = initialiser_modele_dnn()
    
    # Utilisation du chemin spécifique demandé
    chemin_test = r"C:\Games\Python\OpenCv\dataset\image_test.jpg" 
    
    # Appel de la fonction de chargement de Mohamed
    image_test = filtres.load_img(chemin_test)
    
    if image_test is not None:
        resultats = detecter_visages(image_test, modele)
        print(f"L'IA a trouvé {len(resultats)} visage(s) ! Coordonnées : {resultats}")
    else:
        print(f"Erreur : Impossible de charger l'image depuis {chemin_test}")
