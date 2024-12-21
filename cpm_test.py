import argparse
import os
import torch
from torchvision import transforms
from torch.autograd import Variable

import cpm_model
import cpm_data  # pour les classes Scale, ToTensor, etc.

#############################################
#      FONCTION POUR GÉNÉRER CENTER MAP     #
#############################################
import numpy as np
import cv2
from PIL import Image

def read_image(path):
    """
    Lit une image depuis 'path' et la retourne sous forme de tableau numpy (H, W, C).
    """
    with Image.open(path) as img:
        # On convertit en RGB pour être sûr d'avoir 3 canaux
        img = img.convert('RGB')
        # On convertit en array float32 (ou float64 si vous préférez)
        img = np.array(img, dtype=np.float32)
        # Selon la logique de votre pipeline, vous pouvez éventuellement la normaliser
        # (e.g. diviser par 255) si vos transformations s'y attendent.
        # img /= 255.0
    return img

def generate_center_map(img_width, img_height, sigma=21):
    """
    Génère une center map 2D gaussienne au centre de l'image.
    La shape sera (1, H, W) pour ressembler à un channel unique.
    """
    center_x = img_width // 2
    center_y = img_height // 2
    grid_x = np.arange(img_width)
    grid_y = np.arange(img_height)[:, None]
    
    # Gaussian = exp(-((x - cx)^2 + (y - cy)^2) / (2*sigma^2))
    # On veut un pic max = 1 au centre
    # shape finale: (H, W)
    center_map = np.exp(-((grid_x - center_x)**2 + (grid_y - center_y)**2) / (2.0 * sigma**2))
    
    # On rajoute la dimension "channel"
    center_map = center_map[None, ...]  # shape: (1, H, W)
    return center_map

#############################################
#              FONCTION INFÉRENCE           #
#############################################
def predict_image(model, image_path, device='cpu', image_size=(368, 368)):
    # 1) Charger l'image
    original_image = read_image(image_path)  
    # Vous pouvez utiliser cv2.imread / PIL / etc. 
    # Ici cpm_data.read_image est un helper (vous pouvez faire autrement).

    # 2) Appliquer les transformations
    # En entraînement, on faisait: Scale -> RandomHSV -> ToTensor
        # En test, on saute le RandomHSV pour ne pas déformer les couleurs
    transform_test = transforms.Compose([
        cpm_data.ScaleImageOnly(image_size[0], image_size[1]),
        cpm_data.ToTensorImageOnly()
    ])

    original_image = read_image(image_path)        # lit l'image en np.array ou PIL
    input_image = transform_test(original_image)   # => Tensor (C, H, W)
    input_image = input_image.unsqueeze(0)         # => (1, C, H, W)


    # 3) Générer la center map
    center_map_np = generate_center_map(image_size[0], image_size[1], sigma=21)
    center_map_tensor = torch.from_numpy(center_map_np).float().unsqueeze(0) 
    # .unsqueeze(0) pour batch dimension -> shape: (1, 1, H, W)

    # 4) Passer sur le device choisi
    input_image = input_image.to(device)
    center_map_tensor = center_map_tensor.to(device)

    # 5) Mettre le modèle en mode évaluation
    model.eval()
    
    # 6) Prédiction
    with torch.no_grad():
        outputs = model(input_image, center_map_tensor)  # [1, 6, nb_keypoints, H, W]
        pred_final = outputs[:, -1, :, :, :]            # on prend la dernière sortie -> [1, nb_keypoints, H, W]
    
    # 7) Interpréter la heatmap
    # Pour chaque keypoint, on peut localiser le maximum dans la heatmap.
    # On obtient un couple (x, y) par keypoint.
    pred_np = pred_final.cpu().numpy()  # shape: (1, nb_keypoints, H, W)
    keypoints = []
    for i in range(pred_np.shape[1]):
        heatmap = pred_np[0, i, :, :]   # heatmap du i-ème keypoint
        # On localise le maximum
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        keypoints.append((x, y))  # (x, y) dans l'espace 368x368
    
    return keypoints, pred_final

#############################################
#                MAIN DEMO                  #
#############################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Chemin vers une image sur laquelle prédire')
    parser.add_argument('--model', required=True, help='Chemin vers un checkpoint .pth')
    parser.add_argument('--use-cuda', action='store_true', default=False)
    args = parser.parse_args()

    device = 'cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu'

    # 1) Charger le modèle
    #   On doit recréer le même "cpm_model.CPM(...)" que dans le script de training.
    #   Mettez le bon nombre de keypoints (ex: 14, 16, 21, etc. selon votre dataset).
    #   S'il y a un paramètre "num_keypoints" dans LSPDataset, réutilisez la même valeur.
    num_keypoints = 14  # par exemple
    model = cpm_model.CPM(num_keypoints)

    # 2) Charger les poids depuis le checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # 3) Exécuter la prédiction
    keypoints, pred_heatmaps = predict_image(model, args.image, device=device)

    # 4) Afficher / post-traiter les keypoints
    print("Keypoints détectés (x, y) dans l'image 368x368 (après resize) :")
    for idx, (x, y) in enumerate(keypoints):
        print(f"  Kp {idx} : ({x}, {y})")

    # 5) Si besoin, on peut repasser (x, y) au repère de l'image originale
    #    si on veut dessiner par-dessus. Il suffit d'appliquer le ratio 
    #    (width_original / 368, height_original / 368).
    #    Dans cpm_data.Scale, vous voyez comment était gérée la mise à l’échelle.
    #
    #    Ex: 
    #    height_original, width_original = original_image.shape[:2]
    #    scale_w = width_original / 368.0
    #    scale_h = height_original / 368.0
    #    x_real = x * scale_w
    #    y_real = y * scale_h
