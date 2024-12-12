import os
import cv2
import hdf5storage
import argparse

from utils.utils import *
from config import cfg
from train_transunet import TransUNetSeg


#---------------------
#     Parameters
#---------------------


# Full brain
FULL_CROP_PARAMETERS = [180, 200, 318, 180, 200, 62]

# 128^3 sub-volume in the top-right part of the volume
CROP_PARAM_TR = [139, -61, 149, -11, 189, -21] 

# 128^3 sub-volume in the top-left part of the volume
CROP_PARAM_TL = [150, 189, 139, -22, -61, -11]


# Models
MODEL_PATH_TL = './models/model_config_TL.pth'
MODEL_PATH_TR = './models/model_config_TR.pth'

# Thresholds for binarization
SEUIL_ANOMALY_TL = 0.4
SEUIL_ANOMALY_TR = 0.3

# Threshold on lesion size
LESION_PERCENTAGE = 0.9 # percentage
MIN_LESION_SIZE = 20 # voxels

# Select computing device
device = 'cpu:0'
if torch.cuda.is_available():
    device = 'cuda:0'
    print('Cuda available :', device)




#---------------------------
#    Inference Functions
#---------------------------


def read_and_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_torch = cv2.resize(img, (cfg.transunet.img_dim, cfg.transunet.img_dim))
    img_torch = img_torch / 255.
    img_torch = img_torch.transpose((2, 0, 1))
    img_torch = np.expand_dims(img_torch, axis=0)
    img_torch = torch.from_numpy(img_torch.astype('float32')).to(device)

    return img, img_torch


def infer(volume, transunet, treshold=0.5, merged=True):

    preds = np.zeros(volume.shape, dtype=np.uint8)

    for i in range(volume.shape[2]):
        # coronal_slice = volume[:, :, i]

        if i != 0 and i != (volume.shape[2]-1):
            coronal_slice = volume[:,:,i-1:i+2] # 3 consecutive slices
            # print(i, coronal_slice.shape, element_mask.shape)
        elif i == 0:
            coronal_slice = volume[:,:,:i+3] # 3 consecutive slices
            # print(i, coronal_slice.shape)
        else:
            coronal_slice = volume[:,:,i-2:]
            # print(i, coronal_slice.shape)

        img, img_torch = read_and_preprocess(coronal_slice)
        # print(img_torch.shape)
        with torch.no_grad():
            pred_mask = transunet.model(img_torch)
            # print(pred_mask)
            pred_mask = torch.sigmoid(pred_mask)
            # print(pred_mask)
            pred_mask = pred_mask.detach().cpu().numpy().transpose((0, 2, 3, 1))

        orig_h, orig_w = img.shape[:2]
        pred_mask = cv2.resize(pred_mask[0, ...], (orig_w, orig_h))
        pred_mask = thresh_func(pred_mask, thresh=treshold)

        preds[:, :, i] = pred_mask

    return preds



if __name__ == "__main__":
    # Configuration de l'analyse des arguments
    parser = argparse.ArgumentParser(description="Script de prédiction des PWML en ETF 3D.")
    parser.add_argument('-in', '--input', type=str, required=True, help="Chemin vers le fichier d'entrée.")
    parser.add_argument('-out', '--output', type=str, required=True, help="Chemin vers le dossier de sortie.")
    
    args = parser.parse_args()

    #-------------------
    #    Load volume
    #-------------------


    # Load volumes
    image = hdf5storage.loadmat(args.input)['data_repcom']

    print('\nVol.shape :', image.shape)

    # Add padding before crop
    image = np.pad(image, ((100, 100), (100, 100), (100, 100)), 'constant')

    # Crop TR128 volume and mask to get the same shapes
    x_min, y_min, z_min, x_max, y_max, z_max = CROP_PARAM_TR
    image_TR = crop_center(image,x_min, y_min, z_min, x_max, y_max, z_max )

    # Crop TL128 volume and mask to get the same shapes
    x_min, y_min, z_min, x_max, y_max, z_max = CROP_PARAM_TL
    image_TL = crop_center(image,x_min, y_min, z_min, x_max, y_max, z_max )


    #-----------------------------------
    #     Make TransUNet prediction
    #-----------------------------------
    

    # Prediction TR
    transunet = TransUNetSeg(device)
    transunet.load_model(os.path.join(MODEL_PATH_TR))
    pred_mask_TR = infer(image_TR, transunet, treshold=SEUIL_ANOMALY_TR)
    print(f'\nPost-processing prediction TR128 (thresholding above {MIN_LESION_SIZE}):')
    pred_mask_TR = remove_small_lesions_threshold(pred_mask_TR, min_lesion_size=MIN_LESION_SIZE)

    # Prediction TL
    transunet = TransUNetSeg(device)
    transunet.load_model(os.path.join(MODEL_PATH_TL))
    pred_mask_TL = infer(image_TL, transunet, treshold=SEUIL_ANOMALY_TL)
    print(f'\nPost-processing prediction TL128 (thresholding above {MIN_LESION_SIZE}):')
    pred_mask_TL = remove_small_lesions_threshold(pred_mask_TL, min_lesion_size=MIN_LESION_SIZE)


    # Save intermediate outputs
    PRED_PATH = args.output
    os.makedirs(PRED_PATH, exist_ok=True)

    np.save(os.path.join(PRED_PATH, f'pred_TU_3S_TR128_t-{SEUIL_ANOMALY_TR}.npy'), pred_mask_TR.astype(np.uint8))
    np.save(os.path.join(PRED_PATH, f'pred_TU_3S_TL128_t-{SEUIL_ANOMALY_TL}.npy'), pred_mask_TL.astype(np.uint8))
    print(f'\n[INFO] Predictions from TransUNet saved here : {PRED_PATH}')
