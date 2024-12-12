import numpy as np
import cc3d
import cv2
import torch



#############################
#   Traitement des donnees  #
#############################



def vox2mm3(vol):
    """
    Convert number of voxels to volume in mm^3
    """
    voxel_volume = 0.168**3
    return vol * voxel_volume



def crop_center(img,cropx_i,cropy_i,cropz_i,cropx_e,cropy_e,cropz_e):
    x,y,z = img.shape[0],img.shape[1],img.shape[2]
    
    startx = x//2-(cropx_i)
    starty = y//2-(cropy_i)
    startz = z//2-(cropz_i)
    
    endx = x//2+(cropx_e)
    endy = y//2+(cropy_e)
    endz = z//2+(cropz_e)  
    return img[startx:endx,starty:endy,startz:endz]



##############################
#    Seuillage des lésions   #
##############################



def remove_small_lesions(mask, percentage=0.9):
    """
    Only keep 90% of the biggest lesions from a segmentation map (3D Mask)
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    lc = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesion found after crop.')
        return mask, None, None
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # Total number of lesion voxels
    lesion_total = np.sum(mask)
    # Initiate cumulated lesion volume
    volume = 0
    
    # print('LESION TOTAL :', lesion_total)
    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # If the biggest lesion is larger than 90% of the lesional volume
        if lesions[k] >= lesion_total*percentage and c == 0 and N > 1:
            # print(k, ': 1ST IF')
            smallest_lesion_size = lesions[k]
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc +=1
            c = 1
        # print(k, lesions[k])
        volume += lesions[k]
        # Assure that we keep 90% of the biggest lesions by the end
        if volume <= lesion_total*percentage and N > 1: # MODIF FLORA FROM < TO <=
            # print(k, '2ND IF')
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            # print('keep lesion', k, volume)
            
    # Condition to keep if only 1 lesion
    if N == 1:
        # print(k, '3RD IF')
        lesion_cluster = np.where(output == 1)
        new_mask[lesion_cluster] = 1
        lc += 1
    
    if N > 1:
        smallest_lesion_size = lesion_cluster[0].shape[0]
    else: # Only 1 lesion
        smallest_lesion_size = lesion_total

    print(f'{lc} lesions kept over {N} in total (min lesion size = {smallest_lesion_size} voxels or {round(vox2mm3(smallest_lesion_size), 2)} mm3)')

    return new_mask, smallest_lesion_size, round(vox2mm3(smallest_lesion_size), 2)



def remove_small_lesions_threshold(mask, min_lesion_size=10):
    """
    Only keep lesions bigger than `min_lesion_size` from a segmentation map (3D Mask)
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    lc = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesion found after crop.')
        return mask #, None, None
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # Total number of lesion voxels
    lesion_total = np.sum(mask)
    # Initiate cumulated lesion volume
    volume = 0
    
    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # Assure that we keep lesions bigger than threshold
        # print(k, vox2mm3(lesions[k]))
        # if vox2mm3(lesions[k]) > min_lesion_size:
        if lesions[k] > min_lesion_size:
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            volume += lesions[k]
            # print('keep lesion', k, volume)
            c = 1
    
    # if at least one lesion was above threshold
    if N > 1 and c == 1:
        smallest_lesion_size = lesion_cluster[0].shape[0]
    
    # if there is no lesion above the threshold
    if N > 1 and c == 0:
        print('NEW MASK WITH NO LESIONS ABOVE THRESHOLD :', np.max(new_mask))
        return new_mask #, None, None

    if N == 1: # Only 1 lesion
        smallest_lesion_size = lesion_total

    print(lc, 'lesions kept over', N, 'in total (min lesion size =', smallest_lesion_size, 'voxels or', round(vox2mm3(smallest_lesion_size), 2), 'mm3)')
    print(f'{round(volume/lesion_total*100, 1)}% of the lesional volume remaining.')
    return new_mask #, smallest_lesion_size, round(vox2mm3(smallest_lesion_size), 2)



#######################
#      Inférence      #
#######################



def thresh_func(mask, thresh=0.5):
    mask[mask >= thresh] = 1
    mask[mask < thresh] = 0

    return mask


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = torch.sum(pred * target)
    pred_sum = torch.sum(pred * pred)
    target_sum = torch.sum(target * target)

    return 1 - ((2. * intersection + 1e-5) / (pred_sum + target_sum + 1e-5))