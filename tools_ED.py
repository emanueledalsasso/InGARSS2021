import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(ima,low=None,high=None):
  if low is None: low = ima.min()
  if high is None: high = ima.max()
  plt.figure()
  plt.hist(ima.ravel(),bins='auto',density=True,range=[low,high])
  plt.show()

def plot_ima(ima,size=8,ccmap='gray'):
	plt.figure(figsize=(size,size))
	plt.imshow(ima,cmap=ccmap)
	plt.show()

def plot_optical_rgb(imagedata,threshold=None,size=8):
	plt.figure(figsize=(size,size))
	ima = imagedata[:,:,:3]
	if threshold is None: threshold = np.mean(ima) + 3 * np.std(ima)
	plt.imshow(np.clip(ima,0,threshold)/threshold)
	plt.show()

def plotsar(ima,threshold=None,size=8):
	plt.figure(figsize=(size,size))
	if threshold is None: threshold = np.mean(ima) + 3 * np.std(ima)
	plt.imshow(np.clip(ima,0,threshold)/threshold*255)
	plt.show()

def plot_lely(image):
    threshold = 235.90
    image = np.clip(image,0,threshold)
    image = image/threshold*255
    plt.figure()
    plt.imshow(image,cmap='gray')
    plt.show()

def plot_lely_multichannel(im1,im2,im3):
    seuil = 235.90
    im1 = np.clip(im1,0,seuil); im1 = im1/seuil
    im2 = np.clip(im2,0,seuil); im2 = im2/seuil
    im3 = np.clip(im3,0,seuil); im3 = im3/seuil
    image = np.stack((im1,im2,im3),axis=2)
    plt.figure()
    plt.imshow(image)
    plt.show()

def plot_image_rgb(image=None, mask=None, ax=None, factor=3.5/255, clip_range=(0, 1), **kwargs):
    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

    mask_color = [255, 255, 255, 255] if image is None else [255, 255, 0, 100]

    if image is None:
        if mask is None:
            raise ValueError('image or mask should be given')
        image = np.zeros(mask.shape + (3,), dtype=np.uint8)

    ax.imshow(np.clip(image * factor, *clip_range), **kwargs)

    if mask is not None:
        cloud_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

        cloud_image[mask == 1] = np.asarray(mask_color, dtype=np.uint8)

        ax.imshow(cloud_image)

def apply_thresholding(change_map,threshold):
  change_map_ = np.copy(change_map)
  change_map_[change_map>threshold]=255.0
  change_map_[change_map<=threshold]=0.0
  plot_ima(change_map_)

#Function allowing to compute NDVI for data downloaded from the API Sentinel Hub
def fndvi(bands):
  L_band_04 = []
  L_band_08 = []
  for i in range(len(bands)):
    L_04 = []
    L_08 = []
    for j in range(len(bands[0])):
      L_04.append(bands[i][j][2])
      L_08.append(bands[i][j][4])
    L_band_04.append(L_04)
    L_band_08.append(L_08)
  band_04 = np.array(L_band_04)
  band_08 = np.array(L_band_08)
  np.seterr(divide='ignore', invalid='ignore')
  ndvi = (band_08.astype(float) - band_04.astype(float)) / (band_04 + band_08)
  return (ndvi)

def diff_ndvi(ndvi_before, ndvi_after):
  ndvi_modif = np.empty((len(ndvi_before),len(ndvi_before[0]),3))
  for i in range (len(ndvi_before)):
    for j in range (len(ndvi_before[0])):
      if (ndvi_before[i,j]==0 or ndvi_after[i,j]==0): # Une des cases avec nuages -> Rouge
        ndvi_modif [i,j] = np.array([255, 0, 0])
      elif (ndvi_before[i,j]==ndvi_after[i,j]): # Pas de changement -> Vert
        ndvi_modif [i,j] = np.array([0,255,0])
      else: # Changement -> Bleu
        ndvi_modif [i,j] = np.array([0,0,255])
  return ndvi_modif
