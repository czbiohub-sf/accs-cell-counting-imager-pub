

from skimage import io, exposure, filters, transform, morphology, measure, feature
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift

import os
import matplotlib.pyplot as plt
import numpy as np



def main(bgndFilenames, imgFilenames, imagePath):

	for i in range(len(bgndFilenames)):

		img = io.imread(imgFilenames[i]).astype(float)
		bg = io.imread(bgndFilenames[i]).astype(float)

		imgSize = img.shape
		noiseImg = np.random.rand(imgSize[0], imgSize[1])*100
		img, bg, positions = countCells(img,bg)

		# io.imsave(os.path.join(imagePath, 'subtractedRaw_' + str(i) + '.tif'), subtractedRaw)
		img = convertToUint16(img, True)
		io.imsave(os.path.join(imagePath, 'subtractedAligned_' + str(i) + '.tif'), img)


def convertToUint16(img, useHistogram = True):

	if useHistogram:
		mi, ma = np.percentile(img, (0.1, 99.9))
	else:
		mi = img.min()
		ma = img.max()

	img = 65535*(img.astype(float)-mi)/(ma-mi)
	img[img < 0] = 0
	img[img > 65535] = 65535
	img = np.uint16(img)

	return img

def imAlign(imgToAlign, template):
	shift, error, diffphase = register_translation(template.astype(float), imgToAlign.astype(float))
	alignedImage = fourier_shift(np.fft.fftn(imgToAlign), shift)
	alignedImage = np.fft.ifftn(alignedImage)
	return alignedImage

def filterImage(img, lowpass_pixels, highpass_pixels):
	# Filter the images by subtracting a low-passed version (similar to doing rolling ball), then doing a mild low-pass filter
	return filters.gaussian(img - filters.gaussian(img, sigma = highpass_pixels), sigma=lowpass_pixels)

def countCells(img, bg, alignImages = True, subtractBG = True):

	# Downsample the image for speed        
	ds = 2
	img = transform.downscale_local_mean(img, (ds,ds))
	bg = transform.downscale_local_mean(bg, (ds,ds))

	# Ensure floating point
	img = img.astype(float)
	bg = bg.astype(float)

	if alignImages:
	    img = imAlign(img, bg)
	    img = img.real

	if subtractBG:
	    img = img - bg

	#Crop to remove out of focus edges 
	ic = 250
	img = img[ic:-ic, ic:-ic]

	# Filtering the image removes unwanted low- and high- frequency noise
	lp = 2
	hp = 10
	img = filterImage(img, lp, hp)

	# 
	threshold = filters.threshold_otsu(img)
	# threshold = np.percentile(img, 98)

	# generate the background mask
	mask = img > threshold

	# eliminate single-pixel regions in the mask
	mask = morphology.erosion(mask)

	# remove regions that are too large to be cells or clumps of cells
	max_region_area = 200
	min_region_area = 10

	# The mask_label assigns a different pixel value to each unique connected region
	mask_label = measure.label(mask)
	props = measure.regionprops(mask_label)

	# Make a copy of the image, which will get its background masked out (set to zero)
	img_masked = img.copy()

	# Black out the regions that do not fit our size criteria
	for prop in props:
		if prop.area > max_region_area or prop.area < min_region_area:
			img_masked[mask_label==prop.label] = 0

	# # find the cells by identifying all of the local maxima in the masked, filtered, and background-subtracted image
	min_cell_spacing = 5
	positions = feature.peak_local_max(img_masked, indices=True, min_distance=min_cell_spacing, threshold_abs = threshold)

	print("Found " + str(len(positions[:,1])) + " cells")

	# Show the images
	fig = plt.figure(figsize=(8, 3))
	ax = plt.subplot(1,1,1)
	ax.imshow(img, cmap='gray')
	ax.set_axis_off()
	ax.scatter(positions[:,1], positions[:,0], color='red', s=1)
	plt.show()
	return img, bg, positions



if __name__ == "__main__":

	# Not confluent
	imagePath = "C:\\Users\\paul.lebel.CZBIOHUB\\Documents\\ImageShiftTesting\\20200116-134956"	
	bgnd_fns = ["background-20200116-135005-0.png", "background-20200116-135013-1.png",  "background-20200116-135021-2.png"]
	img_fns = ["image-20200116-135004-0.png","image-20200116-135012-1.png", "image-20200116-135019-2.png"]

	# Confluent:
	# imagePath = "C:\\Users\\paul.lebel.CZBIOHUB\\Documents\\ImageShiftTesting"
	# bgnd_fns = ["background-20191217-185540-0.png", "background-20191217-183624-2.png", "background-20191217-184237-3.png", "background-20191217-184929-4.png"]
	# img_fns = ["image-20191217-185538-0.png","image-20191217-183622-2.png", "image-20191217-184236-3.png", "image-20191217-184928-4.png"]

	bgndFilenames = []
	imgFilenames = []

	for i in range(len(bgnd_fns)):
		bgndFilenames.append(os.path.join(imagePath, bgnd_fns[i]))
		imgFilenames.append(os.path.join(imagePath, img_fns[i]))
	
	main(bgndFilenames, imgFilenames, imagePath)