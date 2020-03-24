import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    # #TODO 8
    # #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")
    # #TODO-BLOCK-END

    # minX = -np.inf
    # minY = -np.inf
    # maxX = np.inf
    # maxY = np.inf

    newVals = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            newx,newy,newz = M@(np.asarray([x,y,1]))
            newVals.append([newx,newy,newz]/newz)

    minX = (newVals[np.argmin(newVals,axis = 0)[0]])[0]
    minY = (newVals[np.argmin(newVals,axis = 0)[1]])[1]

    maxX = (newVals[np.argmax(newVals,axis = 0)[0]])[0]
    maxY = (newVals[np.argmax(newVals,axis = 0)[1]])[1]


    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # # BEGIN TODO 10
    # # Fill in this routine
    # #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")
    # #TODO-BLOCK-END
    # # END TODO

    minX, minY, maxX, maxY = imageBoundingBox(img, M)
    # print("max" + str(maxX))
    # print(maxY)

    warpedImg = cv2.warpPerspective(img,M,(acc.shape[1],acc.shape[0]), flags=1)
    print(warpedImg.shape)
    # rgba = np.concatenate((warpedImg, np.zeros((warpedImg.shape[0], warpedImg.shape[1], 1))), axis=2)
    # rgba = cv2.cvtColor(warpedImg, cv2.COLOR_RGB2RGBA)

    blendMin = blendWidth+minX
    blendMax = maxX-blendWidth
    for i in  range(minY,maxY):
        for j in  range(minX,maxX):
            if j < blendMin:
                for k in range(3):
                    acc[i][j][k] += ((j-minX)/blendWidth)*warpedImg[i][j][k]
                acc[i][j][3] += (j-minX)/blendWidth
            elif j > blendMax:
                # rgba[i][j][3] +=  maxX-j/blendWidth
                for k in range(3):
                    acc[i][j][k] += ((maxX-j)/blendWidth)*warpedImg[i][j][k]
                acc[i][j][3] += (maxX-j)/blendWidth
            else:
                for k in range(3):
                    acc[i][j][k] += warpedImg[i][j][k]
                acc[i][j][3] += 1









def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    # #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")
    # #TODO-BLOCK-END
    # # END TODO
    dat_shape0, dat_shape1, dat_shape2 = acc.shape[0],acc.shape[1],acc.shape[2]-1
    img = np.zeros((dat_shape0,dat_shape1,dat_shape2), dtype=np.uint8)
    for i in range(dat_shape0):
        for j in range(dat_shape1):
            for k in range(dat_shape2):
                img[i][j][k] = int(acc[i][j][k]/acc[i][j][3]) if acc[i][j][3] != 0 else 0
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)

    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # # BEGIN TODO 9
        # # add some code here to update minX, ..., maxY
        # #TODO-BLOCK-BEGIN
        # raise Exception("TODO in blend.py not implemented")
        # #TODO-BLOCK-END
        # # END TODO
        values = imageBoundingBox(img, M)
        if values[1]<minY:
            minY = values[1]
        if values[0]<minX:
            minX = values[0]
        if values[3]>maxY:
            maxY = values[3]
        if values[2]>maxX:
            maxX = values[2]


    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    # #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")
    # #TODO-BLOCK-END
    # END TODO

    if is360:
        A = computeDrift(x_init,y_init,x_final,y_final,width)

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

