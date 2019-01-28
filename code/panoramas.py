import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches,makeTestPattern
from PIL import Image
import math


def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    print('matches shape: ', matches.shape)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix without cliping.
    '''
    ######################################
    # TO DO ...

    warp_output_shape = (im1.shape[1] *2,  im1.shape[0]*2)

    # ------------------------   WARPING IMAGE 1 AND 2 ---------------------------------

    im1_warped = cv2.warpPerspective(im1,  np.eye(3,3), warp_output_shape)
    im2_warped = cv2.warpPerspective(im2,  H2to1, warp_output_shape)

    im1_warped = np.asarray(im1_warped)
    im2_warped = np.asarray(im2_warped)
    print('im1 shape :', im1.shape)
    print('im2_warped shape :', im2_warped.shape)

    # ------------------------  CREATING A MASK FOR BLENDING  ---------------------------------

    imgsize = (im2.shape[1], im2.shape[0])  # The size of the image
    mask = Image.new('L', imgsize)  # Create the image
    innerColor = 225  # Color at the center
    outerColor = 0  # Color at the corners

    for y in range(imgsize[1]):
        for x in range(imgsize[0]):
            # Find the distance to the center
            distanceToCenter = math.sqrt((x - imgsize[0] / 2) ** 2 + (y - imgsize[1] / 2) ** 2)
            # Make it on a scale from 0 to 1
            distanceToCenter = float(distanceToCenter) / (math.sqrt(2) * imgsize[0] / 2) + 0.6
            color = outerColor * (distanceToCenter) + innerColor * (1 - distanceToCenter)
            # Place the pixel
            mask.putpixel((x, y), int(color))

    mask = np.asarray(mask)
    mask = mask / 255
    mask_warped = cv2.warpPerspective(mask, H2to1, warp_output_shape)
    masked_warped_channels = np.dstack((mask_warped, np.dstack((mask_warped, mask_warped))))

    # ------------------------  FINDING OVERLAPPING PIXELS  ---------------------------------

    image1_0_pixels = np.where(im1_warped == 0)
    overlapping_pixel_image = im2_warped.copy()
    overlapping_pixel_image[image1_0_pixels] = 0
    overlapping_pixels = np.where(overlapping_pixel_image != 0)

    # ------------------------  MERGING AND BLENDING IMAGES WITH MASK  ---------------------------------

    pano_im = im1_warped + im2_warped

    mask_value = masked_warped_channels[overlapping_pixels]
    pano_im[overlapping_pixels] = im1_warped[overlapping_pixels] * (1 - mask_value) + im2_warped[overlapping_pixels] * (
        mask_value)

    # ------------------------  OPTIMIZING BLENDING OF TWO IMAGES  ---------------------------------

    overlapping_pixels = np.asarray(overlapping_pixels)
    min_row = np.min(overlapping_pixels[0, :])
    max_column = np.max(overlapping_pixels[1, :])
    print('min_row: ', min_row)
    print('max_column: ', max_column)

    for i in range(0, overlapping_pixels.shape[1]):
        x = overlapping_pixels[0, i]
        y = overlapping_pixels[1, i]
        z = overlapping_pixels[2, i]

        if y > (max_column - 50):
            dist = max_column - y
            dist = dist / 50
            pano_im[x-1, y, z] = im2_warped[x - 1, y, z] * (1 - dist) + im1_warped[x, y, z] * dist

    # cv2.imshow("mask", masked_warped_channels)
    # cv2.imshow('im1_warped: ', im1_warped)
    # cv2.imshow('im2_warped: ', im2_warped)
    # cv2.imshow('pano_im: ', pano_im)
    # cv2.waitKey(0)

    return pano_im, im2_warped


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix without cliping.
    '''
    ######################################
    # TO DO ...

    # ------------------------   CALCULATING THE CORNERS ---------------------------------

    top_left_corner_im1 = np.asarray([0,0,1])
    top_right_corner_im1 = np.asarray([im1.shape[1], 0 ,1])
    bottom_left_corner_im1 = np.asarray([0, im1.shape[0], 1])
    bottom_right_corner_im1 = np.asarray([im1.shape[1], im1.shape[0] ,1])

    top_left_corner_im2 = np.asarray([0,0,1])
    top_right_corner_im2 = np.asarray([im2.shape[1],0,1])
    bottom_left_corner_im2 = np.asarray([0,im2.shape[0],1])
    bottom_right_corner_im2 = np.asarray([ im2.shape[1], im2.shape[0],1])

    top_left_corner_im2_warped = np.matmul( H2to1, top_left_corner_im2.T)
    top_left_corner_im2_warped = top_left_corner_im2_warped / top_left_corner_im2_warped[2]

    top_right_corner_im2_warped = np.matmul( H2to1, top_right_corner_im2.T)
    top_right_corner_im2_warped = top_right_corner_im2_warped / top_right_corner_im2_warped[2]

    bottom_left_corner_im2_warped = np.matmul( H2to1, bottom_left_corner_im2.T)
    bottom_left_corner_im2_warped = bottom_left_corner_im2_warped / bottom_left_corner_im2_warped[2]

    bottom_right_corner_im2_warped = np.matmul( H2to1, bottom_right_corner_im2.T)
    bottom_right_corner_im2_warped = bottom_right_corner_im2_warped / bottom_right_corner_im2_warped[2]

    print('top_left_corner_im1: ', top_left_corner_im1[0:2])
    print('top_right_corner_im1: ', top_right_corner_im1[0:2])
    print('bottom_left_corner_im1: ', bottom_left_corner_im1[0:2])
    print('bottom_right_corner_im1: ', bottom_right_corner_im1[0:2])

    print('top_left_corner_im2 before warp: ', top_left_corner_im2[0:2])
    print('top_right_corner_im2 before warp: ', top_right_corner_im2[0:2])
    print('bottom_left_corner_im2 before warp: ', bottom_left_corner_im2[0:2])
    print('bottom_right_corner_im2 before warp: ', bottom_right_corner_im2[0:2])

    print('top_left_corner_im2_warped: ', top_left_corner_im2_warped[0:2])
    print('top_right_corner_im2_warped: ', top_right_corner_im2_warped[0:2])
    print('bottom_left_corner_im2_warped: ', bottom_left_corner_im2_warped[0:2])
    print('bottom_right_corner_im2_warped: ', bottom_right_corner_im2_warped[0:2])

    corners_combined = []
    corners_combined.append(top_left_corner_im1[0:2])
    corners_combined.append(top_right_corner_im1[0:2])
    corners_combined.append(bottom_left_corner_im1[0:2])
    corners_combined.append(bottom_right_corner_im1[0:2])
    corners_combined.append(top_left_corner_im2_warped[0:2])
    corners_combined.append(top_right_corner_im2_warped[0:2])
    corners_combined.append(bottom_left_corner_im2_warped[0:2])
    corners_combined.append(bottom_right_corner_im2_warped[0:2])

    corners_combined = np.asarray(corners_combined)
    print('corners_combined: ', corners_combined)

    combined_top_left_corner_x = min( corners_combined[:,0] )
    combined_top_left_corner_y = min( corners_combined[:,1] )
    combined_bottom_right_corner_x = max( corners_combined[:,0] )
    combined_bottom_right_corner_y = max( corners_combined[:,1] )

    print('combined_top_left_corner_x: ', combined_top_left_corner_x)
    print('combined_top_left_corner_y: ', combined_top_left_corner_y)
    print('combined_bottom_right_corner_x: ', combined_bottom_right_corner_x)
    print('combined_bottom_right_corner_y: ', combined_bottom_right_corner_y)

    combined_height = max( corners_combined[:,1] ) - min( corners_combined[:,1])
    combined_width =  max( corners_combined[:,0] ) - min( corners_combined[:,0])

    print('combined_width: ', combined_width)
    print('combined_height: ', combined_height)

    translation_y = abs(min( corners_combined[:,1] ))
    print('translation_y: ', translation_y)

    aspect_ratio = combined_width/ combined_height
    print('aspect_ratio: ', aspect_ratio)

    output_width = 1000
    output_height = output_width / aspect_ratio
    print('output_width: ', output_width)
    print('output_height: ', output_height)

    M1  = np.zeros( (3,3) )
    M1[0, 2] = 0
    M1[1, 2] = 0
    M1[0, 0] =  output_width/combined_width
    M1[1, 1] =  output_width/combined_width
    M1[2, 2] = 1

    M2  = np.zeros( (3,3) )
    M2[0, 2] = 0
    M2[1, 2] = translation_y
    M2[0, 0] = 1
    M2[1, 1] = 1
    M2[2, 2] = 1

    M = np.matmul(M1, M2)
    print('M: ', M)

    warp_output_shape = (int(output_width),  int(output_height))

    # ------------------------   WARPING IMAGE 1 AND 2 ---------------------------------

    im1_warped = cv2.warpPerspective(im1, M, warp_output_shape)
    im2_warped = cv2.warpPerspective(im2, np.matmul(M, H2to1), warp_output_shape)

    im1_warped = np.asarray(im1_warped)
    im2_warped = np.asarray(im2_warped)
    print('im1_warped shape :', im1_warped.shape)
    print('im2_warped shape :', im2_warped.shape)

    # ------------------------  CREATING A MASK FOR BLENDING  ---------------------------------

    imgsize = (im2.shape[1], im2.shape[0])  # The size of the image
    mask = Image.new('L', imgsize )  # Create the image
    innerColor = 225  # Color at the center
    outerColor = 0  # Color at the corners

    for y in range(imgsize[1]):
        for x in range(imgsize[0]):
            # Find the distance to the center
            distanceToCenter = math.sqrt((x - imgsize[0] / 2) ** 2 + (y - imgsize[1] / 2) ** 2 )
            # Make it on a scale from 0 to 1
            distanceToCenter = float(distanceToCenter) / (math.sqrt(2) * imgsize[0] / 2) +0.6
            color = outerColor * (distanceToCenter) + innerColor* (1 - distanceToCenter)
            # Place the pixel
            mask.putpixel((x, y), int(color))

    mask = np.asarray(mask)
    mask = mask/255
    mask_warped = cv2.warpPerspective(mask, np.matmul(M, H2to1), warp_output_shape)
    masked_warped_channels = np.dstack( ( mask_warped , np.dstack( (mask_warped, mask_warped)  )) )

    # ------------------------  FINDING OVERLAPPING PIXELS  ---------------------------------

    image1_0_pixels = np.where(im1_warped == 0)
    overlapping_pixel_image = im2_warped.copy()
    overlapping_pixel_image[image1_0_pixels] = 0
    overlapping_pixels  = np.where(overlapping_pixel_image!=0)

    # ------------------------  MERGING AND BLENDING IMAGES WITH MASK  ---------------------------------

    pano_im = im1_warped + im2_warped

    mask_value = masked_warped_channels[overlapping_pixels]
    pano_im[overlapping_pixels] = im1_warped[overlapping_pixels]*(1- mask_value)  + im2_warped[overlapping_pixels]*(mask_value)

    # ------------------------  OPTIMIZING BLENDING OF TWO IMAGES  ---------------------------------

    overlapping_pixels = np.asarray(overlapping_pixels)
    min_row = np.min(overlapping_pixels[0, :])
    max_column = np.max(overlapping_pixels[1, :])
    print('min_row: ', min_row)
    print('max_column: ', max_column)

    for i in range(0, overlapping_pixels.shape[1]):
        x = overlapping_pixels[0, i]
        y = overlapping_pixels[1, i]
        z = overlapping_pixels[2, i]

        if y > (max_column - 50):
            dist = max_column - y
            dist = dist/50
            pano_im[x, y, z] = im2_warped[x-1, y, z] * (1-dist) + im1_warped[x, y, z] * dist

        if x < (min_row + 10):
            dist = x - min_row
            dist = dist / 10
            pano_im[x, y, z] = im2_warped[x, y, z] * (1 - dist) + im1_warped[x, y-1, z] * dist


    # cv2.imshow("mask", masked_warped_channels)
    # cv2.imshow('im1_warped: ', im1_warped)
    # cv2.imshow('im2_warped: ', im2_warped)
    # cv2.imshow('pano_im: ', pano_im)
    # cv2.waitKey(0)

    return pano_im



if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    # print('im1 shape: ', im1.shape)
    # print('im2 shape: ', im2.shape)
    #
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # # plotMatches(im1,im2,matches,locs1,locs2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # print(H2to1)
    # np.save('../results/q6_1.npy', H2to1)

    H2to1 = np.load('../results/q6_1.npy')

    # pano_im, warped_image = imageStitching(im1, im2, H2to1)
    # cv2.imwrite('../results/6_1.jpg', warped_image)
    # cv2.imshow('panoramas', pano_im)

    # pano_im_noClip = imageStitching_noClip(im1, im2, H2to1)
    # cv2.imwrite('../results/q6_2_pan.jpg', pano_im_noClip)
    # cv2.imshow('panorama no clip: ', pano_im_noClip)

    im3 = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_3.jpg', im3)
    cv2.imshow('im3: ', im3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()