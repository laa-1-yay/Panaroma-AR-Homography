import numpy as np
import cv2
import matplotlib.pyplot as plt


def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]

    print('gaussian_pyramid shape: ',gaussian_pyramid.shape)
    # print('single gaussian_pyramid shape: ',gaussian_pyramid[:,:,0].shape)

    for i in range(1, gaussian_pyramid.shape[-1]):
        DoG_pyramid_single = gaussian_pyramid[:,:,i] - gaussian_pyramid[:,:,i-1]
        DoG_pyramid.append(DoG_pyramid_single)

    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    print('DoG_pyramid shape: ', DoG_pyramid.shape)

    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...

    principal_curvature = []

    for i in range(0, DoG_pyramid.shape[-1]):

        sobelx = cv2.Sobel(DoG_pyramid[:, :, i], -1, 1, 0, ksize=3)
        sobely = cv2.Sobel(DoG_pyramid[:, :, i], -1, 0, 1, ksize=3)

        Hxx = cv2.Sobel(sobelx, -1, 1, 0, ksize=3)
        Hxy = cv2.Sobel(sobely, -1, 1, 0, ksize=3)
        Hyx = cv2.Sobel(sobelx, -1, 0, 1, ksize=3)
        Hyy = cv2.Sobel(sobely, -1, 0, 1, ksize=3)

        H_trace = np.add(Hxx, Hyy)
        H_determinant = np.add( np.multiply(Hxx, Hyy) , np.multiply(Hxy, Hyx))

        R = np.divide(np.multiply(H_trace, H_trace) , H_determinant)
        principal_curvature.append(R)
        # principal_curvature.append(Hxy)


    principal_curvature = np.stack(principal_curvature, axis=-1)
    # print(principal_curvature)
    print('principal_curvature shape: ', principal_curvature.shape)
    # Compute principal curvature here
    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    ##############
    #  TO DO ...
    # Compute locsDoG here

    kernel = np.ones((3, 3), np.uint8)

    local_max_arr = []
    local_min_arr = []
    for i in range(0, DoG_pyramid.shape[-1]):
        img = DoG_pyramid[:,:,i]
        local_max = cv2.compare(img, cv2.dilate(img, kernel=kernel, iterations=1), cv2.CMP_EQ)
        local_min = cv2.compare(img, cv2.erode(img, kernel=kernel, iterations=1), cv2.CMP_EQ)
        # print('max: ', local_max)
        # print('min: ', local_min)
        local_max_arr.append(local_max)
        local_min_arr.append(local_min)


    local_max_arr = np.stack(local_max_arr, axis=-1)
    local_min_arr = np.stack(local_min_arr, axis=-1)

    num_channels  =  local_max_arr.shape[-1]
    for k in range(0, num_channels):
        for row in range(0, local_max_arr.shape[0]):
            for col in range(0, local_max_arr.shape[1]):
                if local_min_arr[row, col,k] == 255:
                    if DoG_pyramid[row, col, k] > th_contrast and principal_curvature[row, col, k] <th_r:
                        if k == 0:
                            above_channel_pixel = 100000
                            below_channel_pixel = DoG_pyramid[row, col, k + 1]
                        elif k == num_channels-1:
                            below_channel_pixel = 100000
                            above_channel_pixel = DoG_pyramid[row, col, k - 1]
                        else :
                            above_channel_pixel = DoG_pyramid[row, col, k - 1]
                            below_channel_pixel = DoG_pyramid[row, col, k + 1]
                        if np.argmin([DoG_pyramid[row, col, k], above_channel_pixel , below_channel_pixel]) == 0:
                            locsDoG.append([col,row,k])

                if local_max_arr[row, col,k] == 255:
                    if DoG_pyramid[row, col, k] > th_contrast and principal_curvature[row, col, k] <th_r:
                        if k == 0:
                            above_channel_pixel = -100000
                            below_channel_pixel = DoG_pyramid[row, col, k + 1]
                        elif k == num_channels-1:
                            below_channel_pixel = -100000
                            above_channel_pixel = DoG_pyramid[row, col, k - 1]
                        else :
                            above_channel_pixel = DoG_pyramid[row, col, k - 1]
                            below_channel_pixel = DoG_pyramid[row, col, k + 1]
                        if np.argmax([DoG_pyramid[row, col, k], above_channel_pixel , below_channel_pixel]) == 0:
                            locsDoG.append([col,row,k])

    locsDoG = np.asarray(locsDoG)
    # print('locsDoG: ', locsDoG)
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here

    gauss_pyramid = createGaussianPyramid(im)
    #   DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    #  compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #  get local extrema
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid


def plotDoGDetector(im, locsDoG):

    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locsDoG[:,0], locsDoG[:,1], 'r.')
    plt.show()


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    # im = cv2.imread('../data/chickenbroth_04.jpg')
    # im = cv2.imread('../data/prince_book.jpeg')
    # im = cv2.imread('../data/pf_scan_scaled.jpg')

    # im = cv2.imread('../data/incline_L.png')
    print(im.shape)

    # im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)

    # # test DoG pyramid
    # DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # displayPyramid(DoG_pyr)

    # # test compute principal curvature
    # pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)

    # # test get local extrema
    # th_contrast = 0.03
    # th_r = 12
    # locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    print('locsDoG shape:  ' , locsDoG.shape)

    plotDoGDetector(im, locsDoG)


