import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...

    A = []

    for i in range(0, p1.shape[1]):
        u = p2[0, i]
        v = p2[1, i]
        x = p1[0, i]
        y = p1[1, i]
        A.append( [0,0,0,-u,-v,-1, y*u, y*v, y] )
        A.append( [u,v,1,0,0,0,-x*u,-x*v,-x] )


    A = np.asarray(A)

    # print('A shape: ', A.shape) # 8x9

    U, E, V_transpose = np.linalg.svd(A, full_matrices=True)

    # print('V_transpose shape: ', V_transpose.shape) # 8x9
    V_original = V_transpose.T

    # print('V_original shape: ', V_original.shape) # 9x8

    V_last_column = V_original[:, V_original.shape[1]-1]
    H2to1 = np.reshape(V_last_column, (3,3))

    # print(H2to1)

    # TEST OR A*H == 0
    # test = np.matmul(A, H2to1.reshape((9,1)))
    # print(test)

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...

    bestH = None
    inliers_max = -np.inf

    for i in range(0, num_iter):
        num_matches = len(matches)

        points_1 = []
        points_2 = []
        for j in range(0,4):
            random_match_index = np.random.randint(num_matches)
            points_1.append(locs1[matches[random_match_index,0], 0:2])
            points_2.append(locs2[matches[random_match_index,1], 0:2])

        points_1 = np.asarray(points_1)
        points_2 = np.asarray(points_2)

        # print('4 points in image1: ', points_1)
        # print('4 points in image2: ', points_2)

        points_1 = points_1.T
        points_2 = points_2.T

        # print('reshaped 4 points in image1: ', points_1)
        # print('reshaped 4 points in image2: ', points_2)

        H2to1 = computeH(points_1, points_2)
        # print(H2to1)

        inliers = 0
        for j in range(0, matches.shape[0]):
            x = locs1[ matches[j,0], 0]
            y = locs1[ matches[j,0], 1]
            u = locs2[ matches[j,1], 0]
            v = locs2[ matches[j,1], 1]

            left_side = np.matmul(H2to1, np.asarray([u,v,1]).T)
            # print('left_side: ', left_side)
            # print('left_side shape: ', left_side.shape)

            # IN ORDER TO GET 3RD ELEMENT OF left_side AS 1 we divide each element of left_side by 3rd term
            left_side = left_side/ left_side[2]

            # print('left_side_new: ', left_side)

            u_transformed = left_side[0]
            v_transformed = left_side[1]

            # print('x: '+ str(x)+ ', y: ' + str(y) + ', u: '+ str(u)+ ', v: ' + str(v) + ', u transformed: '+ str(u_transformed)+ ', v transformed: ' + str(v_transformed))

            eucl_dist = np.sqrt( np.square(x - u_transformed) + np.square(y - v_transformed)  )
            # print('eucl_dist', eucl_dist)
            if eucl_dist<tol:
                inliers += 1

        print('iteration: ',i,  ', num of inliers: ' , inliers)
        if inliers>inliers_max:
            inliers_max = inliers
            bestH = H2to1

    print('inliers_max: ', inliers_max)
    print('bestH: ', bestH)
    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)

    print('matches shape: ', matches.shape)

    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

