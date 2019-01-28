import numpy as np
from planarH import computeH
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def compute_extrinsics(K, H):
    homography_intrinsic = np.matmul(np.linalg.inv(K), H)
    print('homography_intrinsic: ', homography_intrinsic)

    U, L, VT = np.linalg.svd(homography_intrinsic[:, 0:2])

    factor = np.asarray([[1, 0],
                         [0, 1],
                         [0, 0]])

    rotation_matrix_2_cols = np.matmul(np.matmul(U, factor), VT)
    print('rotation_matrix_2_cols: ', rotation_matrix_2_cols)

    rotation_matrix_col_3 = np.cross(rotation_matrix_2_cols[:, 0], rotation_matrix_2_cols[:, 1])
    print('rotation_matrix_col_3: ', rotation_matrix_col_3)

    R = np.zeros((3, 3))
    R[:, 0:2] = rotation_matrix_2_cols
    R[:, 2] = rotation_matrix_col_3
    print('R: ', R)

    print('Determinant of R: ', np.linalg.det(R))

    lambda_dash_list = []

    for m in range(0, 3):
        for n in range(0, 2):
            lambda_dash_list.append(homography_intrinsic[m, n] / R[m, n])

    lambda_dash = np.average(lambda_dash_list)
    print('lambda_dash: ', lambda_dash)

    T = homography_intrinsic[:, 2] / lambda_dash
    print('T: ', T)

    return R, T


def project_extrinsics(K, W, R, t):
    K_big = np.zeros((3, 4))
    K_big[0:3, 0:3] = K
    print('K_big: ', K_big)

    Homo_transform = np.zeros((4, 4))
    Homo_transform[0:3, 0:3] = R
    Homo_transform[0:3, 3] = t
    Homo_transform[3, 3] = 1
    print('Homo_transform: ', Homo_transform)

    X_list = []
    Y_list = []

    for i in range(0, W.shape[1]):
        W_vector = np.zeros((4, 1))
        W_vector[0:3, 0] = W[:, i]
        W_vector[3, 0] = 1
        # print('W_vector: ', W_vector )

        X_vector = np.matmul(np.matmul(K_big, Homo_transform), W_vector)
        # print('X_vector prev: ', X_vector )

        X_vector = X_vector / np.max(X_vector)
        # print('X_vector: ', X_vector )

        X_list.append(X_vector[0, 0])
        Y_list.append(X_vector[1, 0])

    # X_offset = 310
    # Y_offset = 635
    #
    # X_list = [x + X_offset for x in X_list]
    # Y_list = [y + Y_offset for y in Y_list]

    book_img = mpimg.imread('../data/prince_book.jpeg')
    plt.imshow(book_img)
    plt.scatter(X_list, Y_list, c='y', s=1)
    plt.show()


if __name__ == '__main__':
    W = np.asarray([[0.0, 18.2, 18.2, 0.0],
                    [0.0, 0.0, 26.0, 26.0],
                    [0.0, 0.0, 0.0, 0.0]
                    ])

    X = np.asarray([[483, 1704, 2175, 67],
                    [810, 781, 2217, 2286]
                    ])

    K = np.asarray([[3043.72, 0.0, 1196.0],
                    [0.0, 3043.72, 1604.0],
                    [0.0, 0.0, 1.0]
                    ])

    homography_approx = computeH(X, W[0:2, :])
    print('homography_approx: ', homography_approx)

    R, t = compute_extrinsics(K, homography_approx)

    f = open('../data/sphere.txt', 'r')
    points_sphere = f.readlines()
    f.close()

    # print(len(points_sphere))
    print('Raw points sphere: ', points_sphere)

    X_arr = str.split(points_sphere[0])
    X_arr = [float(i) for i in X_arr]
    print('X_arr: ', X_arr)

    Y_arr = str.split(points_sphere[1])
    Y_arr = [float(i) for i in Y_arr]
    print('Y_arr: ', Y_arr)

    Z_arr = str.split(points_sphere[2])
    Z_arr = [float(i) for i in Z_arr]
    print('Z_arr: ', Z_arr)

    sphere_3D_points = []
    sphere_3D_points.append(X_arr)
    sphere_3D_points.append(Y_arr)
    sphere_3D_points.append(Z_arr)

    sphere_3D_points = np.asarray(sphere_3D_points)
    print('3d points: ', sphere_3D_points)
    print('3d points shape: ', sphere_3D_points.shape)

    center = np.array([813, 1490, 1])
    shift = np.linalg.inv(homography_approx).dot(center)

    print(shift)
    shift = shift / shift[-1]
    print(shift)


    sphere_3D_points[:3, :] = (sphere_3D_points[:3, :].T + shift).T

    X = project_extrinsics(K, sphere_3D_points, R, t)
