import numpy as np
import cv2
from BRIEF import briefLite, briefMatch, plotMatches
import matplotlib.pyplot as plt

def rotateImage(image, angle):
    row,col = (image.shape[0], image.shape[1])
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def plot_bar_chart(matches_num, label):
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, matches_num)
    plt.xlabel('Angle')
    plt.ylabel('No. of Matches')
    plt.xticks(index, label, fontsize=7, rotation=30)
    plt.title('Number of matches at each angle')
    plt.show()

if __name__ == '__main__':

    print('-------------------------   IMAGE 1  -----------------------------')
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    locs1, desc1 = briefLite(im1)

    print('-------------------------   IMAGE 2  -----------------------------')
    im2 = cv2.imread('../data/model_chickenbroth.jpg')

    angles = []
    matches_num = []

    for angle in range(0,361,10):
        print('angle', angle)
        im_rotated = rotateImage(im2, angle)
        locs2, desc2 = briefLite(im_rotated)

        matches = briefMatch(desc1, desc2)
        print('matches shape: ', matches.shape)
        # plotMatches(im1, im_rotated, matches, locs1, locs2)
        matches_num.append(matches.shape[0])
        angles.append(angle)

    plot_bar_chart(matches_num, angles)