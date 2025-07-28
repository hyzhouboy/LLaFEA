"""
@Time ： 2023/2/12 23:31
@Auth ： Haoyue Liu
@File ：rectify_sift.py
@reference ：https://blog.csdn.net/qq_31318135/article/details/105924610
"""
import os
from tqdm import tqdm
import numpy as np
import cv2


def get_npy(data_path):
    """
    find npy files in data path
    :return: list of files found
    """
    files = []
    exts = ['npy']
    isroot = True
    for parent, dirnames, filenames in os.walk(data_path):
        if isroot:
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        isroot = False
    files.sort()
    return files


def get_imgs(data_path):
    """
    find npy files in data path
    :return: list of files found
    """
    files = []
    exts = ['jpg', 'png', 'bmp']
    isroot = True
    for parent, dirnames, filenames in os.walk(data_path):
        if isroot:
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        isroot = False
    files.sort()
    return files


def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, -1, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, -1, 0, 1, ksize=3)
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def findHomography(img1, img2, file, im_type, result_path, failed_list):
    # define constants
    min_match_count = 4
    min_dist_threshold = 0.7
    ransac_reproj_threshold = 5.0

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # find matches
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < min_dist_threshold * n.distance:
            good.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homo, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)

        # 画出匹配的关键单
        matches_mask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)
        img_matched = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        # cv2.namedWindow('match', 0)
        # cv2.resizeWindow('match', 1600, 900)
        # cv2.imshow('match', img_matched)
        # cv2.waitKey(0)

        match_name = file.replace('.npy', '.jpg')
        cv2.imwrite(os.path.join(result_path, match_name), img_matched)

        # 当没有找到单应性矩阵时返回None，类型为<class 'NoneType'>
        if type(homo) != np.ndarray:
            fail_msg = im_type + ': ' + file + '\n'
            print(fail_msg)
            failed_list.append(fail_msg)
            return np.ones((3, 3)), failed_list
        return homo, failed_list
    else:
        # raise Exception("Not enough matches are found - {}/{}".format(len(good), min_match_count))
        fail_msg = im_type + ': ' + file + '\n'
        print(fail_msg)
        failed_list.append(fail_msg)
        return np.ones((3, 3)), failed_list


if __name__ == '__main__':
    # event to image
    base_dir = 'E:/4000datasets/self-record/20230207_rgb-evs_street/1700PM/car4'
    # src_path = os.path.join(base_dir, 'reconstruction_rect')
    src_path = os.path.join(base_dir, 'reconstruction_rect')
    dst_path = os.path.join(base_dir, 'image')
    homography_path = os.path.join(base_dir, 'homography')
    homography_gray_path = os.path.join(homography_path, 'gray')
    homography_gray_show_path = os.path.join(homography_path, 'gray_show')
    homography_gradient_path = os.path.join(homography_path, 'gradient')
    homography_gradient_show_path = os.path.join(homography_path, 'gradient_show')
    if not os.path.exists(homography_gray_path):
        os.makedirs(homography_gray_path)
    if not os.path.exists(homography_gray_show_path):
        os.makedirs(homography_gray_show_path)
    if not os.path.exists(homography_gradient_path):
        os.makedirs(homography_gradient_path)
    if not os.path.exists(homography_gradient_show_path):
        os.makedirs(homography_gradient_show_path)

    src_im_list = get_imgs(src_path)
    dst_im_list = get_imgs(dst_path)

    final_homography_gray = np.ones((3, 3))
    final_homography_gradient = np.ones((3, 3))
    # 记录单应性估计失败的文件
    fail_list = []
    for i, src_im in enumerate(tqdm(src_im_list)):
        im1 = cv2.imread(src_im)
        im2 = cv2.imread(dst_im_list[i])
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        im1_gradient = get_gradient(im1_gray)
        im2_gradient = get_gradient(im2_gray)

        # 通过sift匹配计算单应性矩阵
        index = os.path.basename(src_im).replace('.png', '.npy')
        homography_gray, fail_list = findHomography(im1_gray, im2_gray, index, 'gray', homography_gray_show_path, fail_list)
        if not (homography_gray == np.ones(3)).all():
            final_homography_gray = homography_gray
        np.save(os.path.join(homography_gray_path, index), final_homography_gray)

        homography_gradient, fail_list = findHomography(im1_gradient, im2_gradient, index, 'gradient', homography_gradient_show_path, fail_list)
        if not (homography_gradient == np.ones((3, 3))).all():
            final_homography_gradient = homography_gradient
        np.save(os.path.join(homography_gradient_path, index), final_homography_gradient)

        # 对图像进行变换
        # im2_aligned = cv2.warpPerspective(im2, homography_gradient, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # cv2.imshow("im1_gray", im1_gray)
        # cv2.imshow("im2_gray", im2_gray)
        # cv2.imshow("Aligned Image 2", im2_aligned)
        # cv2.waitKey(0)

    failed_txt = os.path.join(homography_path, 'failed.txt')
    with open(failed_txt, mode='w', encoding='utf-8') as f:
        f.writelines(fail_list)
