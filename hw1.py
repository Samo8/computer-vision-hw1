"""
ITS8030: Homework 1

Please implement all functions below.

For submission create a project called its8030-2021-hw1 and put the solution in there.

Please note that NumPy arrays and PyTorch tensors share memory represantation, so when converting a
torch.Tensor type to numpy.ndarray, the underlying memory representation is not changed.

There is currently no existing way to support both at the same time. There is an open issue on
PyTorch project on the matter: https://github.com/pytorch/pytorch/issues/22402

There is also a deeper problem in Python with types. The type system is patchy and generics
has not been solved properly. The efforts to support some kind of generics for Numpy are
reflected here: https://github.com/numpy/numpy/issues/7370 and here: https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY
but there is currently no working solution. For Dicts and Lists there is support for generics in the
typing module, but not for NumPy arrays.
"""
import cv2
import numpy as np
from numpy.core.fromnumeric import shape
import util
import matplotlib.pyplot as plt
import math


"""
Task 1: Convolution

Implement the function

convolution(image : np.ndarray, kernel : np.ndarray, kernel_width : int, kernel_height : int, add : bool, in_place:bool) -> np.ndarray

to convolve an image with a kernel of size kernel_height*kernel_width.
Use zero-padding around the borders for simplicity (what other options would there be?).
Here:

    image is a 2D matrix of class double
    kernel is a 2D matrix with dimensions kernel_width and kernel_height
    kernel_width and kernel_height are the width and height of the kernel respectively

(Note: in the general case, they are not equal and may not be always odd, so you have to ensure that they are odd.)

    if add is true, then 128 is added to each pixel for the result to get rid of negatives.
    if in_place is False, then the output image should be a copy of the input image. The default is False,
    i.e. the operations are performed on the input image.

Write a general convolution function that can handle all possible cases as mentioned above.
You can get help from the convolution part of the function mean_blur_image (to be implemented in a lab)
to write this function.
"""

image_festival = cv2.imread("songfestival.jpg", cv2.IMREAD_COLOR)
image_festival_rgb = cv2.cvtColor(image_festival, cv2.COLOR_BGR2RGB)

image_cactus = cv2.imread("cactus.jpg", cv2.IMREAD_COLOR)
image_cactus_rgb = cv2.cvtColor(image_cactus, cv2.COLOR_BGR2RGB)

image_yosemite = cv2.imread("yosemite.png", cv2.IMREAD_COLOR)
image_yosemite_rgb = cv2.cvtColor(image_yosemite, cv2.COLOR_BGR2RGB)

image_trains = cv2.imread("virgintrains.jpg", cv2.IMREAD_COLOR)
image_trains_rgb = cv2.cvtColor(image_trains, cv2.COLOR_BGR2RGB)


def save_img(file_name, image):
    cv2.imwrite(file_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def convolution(image: np.ndarray, kernel: np.ndarray, kernel_width: int,
                kernel_height: int, add: bool, in_place: bool = False) -> np.ndarray:
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise "Kernel must have odd numbers of rows and columns"

    if in_place == False:
        image = image.copy()

    image_width = image.shape[0]
    image_height = image.shape[1]

    kernel_half_width = np.short(np.floor(kernel.shape[0]/2))
    kernel_half_height = np.short(np.floor(kernel.shape[1]/2))

    result_img = np.zeros_like(image)
    normalized_kernel = util.normalize_kernel(kernel)
    src_img = image

    if src_img.ndim == 3:
        zeros_lr = np.zeros((kernel_half_width, image_height, 3))
        zeros_tb = np.zeros(
            (image_width + kernel_width - 1, kernel_half_height, 3))
    elif src_img.ndim == 2:
        zeros_lr = np.zeros((kernel_half_width, image_height))
        zeros_tb = np.zeros(
            (image_width + kernel_width - 1, kernel_half_height))

    src_img = np.append(zeros_lr, src_img, axis=0)
    src_img = np.append(src_img, zeros_lr, axis=0)
    src_img = np.append(zeros_tb, src_img, axis=1)
    src_img = np.append(src_img, zeros_tb, axis=1)

    for m in range(0, image_width):
        for n in range(0, image_height):
            sample_image = src_img[m:m + kernel_width, n:n + kernel_height]

            if src_img.ndim == 3:
                for i in range(0, 3):
                    result_img[m][n][i] = np.sum(
                        normalized_kernel * sample_image[:, :, i])
            elif src_img.ndim == 2:
                result_img[m][n] = np.sum(normalized_kernel * sample_image)

    if add == True:
        result_img += 128

    return result_img


"""
Task 2: Gaussian blur

Implement the function

gaussian_blur_image(image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray 

to Gaussian blur an image. "sigma" is the standard deviation of the Gaussian.
Use the function mean_blur_image as a template, create a 2D Gaussian filter
as the kernel and call the convolution function of Task 1.
Normalize the created kernel using the function normalize_kernel() (to
be implemented in a lab) before convolution. For the Gaussian kernel, use
kernel size = 2*radius + 1 (same as the Mean filter) and radius = int(math.ceil(3 * sigma))
and the proper normalizing constant.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0,
and save as "task2.png".
"""


def get_prepared_gkern(sigma: float):
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1

    kernel = util.gkern(kernel_size, sigma)
    return util.normalize_kernel(kernel)


def gaussian_blur_image(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    kernel = get_prepared_gkern(sigma)
    return convolution(image, kernel, kernel.shape[0], kernel.shape[1], False, in_place)


# new_im = gaussian_blur_image(image_festival_rgb, 4.0)
# plt.figure()
# plt.imshow(new_im)
# plt.show()
# save_img("task2.png", new_im)

"""
Task 3: Separable Gaussian blur

Implement the function

separable_gaussian_blur_image (image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray

to Gaussian blur an image using separate filters. "sigma" is the standard deviation of the Gaussian.
The separable filter should first Gaussian blur the image horizontally, followed by blurring the
image vertically. Call the convolution function twice, first with the horizontal kernel and then with
the vertical kernel. Use the proper normalizing constant while creating the kernel(s) and then
normalize using the given normalize_kernel() function before convolution. The final image should be
identical to that of gaussian_blur_image.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0, and save as "task3.png".
"""


def separable_gaussian_blur_image(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    kernel = get_prepared_gkern(sigma)
    kernel_horizontal = kernel[:, [math.floor(kernel.shape[0] / 2)]]
    kernel_vertical = kernel[[math.floor(kernel.shape[1] / 2)], :]

    image = convolution(image, kernel_horizontal, kernel_horizontal.shape[0],
                        kernel_horizontal.shape[1], False, in_place)
    image = convolution(
        image, kernel_vertical, kernel_vertical.shape[0], kernel_vertical.shape[1], False, in_place)

    return image


# new_im = separable_gaussian_blur_image(image_festival_rgb, 4.0)
# plt.figure()
# plt.imshow(new_im)
# plt.show()
# save_img("task3.png", new_im)

"""
Task 4: Image derivatives

Implement the functions

first_deriv_image_x(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray
first_deriv_image_y(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray and
second_deriv_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray

to find the first and second derivatives of an image and then Gaussian blur the derivative
image by calling the gaussian_blur_image function. "sigma" is the standard deviation of the
Gaussian used for blurring. To compute the first derivatives, first compute the x-derivative
of the image (using the horizontal 1*3 kernel: [-1, 0, 1]) followed by Gaussian blurring the
resultant image. Then compute the y-derivative of the original image (using the vertical 3*1
kernel: [-1, 0, 1]) followed by Gaussian blurring the resultant image.
The second derivative should be computed by convolving the original image with the
2-D Laplacian of Gaussian (LoG) kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] and then applying
Gaussian Blur. Note that the kernel values sum to 0 in these cases, so you don't need to
normalize the kernels. Remember to add 128 to the final pixel values in all 3 cases, so you
can see the negative values. Note that the resultant images of the two first derivatives
will be shifted a bit because of the uneven size of the kernels.

To do: Compute the x-derivative, the y-derivative and the second derivative of the image
"cactus.jpg" with a sigma of 1.0 and save the final images as "task4a.png", "task4b.png"
and "task4c.png" respectively.
"""


def first_deriv_image_x(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    kernel = np.array([[-1, 0, 1]])
    image = convolution(image, kernel, kernel.shape[0],
                        kernel.shape[1], True, in_place)
    return gaussian_blur_image(image, sigma, in_place)


def first_deriv_image_y(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    kernel = np.array([[-1], [0], [1]])
    image = convolution(image, kernel, kernel.shape[0],
                        kernel.shape[1], True, in_place)
    return gaussian_blur_image(image, sigma, in_place)


def second_deriv_image(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    image = convolution(image, kernel, kernel.shape[0],
                        kernel.shape[1], True, in_place)
    return gaussian_blur_image(image, sigma, in_place)


# new_im = second_deriv_image(image_cactus_rgb, 4.0)
# save_img("task4c.png", new_im)
# plt.figure()
# plt.imshow(new_im)
# plt.show()


"""
Task 5: Image sharpening

Implement the function
sharpen_image(image : np.ndarray, sigma : float, alpha : float, in_place : bool = False) -> np.ndarray
to sharpen an image by subtracting the Gaussian-smoothed second derivative of an image, multiplied
by the constant "alpha", from the original image. "sigma" is the Gaussian standard deviation. Use
the second_deriv_image implementation and subtract back off the 128 that second derivative added on.

To do: Sharpen "yosemite.png" with a sigma of 1.0 and alpha of 5.0 and save as "task5.png".
"""


def sharpen_image(image: np.ndarray, sigma: float, alpha: float, in_place: bool = False) -> np.ndarray:
    sec_d_blur_img = second_deriv_image(image, sigma) - 128.
    sec_d_blur_img *= alpha
    result_img = image - sec_d_blur_img
    return np.clip(result_img, 0, 255).astype(np.uint8)


# new_im = sharpen_image(image_yosemite_rgb, 1.0, 1.5)
# plt.figure()
# plt.imshow(new_im)
# plt.show()
# save_img("task5.png", new_im)


"""
Task 6: Edge Detection

Implement 
sobel_image(image : np.ndarray, in_place : bool = False) -> np.ndarray
to compute edge magnitude and orientation information. Convert the image into grayscale.
Use the standard Sobel masks in X and Y directions:
[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] and [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] respectively to compute
the edges. Note that the kernel values sum to 0 in these cases, so you don't need to normalize the
kernels before convolving. Divide the image gradient values by 8 before computing the magnitude and
orientation in order to avoid spurious edges. sobel_image should then display both the magnitude and
orientation of the edges in the image.

To do: Compute Sobel edge magnitude and orientation on "cactus.jpg" and save as "task6.png".
"""

# https://en.wikipedia.org/wiki/Sobel_operator


def sobel_image(image: np.ndarray, in_place: bool = False) -> np.ndarray:
    image = image.astype("float32")

    x_dir_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_dir_mask = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    image_x = convolution(
        gray, x_dir_mask, x_dir_mask.shape[0], x_dir_mask.shape[1], False, in_place) * 1./8
    image_y = convolution(
        gray, y_dir_mask, y_dir_mask.shape[0], y_dir_mask.shape[1], False, in_place) * 1./8

    magnitude = np.hypot(image_x, image_y)
    direction = np.arctan2(image_x, image_y)

    return magnitude, direction


# magnitude, direction = sobel_image(image_cactus_rgb)
# mag_dir = magnitude - direction
# plt.figure()
# plt.imshow(magnitude - direction, cmap="gray")
# plt.show()
# save_img("task6.png", mag_dir)


"""
Task 7: Bilinear Interpolation

Implement the function
bilinear_interpolation(image : np.ndarray, x : float, y : float) -> np.ndarray

to compute the linearly interpolated pixel value at the point (x,y) using bilinear interpolation.
Both x and y are real values. Put the red, green, and blue interpolated results in the vector "rgb".

To do: The function rotate_image will be implemented in a lab and it uses bilinear_interpolation
to rotate an image. Rotate the image "yosemite.png" by 20 degrees and save as "task7.png".
"""

# https://en.wikipedia.org/wiki/Bilinear_interpolation


def bilinear_interpolation(image: np.ndarray, x: float, y: float) -> np.ndarray:
    if x < 0 or x > image.shape[1] - 1 or y < 0 or y > image.shape[0] - 1:
        return (0, 0, 0)

    x0 = math.floor(x)
    x1 = math.ceil(x)
    y0 = math.floor(y)
    y1 = math.ceil(y)

    p_00 = image[y0][x0]
    p_01 = image[y0][x1]
    p_10 = image[y1][x0]
    p_11 = image[y1][x1]

    R1 = (y1 - y) * p_01 + (y - y0) * p_11
    R2 = (y1 - y) * p_00 + (y - y0) * p_10

    P = (x1 - x) * R1 + (x - x0) * R2
    return P


def rotate_image(image: np.ndarray, rotation_angle: float, in_place: bool = False) -> np.ndarray:
    return util.rotate_image(bilinear_interpolation, image, rotation_angle, in_place)


# new_im = rotate_image(image_yosemite_rgb, 20.)
# plt.figure()
# plt.imshow(new_im)
# plt.show()
# save_img("task7.png", new_im)


"""
Task 8: Finding edge peaks

Implement the function
find_peaks_image(image : np.ndarray, thres : float, in_place : bool = False) -> np.ndarray
to find the peaks of edge responses perpendicular to the edges. The edge magnitude and orientation
at each pixel are to be computed using the Sobel operators. The original image is again converted
into grayscale in the starter code. A peak response is found by comparing a pixel's edge magnitude
to that of the two samples perpendicular to the edge at a distance of one pixel, which requires the
bilinear_interpolation function
(Hint: You need to create an image of magnitude values at each pixel to send as input to the
interpolation function).
If the pixel's edge magnitude is e and those of the other two are e1 and e2, e must be larger than
"thres" (threshold) and also larger than or equal to e1 and e2 for the pixel to be a peak response.
Assign the peak responses a value of 255 and everything else 0. Compute e1 and e2 as follows:

(please check the separate task8.pdf)

To do: Find the peak responses in "virgintrains.jpg" with thres = 40.0 and save as "task8.png".
What would be a better value for thres?
"""


# helper function for find_peaks_image that returns 0# if image is out, not (0,0,0) when image is colored
def bilinear_interpolation_dimensions_check(image: np.ndarray, x: float, y: float) -> np.ndarray:
    if x < 0 or x > image.shape[1] - 1 or y < 0 or y > image.shape[0] - 1:
        return 0
    return bilinear_interpolation(image, x, y)


def find_peaks_image(image: np.ndarray, thres: float, in_place: bool = False) -> np.ndarray:
    magnitude, direction = sobel_image(image)

    img = np.zeros_like(image)

    image_width = image.shape[0]
    image_height = image.shape[1]

    for c in range(0, image_width):
        for r in range(0, image_height):
            theta = direction[c][r]
            if math.isnan(theta):
                theta = 0

            e1x = c + 1 * np.cos(theta)
            e1y = r + 1 * np.sin(theta)
            e2x = c - 1 * np.cos(theta)
            e2y = r - 1 * np.sin(theta)

            e = magnitude[c][r]
            e1 = bilinear_interpolation_dimensions_check(magnitude, e1x, e1y)
            e2 = bilinear_interpolation_dimensions_check(magnitude, e2x, e2y)

            if e > thres and e > e1 and e > e2:
                img[c][r] = 255.

    return img


# new_im = find_peaks_image(image_trains_rgb, 40.)
# plt.figure()
# plt.imshow(new_im)
# plt.show()
# save_img("task8.png", new_im)

"""
Task 9 (a): K-means color clustering with random seeds (extra task)

Implement the function

random_seed_image(image : np.ndarray, num_clusters : int, in_place : bool = False) -> np.ndarray

to perform K-Means Clustering on a color image with randomly selected initial cluster centers
in the RGB color space. "num_clusters" is the number of clusters into which the pixel values
in the image are to be clustered. Use random.randint(0,255) to initialize each R, G and B value.
to create #num_clusters centers, assign each pixel of the image to its closest cluster center
and then update the cluster centers with the average of the RGB values of the pixels belonging
to that cluster until convergence. Use max iteration # = 100 and L1 distance between pixels,
i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30 (or anything suitable). Note: Your code
should account for the case when a cluster contains 0 pixels during an iteration. Also, since
this algorithm is random, you will get different resultant images every time you call the function.

To do: Perform random seeds clustering on "flowers.png" with num_clusters = 4 and save as "task9a.png".
"""


def random_seed_image(image: np.ndarray, num_clusters: int, in_place: bool = False) -> np.ndarray:
    "implement the function here"
    raise "not implemented yet!"


"""
Task 9 (b): K-means color clustering with pixel seeds (extra)

Implement the function
pixel_seed_image(image : np.ndarray, num_clusters: int, in_place : bool = False)
to perform K-Means Clustering on a color image with initial cluster centers sampled from the
image itself in the RGB color space. "num_clusters" is the number of clusters into which the
pixel values in the image are to be clustered. Choose a pixel and make its RGB values a seed
if it is sufficiently different (dist(L1) >= 100) from already-selected seeds. Repeat till
you get #num_clusters different seeds. Use max iteration # = 100 and L1 distance between pixels,
 i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
 the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30.

To do: Perform pixel seeds clustering on "flowers.png" with num_clusters = 5 and save as "task9b.png".
"""


def pixel_seed_image(image: np.ndarray, num_clusters: int, in_place: bool = False) -> np.ndarray:
    "implement the function here"
    raise "not implemented yet!"
