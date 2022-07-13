import numpy as np
import cv2
import os
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def read_image():
    image = []
    for filename in os.listdir('images'):
        img = io.imread(os.path.join('images', filename))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image.append(img_gray)

    return image


def otsu(img):
    fig = plt.figure()
    fig.set_size_inches(14,20)
    spec = gridspec.GridSpec(ncols=5, nrows=8, figure=fig)

    otsu_algorithm = []
    otsu_opencv = []
    for i in range(8):
        pixel_number = img[i].shape[0] * img[i].shape[1]
        hist, bin = np.histogram(img[i], np.array(range(0, 257)))
        # print(hist,bin)
        # exit()
        final_thresh = -1
        final_var = -1
        intensity_arr = np.arange(256)
        # t-->threshold
        for t in bin[1:-1]:  # bin--> 1 - 255
            pixelCount_b = np.sum(hist[:t])  # pixel count background
            pixelCount_f = np.sum(hist[t:])  # pixel count forground
            Wb = pixelCount_b / pixel_number
            Wf = pixelCount_f / pixel_number

            miub = np.sum(intensity_arr[:t] * hist[:t]) / float(pixelCount_b)
            miuf = np.sum(intensity_arr[t:] * hist[t:]) / float(pixelCount_f)

            variance = Wb * Wf * (miub - miuf) ** 2
            if variance > final_var:
                final_thresh = t - 1
                final_var = variance

        otsu_algorithm.append(final_thresh)

        final_img = img[i].copy().astype('uint8')
        final_img[final_img > final_thresh] = 255
        final_img[final_img < final_thresh] = 0

        # OpenCv otsu threshold
        th, cvimg = cv2.threshold(img[i], 0, 255, cv2.THRESH_OTSU)
        otsu_opencv.append(int(th))

        ax1 = fig.add_subplot(spec[i, 0])
        ax2 = fig.add_subplot(spec[i, 1])
        ax3 = fig.add_subplot(spec[i, 2])
        ax4 = fig.add_subplot(spec[i, 3])
        ax5 = fig.add_subplot(spec[i, 4])
        ax1.set_title('Original image',fontsize=4)
        ax2.set_title('Otsu image',fontsize=4)
        ax3.set_title('OpenCv image',fontsize=4)
        ax4.set_title('Histogram original with threshold',fontsize=4)
        ax5.set_title('Histogram otsu with threshold',fontsize=4)
        ax1.imshow(img[i], cmap='gray')
        ax2.imshow(final_img, cmap='gray')
        ax3.imshow(cvimg, cmap='gray')

        # remove tick label
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])

        # Histogram
        _, _, bars = ax4.hist(img[i].ravel(), 255, [0, 255], color='blue')
        for bar in bars:
            if bar.get_x() > final_thresh:
                bar.set_facecolor('yellow')
        ax4.axvline(final_thresh, linestyle='--', linewidth=1, color='gray')

        ax5.hist(final_img.ravel(), 255, [0, 255], color='red')
        ax5.axvline(final_thresh, linestyle='--', linewidth=1, color='gray')


    plt.tight_layout()
    plt.savefig('otsu images.jpg')
    plt.show()
    print('the best threshold with otsu algorithm is:', otsu_algorithm)
    print('the best threshold with otsu openCv is:', otsu_opencv)


if __name__ == '__main__':
    img = read_image()
    otsu(img)
