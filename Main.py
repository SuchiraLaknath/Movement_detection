import cv2
import numpy as np
import math

## Parameters ##
video_path = './Data/video.mp4'

def convolution(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

            output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output


def normal(x, mean, standard_deviasion):
    return 1 / (np.sqrt(2 * np.pi) * standard_deviasion * np.e ** (-np.power((x - mean) / standard_deviasion, 2) / 2))


def g_kernel(ksize, sigma=1):
    kernel_1D = np.linspace(-(ksize // 2), ksize // 2, ksize)
    for i in range(ksize):
        kernel_1D[i] = normal(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D


def gaussian_blur(image, kernel_size):
    sigma = math.sqrt(kernel_size)
    kernel = g_kernel(kernel_size, sigma=sigma)
    return convolution(image, kernel)


def differenceImage(frame1, frame2):
    a = frame1 - frame2
    b = np.uint8(frame1 < frame2) * 254 + 1
    return a * b


def threshold(frame, threshold_value):
    return ((frame > threshold_value) * 255).astype('uint8')


def main():

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    frame1 = cv2.resize(frame1, (480, 360))
    frame2 = cv2.resize(frame2, (480, 360))

    while cap.isOpened():

        framek = frame1
        framek1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        framek2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = differenceImage(framek1, framek2)

        # The Gaussian smoothing step is bypassed for saving computation power
        ###################################

        #diff = gaussian_blur(diff, 5)

        #########################################
        thresh = threshold(diff, 10)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if h > (frame_height*0.3) and w > (frame_width*0.02):
                cv2.rectangle(framek, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #image = cv2.resize(framek, (1280, 720))

        cv2.imshow("feed", framek)

        frame1 = frame2.copy()
        ret, frame2 = cap.read()
        frame2 = cv2.resize(frame2, (480, 360))
        if cv2.waitKey(40) == 27:
            break
            cv2.destroyAllWindows()
            cap.release()
    
if __name__ == '__main__':
    
    main()