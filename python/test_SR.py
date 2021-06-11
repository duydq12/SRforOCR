import cv2
import numpy as np


def VDSR_predict(net, im):
    # input image preprocess
    im = im.astype(np.float64)
    # preparing input data
    row, col = im.shape[:2]
    im_data = im.reshape((1, 1, row, col))

    net.setInput(im_data, 'data')
    # forward
    detections = net.forward()
    detections = detections.reshape((row, col))
    detections = np.clip(detections, 0, 255)
    im_h = detections.astype(np.uint8)
    return im_h


def process_SR(net, image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if len(image.shape) == 3:
        im_3c = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(im_3c)
        y_sr = VDSR_predict(net, y)
        im_3c = cv2.merge((y_sr, cr, cb))
        im_sr = cv2.cvtColor(im_3c, cv2.COLOR_YCrCb2RGB)
    else:
        im_sr = VDSR_predict(net, image)

    return im_sr


def main(net, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    im_sr1 = np.zeros((2 * h, 2 * w, c), dtype=np.uint8)

    im = image[:round(h / 2), :round(w / 2), :]
    im_sr1[:round(h / 2) * 2, :round(w / 2) * 2, :] = process_SR(net, im)

    im = image[round(h / 2):, :round(w / 2), :]
    im_sr1[round(h / 2) * 2:, :round(w / 2) * 2, :] = process_SR(net, im)

    im = image[:round(h / 2), round(w / 2):, :]
    im_sr1[:round(h / 2) * 2, round(w / 2) * 2:, :] = process_SR(net, im)

    im = image[round(h / 2):, round(w / 2):, :]
    im_sr1[round(h / 2) * 2:, round(w / 2) * 2:, :] = process_SR(net, im)

    img_cubic = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    test = im_sr1 - img_cubic
    cv2.imshow("input", image.astype(np.uint8))
    cv2.imshow("cubic", img_cubic)
    cv2.imshow("output", im_sr1.astype(np.uint8))
    cv2.waitKey(0)


if __name__ == '__main__':
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("../models/VDSR_mat_custom.prototxt",
                                   "../models/model1_trained_with_edge.caffemodel")
    main(net, cv2.imread("../python/8.jpg"))
