import os
import sys
import time
import operator
import cv2
import numpy as np
from numpy import linalg as LA
from PIL import Image
from sklearn import svm
import joblib  # save / load model


class ObjectDetector(object):
    def __init__(self, input_img, feature_extractor, model, nms, image_pyramid=[140, 210, 315, 470], window_step=16, window_height=128, window_width=64, prob_threshold=0.9):
        self.input_img = input_img
        self.image_pyramid = image_pyramid
        self.feature_extractor = feature_extractor
        self.model = model
        self.window_step = window_step
        self.window_height = window_height
        self.window_width = window_width
        self.prob_threshold = prob_threshold
        self.nms = nms

        self.svm_model = joblib.load(self.model)
        pass

    def read_image_with_pillow(self, img_path, is_gray=True):
        pil_im = Image.open(img_path).convert('RGB')
        img = np.array(pil_im)
        img = img[:, :, ::-1].copy()  # Convert RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def fit_svm(self, f):
        # predict
        pred_y1 = self.svm_model.predict(np.array([f]))
        pred_y = self.svm_model.predict_proba(np.array([f]))

        class_probs = pred_y[0]
        max_class, max_prob = max(
            enumerate(class_probs), key=operator.itemgetter(1))

        is_person = max_class == 1
        return is_person, max_prob

    def nms(self):
        pass

    def __call__(self):
        img = self.read_image_with_pillow(
            img_path=self.input_img, is_gray=True)
        h, w = img.shape[:2]

        d_img = cv2.imread(self.input_img)

        time_start = time.time()
        n_windows = 0
        boxes = []
        for idx, new_height in enumerate(self.image_pyramid):
            new_width = int(new_height/h*w)

            if self.window_width > new_width or self.window_height > new_height:
                continue

            new_img = cv2.resize(src=img, dsize=(new_width, new_height))
            max_x = new_width - self.window_width
            max_y = new_height - self.window_height

            print('Scale (h=%d, w=%d)' % (new_height, new_width))

            x = 0
            y = 0

            while y <= max_y:
                while x <= max_x:
                    n_windows += 1
                    patch = new_img[y:y+self.window_height,
                                    x:x+self.window_width]
                    f = self.feature_extractor(patch)
                    is_person, prob = self.fit_svm(f)

                    if is_person and prob > self.prob_threshold:
                        print('* prob: %.2f' % prob)
                        x1 = int(x/new_width*w)
                        y1 = int(y/new_height*h)
                        x2 = int((x+self.window_width)/new_width*w)
                        y2 = int((y+self.window_height)/new_height*h)
                        #cv2.rectangle(d_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        boxes.append([x1, y1, x2, y2])
                    x += self.window_step
                    pass
                x = 0
                y += self.window_step
                pass
            pass

        pboxes = self.nms(np.array(boxes))

        for box in pboxes:
            cv2.imwrite('crop_%s.jpg' % time.time(),
                        d_img[box[1]:box[3], box[0]:box[2], :])

        for box in pboxes:
            cv2.rectangle(d_img, (box[0], box[1]),
                          (box[2], box[3]), (0, 255, 0), 2)

        time_end = time.time()
        print('Processed %d windows in %.2f seconds' %
              (n_windows, time_end-time_start))
        cv2.imwrite('done.jpg', d_img)

        pass
