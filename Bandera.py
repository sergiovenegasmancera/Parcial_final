import cv2
import numpy as np
import math
import sys
import os

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

class Bandera():
    def __init__(self, flag):
        self.flag = flag

    def colores(self):
        image = cv2.cvtColor(self.flag, cv2.COLOR_BGR2RGB)
        method = 'kmeans'

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        image = np.array(image, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))

        # Fitting model on a small sub-sample of the data
        image_array_sample = shuffle(image_array, random_state=0)[:10000]

        # Display all results, alongside original image
        ##################################
        #plt.figure(1)
        #plt.clf()
        #plt.axis('off')
        #plt.title('Original image')
        #plt.imshow(image)
        ##################################
        Euclidean_distance = []
        K = []

        for i in range(4):
            n_colors = i + 1

            if method == 'gmm':
                model = GMM(n_components=n_colors).fit(image_array_sample)
            else:
                model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

            # Get labels for all points
            if method == 'gmm':
                labels = model.predict(image_array)
                centers = model.means_
            else:
                labels = model.predict(image_array)
                centers = model.cluster_centers_

            # Distance of each color to the center of the cluster
            aux = np.zeros((1, n_colors))

            for j in range(rows * cols):
                R = math.pow(image_array[j, 0] - centers[labels[j], 0], 2)
                G = math.pow(image_array[j, 1] - centers[labels[j], 1], 2)
                B = math.pow(image_array[j, 2] - centers[labels[j], 2], 2)
                distance = math.sqrt(R + G + B)
                aux[0, labels[j]] += distance

            K.append(n_colors)
            Euclidean_distance.append(np.sum(aux))

            #plt.figure(2)
            #plt.clf()
            #plt.axis('off')
            #plt.title('Quantized image ({} colors, method={})'.format(n_colors, method))


            #d = centers.shape[1]
            #image_clusters = np.zeros((rows, cols, d))
            #label_idx = 0
            #for a in range(rows):
            #    for b in range(cols):
            #        image_clusters[a][b] = centers[labels[label_idx]]
            #        label_idx += 1

            #plt.imshow(image_clusters)

            #plt.show()

        Euclidean_distance = np.array(Euclidean_distance)
        min_val = Euclidean_distance.min()
        min_pos = int(Euclidean_distance.argmin())+1


        #plt.plot(K, Euclidean_distance)
        #plt.xlabel('K')
        #plt.ylabel('Euclidean distance')
        #plt.title('suma de distancias intra-cluster vs n_color')
        #plt.show()

        return min_pos

    def porcentaje(self):
        # Gray levels histogram
        image_gray = cv2.cvtColor(self.flag, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
        hist = np.array(hist)
        porc = np.zeros(4)
        indices = np.array(np.where(hist > 0))
        total = 0
        for i in range(indices.shape[1]):
            total += hist[indices[0,i],:]
            porc[i] = hist[indices[0,i],:]

        porc /= total
        #plt.plot(hist, color='gray')
        #plt.xlim([0, 256])
        #plt.show()
        return porc

    def orientacion(self):
        window_name = ('Sobel Demo - Simple Edge Detector')
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        src = self.flag
        # Check if image is loaded fine

        src = cv2.GaussianBlur(src, (3, 3), 0)

        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        # Gradient-Y
        # grad_y = cv.Scharr(gray,ddepth,0,1)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        sum_x = np.sum(abs_grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        sum_y = np.sum(abs_grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        #cv2.imshow(window_name, grad)
        #cv2.waitKey(0)
        if sum_x > 0 and sum_y == 0:
            orientacion = 'vertical'
        elif sum_y > 0 and sum_x == 0:
            orientacion = 'horizontal'
        else:
            orientacion = 'mixta'

        return orientacion




