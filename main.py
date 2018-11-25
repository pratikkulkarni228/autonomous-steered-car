'''
Author: Pratik kulkarni
Email: pratikkulkarni228@gmail.com
'''
import numpy as np
import cv2
from tqdm import tqdm
import os

from numpy.ma import argmax

path_to_img = '../autonomous_driving/training/images/'
training_csv_file='../autonomous_driving/training/steering_angles.csv'

def prepare_X(frame_nums,path_to_img):
    input_X = []
    print('Pre processing and preparing input . . .')
    for i in range(len(frame_nums)):
        frame_num = int(frame_nums[i])
        path = path_to_img + '/' + str(int(frame_num)).zfill(4) + '.jpg'
        img_full = cv2.imread(path, 0)
        img_full = cv2.resize(img_full, (0, 0), fx=0.04, fy=0.04)
        img_full = img_full / 255  # normalise in 0 to 1
        # img_full = np.mean(img_full, axis=2) #grayscale
        input_X.append(img_full.flatten())
    input_X = np.asarray(input_X)
    print('Data Processed!')
    return input_X

def prepare_Y(training_csv_file):
    data = np.genfromtxt(training_csv_file, delimiter = ',')
    print('Pre processing and preparing label data...')
    steering_angles = data[:,1]
    input_Y=np.zeros([1500,64])
    add_array = [0.1, 0.32, 0.61, 0.89, 1, 0.89, 0.61, 0.32, 0.1]
    for i in range(len(steering_angles)):
        a = int(np.interp(steering_angles[i], [-170, 30], [0, 64]))
        if ((int(a) + 4) <= 63) and ((int(a) - 4) >= 0):
            input_Y[i][a - 4:a + 5] += add_array
    print('Done!')
    return input_Y



def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''

    #Hyper params
    data = np.genfromtxt(csv_file, delimiter=',')
    frame_nums = data[:, 0]
    steering_angles = data[:, 1]

    input_X = prepare_X(frame_nums,path_to_images)
    input_Y=prepare_Y(csv_file)

    # Train using ADAM optimizer
    # Train using ADAM optimizer
    num_iterations = 3000
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08

    m0 = np.zeros(146560)  # Initialize first moment vector
    v0 = np.zeros(146560)  # Initialize second moment vector
    t = 0.0

    losses = []  # For visualization
    mt = m0
    vt = v0

    NN = NeuralNetwork()
    print('Training your network...')
    for i in tqdm(range(num_iterations)):
        t += 1
        grads = NN.computeGradients(X=input_X, y=input_Y)
        mt = beta1 * mt + (1 - beta1) * grads
        vt = beta2 * vt + (1 - beta2) * grads ** 2
        mt_hat = mt / (1 - beta1 ** t)
        vt_hat = vt / (1 - beta2 ** t)

        params = NN.getParams()
        new_params = params - alpha * mt_hat / (np.sqrt(vt_hat) + epsilon)
        NN.setParams(new_params)

        losses.append(NN.costFunction(X=input_X, y=input_Y))





    return NN


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    im_full = cv2.imread(image_file,0)
    im_full = cv2.resize(im_full, (0, 0), fx=0.04, fy=0.04)
    img_full = im_full / 255  # normalise in 0 to 1
    #img_full = np.mean(img_full, axis=2)  # grayscale
    ## Perform inference using your Neural Network (NN) here.
    #input_X=np.reshape(img_full,[-1,2226])
    input_X = img_full.flatten()
    yHat = NN.forward(input_X)
    arg = argmax(yHat)
    angle = np.interp(arg, [0, 64], [-170, 30])
    return angle


class NeuralNetwork(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2226
        self.outputLayerSize = 64
        self.hiddenLayerSize = 64

        # Weights (parameters)
        limit = np.sqrt(6 / (self.inputLayerSize + self.hiddenLayerSize))
        self.W1 = np.random.uniform(-limit, limit, (self.inputLayerSize, self.hiddenLayerSize))

        limit = np.sqrt(6 / (self.hiddenLayerSize + self.outputLayerSize))
        self.W2 = np.random.uniform(-limit, limit, (self.hiddenLayerSize, self.outputLayerSize))

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        # print('shape of yHat',yHat.shape)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        # print('sigmoidprime shape',(np.exp(-z)/((1+np.exp(-z))**2)).shape)
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * np.sum((y - self.yHat) ** 2)
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        # print('delta3 shape: ',delta3.shape)
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    # Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))



def main():
    #Train
    NN=train(path_to_img,training_csv_file)
    predicted_angles = []
    path, dirs, files = next(os.walk(path_to_img))
    file_count = len(files)

    #Test
    for i in range(file_count):
        im_path = path_to_img+ files[i]
        #predicted_angles.append(predict(NN, im_path))
        predicted_angles.append(predict(NN,im_path))

    data = np.genfromtxt('../autonomous_driving/training/steering_angles.csv', delimiter = ',')
    steering_angles = data[:,1]

    #Calculate Loss
    RMSE = np.sqrt(np.mean((np.array(predicted_angles)- steering_angles)**2))
    #RMSE = round(RMSE, 3)
    print('Test Set RMSE = ' + str(RMSE) + ' degrees.')

if __name__=='__main__':
    main()
