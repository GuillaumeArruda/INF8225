import gzip
import pickle
import numpy as np

def numberListToOneHotVectors(values, size):
    oneHotVectors = np.zeros([values.size, size])
    for (i,value) in enumerate(values):
        oneHotVectors[i][value] = 1
    return oneHotVectors

def pw_linear(x, a):
    if(x >= 0):
        return x
    else:
        return a * x

def pw_linear_derivative(x,a):
    if(x >= 0):
        return 1
    else:
        return a
    
def softmax(x):
    e = np.exp(x)
    dist = e / np.sum(e)
    return dist


vector_pw_linear = np.vectorize(pw_linear, otypes=[np.float])
vector_pw_linear_derivative = np.vectorize(pw_linear_derivative, otypes=[np.float])

class OutputNeuralLayer:
    def __init__(self,layers,previousLayer, learning_rate):
        self.learning_rate = learning_rate
        self.W = np.random.random([layers[0],layers[0]])
        self.previousLayer = previousLayer
        self.nabla = np.zeros(self.W.shape)
        return

    def feedfoward(self,input):
        self.input = input
        self.inValue = np.dot(self.W,input)
        self.activation = softmax(self.inValue)
        return self.activation

    def backpropagate(self,Y):
        self.deltas = -(Y.T - self.activation)
        return self.deltas

    def update(self):
        self.nabla = self.nabla + self.deltas
        self.W = self.W - self.nabla
        return

class NeuralLayer:
    def __init__(self, layers, a, previousLayer, number_of_layer, learning_rate):
        if number_of_layer > 2:
            self.nextLayer = NeuralLayer(layers,a,self,number_of_layer - 1, learning_rate)
        else:
            self.nextLayer = OutputNeuralLayer(layers,self,learning_rate)
        self.a = a
        self.learning_rate = learning_rate
        self.W = np.random.random([layers[number_of_layer - 2],layers[number_of_layer - 1]])
        self.previousLayer = previousLayer
        self.nabla = np.zeros(self.W.shape)
        return

    def feedfoward(self, input):
        self.input = input
        self.inValue = np.dot(self.W, input)
        self.activation = vector_pw_linear(self.inValue,self.a)
        return self.nextLayer.feedfoward(self.activation)

    def backpropagate(self, Y):
        deltas = self.nextLayer.backpropagate(Y)
        derivative = vector_pw_linear_derivative(self.inValue, self.a)
        self.deltas = np.dot(self.W.T, deltas.T) * derivative
        return self.deltas.T
    
    def update(self):
        if self.nextLayer != None:
            self.nextLayer.update()
        self.nabla = self.nabla + self.deltas
        self.W = self.W - self.nabla
        return




class NeuralNetwork:
    def __init__(self, layers, a, learning_rate):
        self.number_of_layers = layers.size
        self.layer = NeuralLayer(layers, a, None, self.number_of_layers, learning_rate)
        return
       
    def train(self, train_x, train_y):
        i = 0
        while True:
            i += 1
            self.layer.feedfoward(train_x[i:20,:].T)
            self.layer.backpropagate(train_y[i:20,:])
            self.layer.update()
        return 



if __name__ == '__main__':
    with gzip.open('mnist.pkl.gz','rb') as f:
        train_set,valid_set, test_set = pickle.load(f)

    number_of_possible_y = 10
    learning_rate = 0.001
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set
    train_y = numberListToOneHotVectors(train_y,number_of_possible_y)
    valid_y = numberListToOneHotVectors(valid_y,number_of_possible_y)
    test_y = numberListToOneHotVectors(test_y, number_of_possible_y)

    layers = np.array([number_of_possible_y,100,784]) #From the output to the input
    network = NeuralNetwork(layers, 0.1 , learning_rate)
    network.train(train_x,train_y)




