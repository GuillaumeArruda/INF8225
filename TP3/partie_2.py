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
    if(x > 0):
        return 1
    else:
        return a
   
#Softmax function found at https://gist.github.com/aelaguiz/4453854
def softmax(w):
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist

vector_pw_linear = np.vectorize(pw_linear, otypes=[np.float])
vector_pw_linear_derivative = np.vectorize(pw_linear_derivative, otypes=[np.float])

class NeuralLayer:
    def __init__(self,previous_layer, layers, number_of_layers, a, learning_rate):
        if number_of_layers > 1:
            self.nextLayer = NeuralLayer(self, layers, number_of_layers - 1, a, learning_rate)
        else:
            self.nextLayer = None
        self.previous_layer = previous_layer
        self.theta = (np.random.randn(layers[-number_of_layers],layers[-number_of_layers - 1] + 1)) / np.sqrt(layers[-number_of_layers - 1] + 1)
        self.a = a
        self.learning_rate = learning_rate

    def feedfoward(self, X):
        self.input = np.append(X,np.ones((1,np.size(X,axis=1))),axis=0)
        self.inValue = np.dot(self.theta, self.input)
        if self.nextLayer == None:
            self.activation = softmax(self.inValue)
            return self.activation
        else:
            self.activation = vector_pw_linear(self.inValue, self.a)
            return self.nextLayer.feedfoward(self.activation)

    def backpropagate(self, X, Y):
        if self.nextLayer == None:
            self.delta = -(Y - self.activation)
            #lossFunction = np.sum(Y * np.log(self.activation))
            #print lossFunction
        else:
            self.nextLayer.backpropagate(X,Y)
            derivative = vector_pw_linear_derivative(self.inValue, self.a)
            self.delta = derivative * np.dot(self.nextLayer.theta[:,:-1].T, self.nextLayer.delta)

    def update(self, minibatchSize):
        if self.nextLayer != None:
            self.nextLayer.update(minibatchSize)
        self.theta = self.theta - self.learning_rate * np.dot(self.delta, self.input.T)

class NeuralNetwork:
    def __init__(self, layers, a, learning_rate):
        self.starting_layer = NeuralLayer(None,layers, len(layers) - 1, a, learning_rate)

    def train(self, train_x, train_y, valid_x, valid_y, minibatchSize):
        n = len(train_x)
        mini_batches_x = [train_x[k:k+minibatchSize] for k in xrange(0, n, minibatchSize)]
        mini_batches_y = [train_y[k:k+minibatchSize] for k in xrange(0, n, minibatchSize)]
        converged = False
        lastPrecision = -1
        precision = 0
        while ~converged:
            i = 0
            for mini_batch_x, mini_batch_y in zip(mini_batches_x, mini_batches_y):
                f = self.starting_layer.feedfoward(mini_batch_x.T)
                f = self.starting_layer.backpropagate(mini_batch_x.T, mini_batch_y.T)
                self.starting_layer.update(minibatchSize)
                i = i + 1
                if i % 10 == 0:
                    precision = self.evaluate(valid_x.T, valid_y.T)
                    converged = precision < lastPrecision
                    lastPrecision = precision
                    print precision
                    if converged:
                        break

    def evaluate(self, test_x, test_y):
        results = self.starting_layer.feedfoward(test_x)
        results = [np.argmax(y) for y in results.T]
        correct = sum(int(x == y) for (x,y) in zip(results, test_y))
        return float(correct) / np.size(test_x,axis=1)

if __name__ == '__main__':
    with gzip.open('mnist.pkl.gz','rb') as f:
        train_set,valid_set, test_set = pickle.load(f)
    np.random.seed(1)
    number_of_possible_y = 10
    learning_rate = 0.00005
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set
    train_y_oh = numberListToOneHotVectors(train_y, number_of_possible_y)
    valid_y_oh = numberListToOneHotVectors(valid_y, number_of_possible_y)
    test_y_oh = numberListToOneHotVectors(test_y, number_of_possible_y)

    layers = np.array([784, 30, 10])
    network = NeuralNetwork(layers, 0.05 , learning_rate)
    network.train(train_x, train_y_oh, valid_x, valid_y, 100)





