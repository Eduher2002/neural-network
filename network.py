"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        #Se establece el numero de neuronas que tendra cada capa de la red neuronal
        self.num_layers = len(sizes)
        self.sizes = sizes
        #se generan numeros aleatorios para los biases y los pesos de cada neurona sigmoide 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        #Aqui se define la funcion sigmoide (la salida y/o entrada que tendra cada neurona) con ayuda de los pesos y los biases generados en el paso anterior
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #Estas son las variables con las que va a trabajar en esta sección SGD del código 
            #Aqui comienza el SGD, se llama a los datos de entrenamiento, dichos datos en conjunto forman el mini batch, una vez que se terminen los datos de entrenamiento de un mini batch se cumplira una epoca, tambien se define el eta que se usara para re-definir los pesos y los biases.
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            #Comienza a decir cuandos datos de entrenamiento se tendran en cuenta para el entrenamiento de la red. 
            test_data = list(test_data)
            n_test = len(test_data) #guarda en la variable n_test el numero de datos de prueba

        training_data = list(training_data)
        n = len(training_data) #Aqui guarda en la variable n el numero de datos de entrenamiento. 
        for j in range(epochs):
            #Comienza a iterar en el rango del numero de epocas y digamos que "barajea o reorganiza" los datos de entrada del minibatch, o los datos de entrenamiento aleatorios
            random.shuffle(training_data) #Se general los datos del mini batch aleatorios, estos datos recordemos que son de entrenamiento.
            #Aqui define el tamaño de los mini batches, y cada que se termina una epoca se vuelven a evaluar con otros valores aleatorios de otro minibatch, o sea de otro conjunto de entradas aleatorias.
            mini_batches = [
                training_data[k:k+mini_batch_size] # Aqui se dice con cuantos minibatches se va a trabajar, pues recordemos que todo el conjunto de entradas se divide en pequeñas muestras de datos de entrada aleatorios 
                for k in range(0, n, mini_batch_size)] 
            for mini_batch in mini_batches:
                #para cada mini batch se calcula un paso en el descenso del gradiente
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                #aqui la verdad no entiendo muy bien profe , pero me parece que el programa escribe los resultados que va obteniendo en cada epoca con su respectivo mini batch.
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                #Aqui me parece que el programa interpreta que sino ha terminado una epoca, entonces que la siga ejecutando hasta terminarla. 
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
    
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        #Inicializa los biases y pesos en ceros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: #Para cada dato de la lista de tuplas (x,y) se calcula el gradiente en ese punto para esa b y ese peso en especifico. 
            #Dado un mini batch, se calcularan las entradas del gradiente de la funcion de costo usando backpropagation. 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #aqui manda a llamar a la funcion backprop (la cual se define abajo) y guarda los resultados de las tuplas dadas por el backprop en las variables nabla_b y nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            #Aqui se modifica el valor de los pesos y los biases que deberian tener para disminuir la funcion de costo
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        #inicializa en ceros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            #Se dice como esta representado el argumento de cada sigmoide, es decir, el numero que ingresa a la funcion de activacion.
            #Se expresa como el producto punto, pues w y x en realidad son vectores 
            z = np.dot(w, activation)+b
            zs.append(z) #Aqui agrega elementos al final de dicha lista
            activation = sigmoid(z) #Se reconoce la funcion de activacion como la funcion sigmoide cuyo argumento es la variable z, la cual ya esta definida arriba 
            activations.append(activation) #Analogamente se agregan elementos al final de esta lista, dicha lista es la salida de la funcion de activacion (la cual a su vez es un vector)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) 
        nabla_b[-1] = delta #Este es el elemento del gradiente de la funcion de costo con respecto a b, el -1 indica que se comienza a partir del ultimo elemento de la lista, pues recordemos que el backpropagation va hacia atras usando la regla de la cadena. 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #Este es el analogo de nabla_b, solo que se coloca la operacion de transpuesta por la comodidad de los indices. 
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): #Define el corrimiento de la l, el indice l corre sobre el numero de capas que tiene la red neuronal. 
            z = zs[-l]
            sp = sigmoid_prime(z) #Se guarda la funcion de la derivada de la funcion sigmoide en la variable sp
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #Este es el cambio en la salida de la l-esima capa
            nabla_b[-l] = delta #Aqui unicamente se define la omponente del gradiente de la funcion de costo respecto de los biases.
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #Aqui se expreso la funcion a la que llegamos en clase, es decir, la entrada del vector del gradiente d ela funcion de costo respecto de los pesos como el producto entre las matrices de las salidas de las neuronas de cada capa con la matriz de los cambios en la j-esima neurona de la l-esima capa
        return (nabla_b, nabla_w) # Aqui devuelve las componentes del vector gradiente de la funcion de costo. 

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data] #En estos pasos evalua la efectividad de la red neuronal a traves del test data
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z)) #forma explicita de la funcion sigmoide

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) #Se expresa mas convenientemente la derivada de la funcion de activacion 
