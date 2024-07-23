import numpy as np

class Neuron:

    INPUT_NEURONS_NUM = 2
    HIDDEN_NEURONS_NUM = 3
    OUTPUT_NEURONS_NUM = 2
    
    neurons = []
    last_id = 0

    input_weights = np.random.rand(3,2) - 0.5
    output_weights = np.random.rand(3,2) - 0.5

    hidden_biases = np.zeros(HIDDEN_NEURONS_NUM)
    output_biases = np.zeros(OUTPUT_NEURONS_NUM)


    def __init__(self, type:int, val=0, bias=0):
        self.id = Neuron.last_id + 1
        self.bias = bias
        self.type = type
        self.val = val

        Neuron.neurons.append(self)
        Neuron.last_id = self.id

    def __repr__(self):
        return f"Neuron{self.id}(type: {self.check_type()}, bias: {self.bias} value: {self.val})"
    
    def check_type(self):
        if self.type == -1:
            return "Input"
        elif self.type == 0:
            return "Hidden"
        elif self.type == 1:
            return "Output"

    @staticmethod
    def ReLU(value):
        if value < 0:
            return 0
        else:
            return value
        

def create_input_neurons(*inputs):
    for input in inputs:
        Neuron(-1, input)


def create_hidden_neurons():
    #initializing values
    inputs = np.array([i.val for i in Neuron.neurons if i.type == -1])
    biases = Neuron.hidden_biases
    weights = Neuron.input_weights

    #creating hidden neurons by matrix multiplication
    hiddens = weights.dot(inputs)
    hiddens = hiddens + biases

    hidden_neurons = np.array([])

    #ReLU function
    for neuron in hiddens:
        value = Neuron.ReLU(neuron)
        hidden_neurons = np.append(hidden_neurons, [value])

    #creating neurons with Neuron class
    for neuron in hidden_neurons:
        Neuron(0, neuron)

    


create_input_neurons(167, 56)
create_hidden_neurons()
print(Neuron.neurons)
print(Neuron.hidden_biases, Neuron.output_biases)
    



    