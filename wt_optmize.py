# inbuilt dataset iris
from sklearn.datasets import load_iris
from sklearn import preprocessing as pp
from sklearn import utils
from sklearn import metrics
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Function to generate the intial population
def intial_pop(input_neurons, hidden_neurons, output_neurons, pop_size):

    # Generate random real values from -2.0 to 2.0
    low, high = -2.0, 2.0
    params = [dict() for x in range(pop_size)]  # array of dictionary

    for x in range(pop_size):
        wH = np.random.uniform(
            low, high, size=(hidden_neurons, input_neurons)
        )  # weights of hidden
        wO = np.random.uniform(
            low, high, size=(output_neurons, hidden_neurons)
        )  # weights of output
        bH = np.random.uniform(low, high, size=(
            hidden_neurons, 1))  # bias of hidden
        bO = np.random.uniform(low, high, size=(
            output_neurons, 1))  # bias of output

        params[x] = {"wH": wH, "wO": wO, "bH": bH, "bO": bO}

    return params

# Function to compute the losses for the population i.e. fitness
def get_loss(params, pop_size, X, Y):

    # Run a loop for the entire population
    losses = [float() for x in range(pop_size)]
    accuracy = [float() for x in range(pop_size)]

    for x in range(pop_size):
        wH = params[x]["wH"]
        bH = params[x]["bH"]
        zH = np.dot(wH, X.T) + bH

        aH = np.maximum(zH, 0)  # output of hidden layer

        # output layer
        wO = params[x]["wO"]
        bO = params[x]["bO"]
        zO = np.dot(wO, aH) + bO

        zO = zO.T  # (m, output_neurons)

        # Apply softmax to it
        aO = scipy.special.softmax(zO)

        # Stores the index of the maximum prob out of the output
        y_pred = np.argmax(aO, axis=1)

        # binarize the labels with fixed classes
        y_pred = pp.label_binarize(y_pred, classes=[0, 1, 2])

        # calculate the accuracy and loss
        accuracy[x] = metrics.accuracy_score(Y, y_pred)
        losses[x] = metrics.log_loss(Y, aO)

    return (losses, np.mean(accuracy), np.max(accuracy))


# Function to calculate the fitness of the population
def get_fitness(losses, pop_size):

    # fitness array
    fitness = [float() for x in range(pop_size)]

    # invert the losses as larger loss means smaller area
    losses = [(1.0 / x) for x in losses]

    # sum of the losses
    sum = np.sum(losses)

    # Get the fitness value from losses
    for x in range(pop_size):
        fitness[x] = (losses[x] / sum) * 100  # area in form of percentage

    return fitness


# Function to generate two parents for Roulette Wheel selection
def roulette_wheel(fitness, pop_size):

    # get 2 parents without replacement with cumulative frequency
    count = 0
    # initially both the parents are at index -1
    p1 = -1
    p2 = -1
    while count < 2:
        val = np.random.random() * 100  # get a random number between 0 and 100

        # cumulative frequency
        cf = 0
        i = 0  # index i
        while cf < val and i < pop_size:
            cf = cf + fitness[i]
            i = i + 1

        if p1 == -1:
            p1 = i-1  # store the first parent
            count = count + 1
        elif p2 == -1 and not(p1 == (i-1)):
            p2 = i-1  # store the second parent and must not be same as first
            count = count + 1

    return (p1, p2)


# Function to flatten the weights and biases matrices into one array
def flatten(params, pop_size, tot_genes):

    # contains the flattened chromosomes of the 30 population
    flattened_pop = np.ndarray((pop_size, tot_genes))

    # traverse the entire population and flatten each chromosome
    for x in range(pop_size):
        wH = np.matrix.flatten(params[x]["wH"])
        bH = np.matrix.flatten(params[x]["bH"])
        wO = np.matrix.flatten(params[x]["wO"])
        bO = np.matrix.flatten(params[x]["bO"])

        chromosome = np.concatenate((wH, wO, bH, bO))

        flattened_pop[x] = chromosome

    return flattened_pop


# Function to unflatten the array into weights and biases matrices
def unflatten(pop, pop_size, input_neurons, hidden_neurons, output_neurons):
    params = [dict() for x in range(pop_size)]  # array of dictionary

    for x in range(pop_size):
        wH = np.reshape(
            pop[x][0: (input_neurons * hidden_neurons)],
            (hidden_neurons, input_neurons),
        )
        wO = np.reshape(
            pop[x][
                (input_neurons * hidden_neurons): (
                    input_neurons * hidden_neurons + output_neurons * hidden_neurons
                )
            ],
            (output_neurons, hidden_neurons),
        )
        bH = np.reshape(
            pop[x][
                (input_neurons * hidden_neurons + output_neurons * hidden_neurons): (
                    input_neurons * hidden_neurons
                    + output_neurons * hidden_neurons
                    + hidden_neurons
                )
            ],
            (hidden_neurons, 1),
        )
        bO = np.reshape(
            pop[x][
                (
                    input_neurons * hidden_neurons
                    + output_neurons * hidden_neurons
                    + hidden_neurons
                ):
            ],
            (output_neurons, 1),
        )
        params[x] = {"wH": wH, "wO": wO, "bH": bH, "bO": bO}

    return params  # return the new population


# Function to generate the offsprings using crossover
def get_offsprings(p1, p2, cross_rate):
    # generate a random float between 0 and 1
    cross_prob = np.random.rand()

    # o1 and o2 store the offspring 1 and offspring 2
    o1, o2 = p1, p2

    # Perform crossover
    if cross_prob < cross_rate:
        # Get the boundary for the one point
        boundary = np.random.randint(0, tot_genes)

        o1 = np.append(p1[:boundary], p2[boundary:])
        o2 = np.append(p2[:boundary], p1[boundary:])

    return (o1, o2)


# Function to perform mutation on a chromosome based on the mutation rate
def mutate(chromo, tot_genes, mut_rate):
    count = 0
    low, high = -1.0, 1.0
    for i in range(tot_genes):
        mut_prob = np.random.rand()
        if mut_prob < mut_rate:
            chromo[i] = float(np.random.uniform(
                low=low, high=high, size=(1, 1)))

    return chromo


# Function to generate the next population using the initial flattend population and fitness function values
def generate_nextpop(pop, fitness, pop_size, tot_genes, mut_rate, cross_rate):
    # Stores the new population
    new_pop = np.ndarray((pop_size, tot_genes))
    cur_size = 0

    # Perform elitism by getting the top 4 indices by sorting the fitness value in descending order. For this we use argsort and get the first 4 individuals. Takes np.array as argument
    indices = (-np.array(fitness)).argsort()[:4]
    for index in indices:
        new_pop[cur_size] = pop[index]
        cur_size = cur_size + 1

    # Perform selection, then crossover generating 26 offsprings on which we will perform mutation with mutation rate = mut_rate and crossover rate = cross_rate
    # run the loop for half the required offsprings
    for i in range(int((pop_size - cur_size) / 2)):
        p1, p2 = roulette_wheel(fitness, pop_size)  # generate 2 parents

        # generate the 2 offsprings
        o1, o2 = get_offsprings(pop[p1], pop[p2], cross_rate)

        new_pop[cur_size] = o1
        cur_size = cur_size + 1

        new_pop[cur_size] = o2
        cur_size = cur_size + 1

    # Perform mutation on the crossover generated offsprings.
    for i in range(4, pop_size):
        new_pop[i] = mutate(new_pop[i], tot_genes, mut_rate)

    return new_pop


# -------------------------------------MAIN FUNCTION--------------------------------------- #
# Load the Dataset
dataset = load_iris()  # load the dataset

X = dataset["data"]  # get the data fields
Y = dataset["target"]  # get the class

# get the population size
pop_size = 30
"""
ARCHITECTURE OF NN ---

Number of input neurons = 4
Number of hidden layers = 1
Number of hidden neurons = 4
Number of output neurons = 3

Total parameters = 4*4 + 4*3 + 4 + 3 = 16 + 12 + 6 = 35
"""
input_neurons = X.shape[1]
hidden_neurons = 4
output_neurons = np.unique(Y).shape[0]
tot_genes = (
    hidden_neurons * input_neurons
    + output_neurons * hidden_neurons
    + hidden_neurons
    + output_neurons
)

params = intial_pop(input_neurons, hidden_neurons, output_neurons, pop_size)

# Convert labels to form of binary form
encoder = pp.LabelBinarizer()
Y = encoder.fit_transform(Y)

# shuffle the input and output classes
X_shuffle, Y_shuffle = utils.shuffle(X, Y)
"""
The number of iterations is 100. The crossover used is one-point crossover and mutation is performed on the offsprings generated from the crossover. Top 4 individuals are passed as it is to the next population and the rest 26 (26 offsprings - 13 crossover - 26 parents) are generated using crossover and mutation. 
The mutation probability = 1/(iterationnum + 10) 
The crossover probability = 0.8 (always perform crossover)
"""
num_iterations = 1000
mut_rate = 0
cross_rate = 0.8
# stores the average loss
loss_avg = [float(x) for x in range(num_iterations)]

# stores the max and min fitness
max_fit = [float(x) for x in range(num_iterations)]
min_fit = [float(x) for x in range(num_iterations)]
avg_fit = [float(x) for x in range(num_iterations)]

# stores the average accuracy and max accuracy
avg_accuracy = [float(x) for x in range(num_iterations)]
max_accuracy = [float(x) for x in range(num_iterations)]

for iter_num in range(num_iterations):
    # Set the mutation rate
    mut_rate = (1 / (iter_num + 10))

    # Get the losses
    losses, avg_accuracy[iter_num], max_accuracy[iter_num] = get_loss(
        params, pop_size, X_shuffle, Y_shuffle)

    # Calculate the fitness of the population
    fitness = get_fitness(losses, pop_size)

    # Flatten the matrices of the weights and biases.
    pop = flatten(params, pop_size, tot_genes)

    # Generate the next population in form of chromosomes (1D)
    newpop = generate_nextpop(pop, fitness, pop_size,
                              tot_genes, mut_rate, cross_rate)

    # Unflatten the new population
    params = unflatten(newpop, pop_size, input_neurons,
                       hidden_neurons, output_neurons)

    loss_avg[iter_num] = np.mean(losses)
    max_fit[iter_num] = np.max(fitness)
    min_fit[iter_num] = np.min(fitness)
    avg_fit[iter_num] = np.mean(fitness)
    if (iter_num + 1) % 10 == 0:
        print("Loss after {} iterations =  {}. Accuracy = {}".format(
            (iter_num + 1), loss_avg[iter_num], avg_accuracy[iter_num]))

# Plot the min, max and avg fitness
plt.figure(1)
(line1,) = plt.plot(max_fit, color="green", label="Maximum Fit")
(line2,) = plt.plot(min_fit, color="blue", label="Minimum Fit")
(line3,) = plt.plot(avg_fit, color="red", label="Average Fit")
plt.legend(handles=[line1, line2, line3])
plt.xlabel("Iteration Number")
plt.ylabel("Fitness Value")

# Plot the losses
plt.figure(2)
line1, = plt.plot(loss_avg, label="Average Loss")
plt.legend(handles=[line1])
plt.xlabel("Iteration Number")
plt.ylabel("Loss")

# Plot the average accuracy
plt.figure(3)
line1, = plt.plot(avg_accuracy, color="green", label="Average Accuracy")
line2, = plt.plot(max_accuracy, color="red", label="Max Accuracy")
plt.legend(handles=[line1, line2])
plt.xlabel("Iteration Number")
plt.ylabel("Accuracy")

# Show the plots
plt.show()
