import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        not_converge = True
        while not_converge:
            not_converge = False
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x,nn.as_scalar(y))
                    not_converge = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.m1 = nn.Parameter(1, 200)
        self.m2 = nn.Parameter(200, 1)
        self.b1 = nn.Parameter(1, 200)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xm1 = nn.Linear(x, self.m1)
        lay1 = nn.ReLU(nn.AddBias(xm1, self.b1))
        return nn.AddBias(nn.Linear(lay1, self.m2), self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(200):
            loss = self.get_loss(x, y)
            if nn.as_scalar(loss) < .02:
                break
            grad_m1, grad_b1, grad_m2, grad_b2 = nn.gradients(loss, [self.m1, self.b1, self.m2, self.b2])
            rate = -.06
            self.m1.update(grad_m1, rate)
            self.m2.update(grad_m2, rate)
            self.b1.update(grad_b1, rate)
            self.b2.update(grad_b2, rate)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.layers = 784
        self.m1 = nn.Parameter(self.layers, 200)
        self.b1 = nn.Parameter(1,200)
        self.m2 = nn.Parameter(200, 150)
        self.b2 = nn.Parameter(1, 150)
        self.m3 = nn.Parameter(150, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xm1 = nn.Linear(x, self.m1)
        bias1 = nn.AddBias(xm1, self.b1)
        bias2 = nn.AddBias(nn.Linear(nn.ReLU(bias1), self.m2), self.b2)
        bias3 = nn.AddBias(nn.Linear(nn.ReLU(bias2), self.m3), self.b3)

        return bias3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        rate = -.05
        while dataset.get_validation_accuracy() < .974:
            if (dataset.get_validation_accuracy() >= .89):
                rate = -.005
            for x,y in dataset.iterate_once(10):
                loss = self.get_loss(x, y)
                grad_m1, grad_b1, grad_m2, grad_b2, grad_m3, grad_b3 = nn.gradients(loss,[self.m1, self.b1, self.m2, self.b2, self.m3, self.b3])
                self.m1.update(grad_m1, rate)
                self.m2.update(grad_m2, rate)
                self.m3.update(grad_m3, rate)
                self.b1.update(grad_b1, rate)
                self.b2.update(grad_b2, rate)
                self.b3.update(grad_b3, rate)

class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = -0.15
        self.numTrainingGames = 4000
        self.batch_size = 50
        self.parameters = [nn.Parameter(state_dim, 800), nn.Parameter(1, 800),
                          nn.Parameter(800, 100), nn.Parameter(1, 100),
                          nn.Parameter(100, action_dim), nn.Parameter(1, action_dim)]


    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"

        Q_predict = self.run(states)
        if not Q_predict:
            return None
        return nn.SquareLoss(Q_predict, Q_target)
        

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"

        y = states
        y = nn.Linear(y, self.parameters[0])
        y = nn.AddBias(y, self.parameters[1])

        for i in range(1, len(self.parameters) // 2):
            y = nn.ReLU(y)
            y = nn.Linear(y, self.parameters[2*i])
            y = nn.AddBias(y, self.parameters[2*i + 1])

        return y

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states,Q_target)
        gradients = nn.gradients(loss, self.parameters)
        for i in range(len(self.parameters)):
            self.parameters[i].update(gradients[i], self.learning_rate)