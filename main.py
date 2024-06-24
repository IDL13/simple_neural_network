import neural_network as nn
import numpy as np

def main():
    data = np.array([
        [-2, -1],
        [25, 6],
        [17, 4],
        [-15, -6],
    ])

    all_y_trues = np.array([
        1,
        0, 
        0, 
        1,
    ])

    network = nn.NeuralNetwork("sigmoid")
    network.train(data, all_y_trues)

    olga = np.array([-7, -3])
    oleg = np.array([20, 2])

    print("Olga: %.3f" % network.feedforward(olga))
    print("Oleg: %.3f" % network.feedforward(oleg))

if __name__ == "__main__":
    main()