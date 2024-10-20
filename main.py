from math import floor
import numpy as np
import multiprocessing
from MultiLayerFFNN import MLFFNN


def train_nn(args):
    input_size, hidden_layers, hidden_sizes, output_size, features, labels, num_epochs, learning_rate, features_test, labels_test, idx = args
    nn = MLFFNN(input_size, hidden_layers, hidden_sizes, output_size, features, labels, num_epochs, learning_rate)
    nn.train(features_test, labels_test, idx)
    return idx, nn
def read_data(file):
    with open(file) as f:
        data = [line.split() for line in f.readlines()[1:]]
    labels = [int(row[0]) for row in data]  # Ensure labels are integers
    features = [row[1:] for row in data]
    return labels, features

if __name__ == "__main__":
    np.random.seed(7246325)
    labels, features = read_data("L30fft1000.out")
    _, features2 = read_data("L30fft16.out")
    _, bigFeatures = read_data("L30fft1000.out")
    #_, features3 = read_data("L30fft64.out")
    for i in range(len(features)):
        bigFeatures[i].extend(features2[i])
        #features[i].extend(features3[i])

    #shuffle data
    shuffled_indices = np.random.permutation(len(labels))
    labels = np.array(labels, dtype=int)[shuffled_indices]
    features = np.array(features, dtype=float)[shuffled_indices]
    bigFeatures = np.array(bigFeatures, dtype=float)[shuffled_indices]
    test_size = (1/3)
    train_size = int(len(labels) * (1 - test_size))
    labels, labels_test = labels[:train_size], labels[train_size:]
    features, features_test = features[:train_size], features[train_size:]
    bigFeatures, bigFeatures_test = bigFeatures[:train_size], bigFeatures[train_size:]
    input_size = features.shape[1]
    hidden_size = floor(input_size*1)  # Increase hidden layer size
    output_size = 2
    num_epochs = 1500
    num = 20
    learning_rates = np.linspace(0.00025, 0.0025, num=num)
    NNs = []
    processes = []
    train_args = []
    #2000 epochs, cross entropy vs MSE, both architectures
    #Add comparison for 1000 + 16
    for i, lr in enumerate(learning_rates):
        # Architecture 1: Single hidden layer
        train_args.append((input_size, 1, [hidden_size], output_size, features, labels, num_epochs, lr, features_test,
                           labels_test, i))

        # Architecture 2: Three hidden layers
        #train_args.append((input_size, 3, [floor(input_size * 0.2), floor(input_size * 0.3), floor(input_size * 0.2)],output_size, features, labels, num_epochs, lr, features_test, labels_test, i + 10))

    batch_size = 2
    neural_networks = []
    for i in range(0, len(train_args), batch_size):
        with multiprocessing.Pool(processes=(batch_size)) as pool:
            results = pool.map(train_nn, train_args[i:i+batch_size])
            results.sort(key=lambda x: x[0])
        neural_networks.extend([nn for _, nn in results])
    # with multiprocessing.Pool(processes=(num)) as pool:
    #     results = pool.map(train_nn, train_args)
    #     results.sort(key=lambda x: x[0])
    #     neural_networks = [nn for _, nn in results]
    for idx, nn in enumerate(neural_networks):
        labels_one_hot = np.eye(len(nn.output_neurons))[nn.labels]
        correct = 0
        incorrect = 0
        for i in range(len(features)):
            input_data = features[i]
            y_true = labels_one_hot[i]
            y_pred, _ = nn.forward_pass(input_data)
            if np.argmax(y_pred) == labels[i]:
                correct += 1
            else:
                incorrect += 1
        correct_test = 0
        incorrect_test = 0
        labels_one_hot = np.eye(len(nn.output_neurons))[labels_test]
        for i in range(len(features_test)):
            input_data = features_test[i]
            y_true = labels_one_hot[i]
            y_pred, _ = nn.forward_pass(input_data)
            if np.argmax(y_pred) == labels_test[i]:
                correct_test += 1
            else:
                incorrect_test += 1
        print(f"Correct: {correct}, Incorrect: {incorrect}")
        print(f"Correct Test: {correct_test}, Incorrect Test: {incorrect_test}")
        # nn = MLFFNN(input_size, 1, [hidden_size], output_size, features, labels, num_epochs, learning_rate)
        # labels_one_hot = np.eye(len(nn.output_neurons))[labels_test]
        # nn.train(features_test, labels_one_hot)
    #nn.exportNetwork("network.pkl")
    #nn = MLFFNN(savedNetwork="network.pkl") Predicted: [0.93450774 0.06563348], True: [1. 0.] WORST PREDICTION
    # labels_one_hot = np.eye(len(nn.output_neurons))[nn.labels]
    def prediction(prediction):
        return np.argmax(prediction)
    # correct = 0
    # incorrect = 0
    # for i in range(len(features)):
    #     input_data = features[i]
    #     y_true = labels_one_hot[i]
    #     y_pred, _ = nn.forward_pass(input_data)
    #     print(f"Predicted: {y_pred}, True: {y_true}")
    #     if prediction(y_pred) == labels[i]:
    #         correct += 1
    #     else:
    #         incorrect += 1
    # print(f"Correct: {correct}, Incorrect: {incorrect}")


    # for i in range(len(features_test)):
    #     input_data = features_test[i]
    #     y_true = labels_one_hot[i]
    #     y_pred, _ = nn.forward_pass(input_data)
    #     print(f"Predicted: {y_pred}, True: {y_true}")
    #     if prediction(y_pred) == labels_test[i]:
    #         correct += 1
    #     else:
    #         incorrect += 1
    # print(f"Correct: {correct}, Incorrect: {incorrect}")