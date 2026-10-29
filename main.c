#include <math.h>
#include <stdio.h>

#define DATA_SIZE 4

const double LEARNING_RATE = 0.1;
const double EPOCHS = 1000;

typedef struct {
  double weights[2];
  double bias;
} Neuron;

typedef struct {
  Neuron h1; // Hidden layer 1
  Neuron h2; // Hidden layer 2
  Neuron o1; // Output layer
} NeuralNetwork;

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double sigmoidDerivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

double dot(const double weights[2], const double inputs[2]) { return weights[0] * inputs[0] + weights[1] * inputs[1]; }

double neuronFeedforward(Neuron *neuron, double inputs[2]) {
  double total = dot(neuron->weights, inputs) + neuron->bias;
  return sigmoid(total);
}

double networkFeedforward(NeuralNetwork *network, double inputs[2]) {
  double h1Output = neuronFeedforward(&network->h1, inputs);
  double h2Output = neuronFeedforward(&network->h2, inputs);

  inputs[0] = h1Output;
  inputs[1] = h2Output;
  double output = neuronFeedforward(&network->o1, inputs);
  return output;
}

void resetNetwork(NeuralNetwork *network) {
  network->h1.bias = 0;
  network->h2.bias = 0;
  network->o1.bias = 0;

  network->h1.weights[0] = 0;
  network->h2.weights[0] = 0;
  network->o1.weights[0] = 0;

  network->h1.weights[1] = 1;
  network->h2.weights[1] = 1;
  network->o1.weights[1] = 1;
}

double mseLoss(double *yTrue, double *yPred, int n) {
  double sum = 0;
  for (int i = 0; i < n; ++i)
    sum += pow(yTrue[i] - yPred[i], 2);

  return sum / n;
}

void train(NeuralNetwork *network, const double data[][2], const double yTrue[DATA_SIZE]) {
  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    for (int j = 0; j < DATA_SIZE; ++j) {
      double sumH1 = dot(network->h1.weights, data[j]) + network->h1.bias;
      double h1 = sigmoid(sumH1);

      double sumH2 = dot(network->h2.weights, data[j]) + network->h2.bias;
      double h2 = sigmoid(sumH2);

      double newInputs[2] = {h1, h2};
      double sumO1 = dot(network->o1.weights, newInputs) + network->o1.bias;
      double o1 = sigmoid(sumO1);
      double yPred = o1;

      double d_L_w_yPred = -2 * (yTrue[j] - yPred);

      // Output neuron
      double d_yPred_w_h1 = network->o1.weights[0] * sigmoidDerivative(sumO1);
      double d_yPred_w_h2 = network->o1.weights[1] * sigmoidDerivative(sumO1);

      double d_yPred_w_w5 = h1 * sigmoidDerivative(sumO1);
      double d_yPred_w_w6 = h2 * sigmoidDerivative(sumO1);

      double d_yPred_w_b3 = sigmoidDerivative(sumO1);

      // H1 Neuron
      double d_h1_w_w1 = data[j][0] * sigmoidDerivative(sumH1);
      double d_h1_w_w2 = data[j][1] * sigmoidDerivative(sumH1);

      double d_h1_w_b = sigmoidDerivative(sumH1);

      // H2 Neuron
      double d_h2_w_w1 = data[j][0] * sigmoidDerivative(sumH2);
      double d_h2_w_w2 = data[j][1] * sigmoidDerivative(sumH2);

      double d_h2_w_b = sigmoidDerivative(sumH2);

      network->h1.weights[0] -= LEARNING_RATE * d_L_w_yPred * d_yPred_w_h1 * d_h1_w_w1; // w1
      network->h1.weights[1] -= LEARNING_RATE * d_L_w_yPred * d_yPred_w_h1 * d_h1_w_w2; // w2
      network->h1.bias -= LEARNING_RATE * d_L_w_yPred * d_yPred_w_h1 * d_h1_w_b;        // b1

      network->h2.weights[0] -= LEARNING_RATE * d_L_w_yPred * d_yPred_w_h2 * d_h2_w_w1; // w3
      network->h2.weights[1] -= LEARNING_RATE * d_L_w_yPred * d_yPred_w_h2 * d_h2_w_w2; // w4
      network->h2.bias -= LEARNING_RATE * d_L_w_yPred * d_yPred_w_h2 * d_h2_w_b;        // b2

      network->o1.weights[0] -= LEARNING_RATE * d_L_w_yPred * d_yPred_w_w5; // w5
      network->o1.weights[1] -= LEARNING_RATE * d_L_w_yPred * d_yPred_w_w6; // w6
      network->o1.bias -= LEARNING_RATE * d_L_w_yPred * d_yPred_w_b3;       // w6
    }
  }
}

int main() {
  const double data[DATA_SIZE][2] = {
      {-2, -1},
      {25, 6},
      {17, 4},
      {-15, -6},
  };

  const double yTrues[DATA_SIZE] = {1, 0, 0, 1};

  NeuralNetwork network;
  resetNetwork(&network);

  train(&network, data, yTrues);

  double emily[2] = {-7, -3};
  double frank[2] = {20, 2};
  printf("%f\n", networkFeedforward(&network, emily));
  printf("%f\n", networkFeedforward(&network, frank));

  return 0;
}
