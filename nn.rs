const EPOCHS: i32 = 1000;
const LEARNING_RATE: f32 = 0.1;

fn sigmoid(x: &f32) -> f32 {
    let e: f32 = 2.718281828459045;
    1.0 / (1.0 + e.powf(-x))
}

fn sigmoid_derivative(x: &f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn dot(weights: &[f32; 2], inputs: &[f32; 2]) -> f32 {
    weights[0] * inputs[0] + weights[1] * inputs[1]
}

fn mse_loss(y_true: &Vec<f32>, y_pred: &Vec<f32>) -> f32 {
    if y_true.len() != y_pred.len() {
        panic!("Why are they not equal???");
    }
    let n = y_true.len();
    let mut i = 0;

    let mut sum: f32 = 0.0;
    while i < n {
        sum += (y_true[i] - y_pred[i]).powi(2);
        i += 1;
    }

    sum / n as f32
}

#[derive(Debug)]
struct Neuron {
    weights: [f32; 2],
    bias: f32,
}

impl Neuron {
    fn feed_forward(&self, inputs: &[f32; 2]) -> f32 {
        let total: f32 = dot(&self.weights, &inputs);
        sigmoid(&total)
    }
}

struct NeuralNetwork {
    h1: Neuron,
    h2: Neuron,
    o1: Neuron,
}

impl NeuralNetwork {
    fn feed_forward(&self, inputs: &[f32; 2]) -> f32 {
        let new_inputs: [f32; 2] = [
            self.h1.feed_forward(&inputs),
            self.h2.feed_forward(&inputs)
        ];

        self.o1.feed_forward(&new_inputs)
    }

    fn train(&mut self, data: &Vec<[f32; 2]>, y_true: &Vec<f32>) {
        if data.len() != y_true.len() {
            panic!("The training and validation data lengths do not match");
        }

        let mut epoch = 0;
        let data_len = data.len();

        while epoch < EPOCHS {
            let mut j = 0;
            while j < data_len {
                // Perform the feedforwards at each neuron
                let sum_h1: f32 = dot(&self.h1.weights, &data[j]) + self.h1.bias;
                let h1: f32 = sigmoid(&sum_h1);

                let sum_h2: f32 = dot(&self.h2.weights, &data[j]) + self.h2.bias;
                let h2: f32 = sigmoid(&sum_h2);

                let new_inputs: [f32; 2] = [h1, h2];
                let sum_o1 = dot(&self.o1.weights, &new_inputs) + self.o1.bias;
                let o1 = sigmoid(&sum_o1);
                let y_pred = o1;

                let d_L_w_y_pred = -2.0 * (y_true[j] - y_pred);

                // Output neuron
                let d_y_pred_w_h1 = self.o1.weights[0] * sigmoid_derivative(&sum_o1);
                let d_y_pred_w_h2 = self.o1.weights[1] * sigmoid_derivative(&sum_o1);

                let d_y_pred_w_w5 = h1 * sigmoid_derivative(&sum_o1);
                let d_y_pred_w_w6 = h2 * sigmoid_derivative(&sum_o1);

                let d_y_pred_w_b3 = sigmoid_derivative(&sum_o1);

                // H1 Neuron
                let d_h1_w_w1 = data[j][0] * sigmoid_derivative(&sum_h1);
                let d_h1_w_w2 = data[j][1] * sigmoid_derivative(&sum_h1);

                let d_h1_w_b = sigmoid_derivative(&sum_h1);

                // H2 Neuron
                let d_h2_w_w1 = data[j][0] * sigmoid_derivative(&sum_h2);
                let d_h2_w_w2 = data[j][1] * sigmoid_derivative(&sum_h2);

                let d_h2_w_b = sigmoid_derivative(&sum_h2);

                // Update the weights and biases
                self.h1.weights[0] -= LEARNING_RATE * d_L_w_y_pred * d_y_pred_w_h1 * d_h1_w_w1;
                self.h1.weights[1] -= LEARNING_RATE * d_L_w_y_pred * d_y_pred_w_h1 * d_h1_w_w2;
                self.h1.bias -= LEARNING_RATE * d_L_w_y_pred * d_y_pred_w_h1 * d_h1_w_b;

                self.h2.weights[0] -= LEARNING_RATE * d_L_w_y_pred * d_y_pred_w_h2 * d_h2_w_w1;
                self.h2.weights[1] -= LEARNING_RATE * d_L_w_y_pred * d_y_pred_w_h2 * d_h2_w_w2;
                self.h2.bias -= LEARNING_RATE * d_L_w_y_pred * d_y_pred_w_h2 * d_h2_w_b;

                self.o1.weights[0] -= LEARNING_RATE * d_L_w_y_pred * d_y_pred_w_w5;
                self.o1.weights[1] -= LEARNING_RATE * d_L_w_y_pred * d_y_pred_w_w6;
                self.o1.bias -= LEARNING_RATE * d_L_w_y_pred * d_y_pred_w_b3;

                j += 1;
            }

            epoch += 1;
        }
    }
}

fn main() {
    let neuron1 = Neuron {
        weights: [0.0, 1.0],
        bias: 0.0,
    };

    let neuron2 = Neuron {
        weights: [0.0, 1.0],
        bias: 0.0,
    };

    let neuron3 = Neuron {
        weights: [0.0, 1.0],
        bias: 0.0,
    };

    let data: Vec<[f32; 2]> = vec![
        [-2.0, -1.0],
        [25.0, 6.0],
        [17.0, 4.0],
        [-15.0, -6.0]
    ];
    let y_trues: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

    let mut neural_network = NeuralNetwork {
        h1: neuron1,
        h2: neuron2,
        o1: neuron3,
    };

    neural_network.train(&data, &y_trues);

    let emily: [f32; 2] = [-7.0, -3.0];
    let frank: [f32; 2] = [20.0, 2.0];

    println!("{}", neural_network.feed_forward(&emily));
    println!("{}", neural_network.feed_forward(&frank));
}
