# Neural Network

A small neural network implemented from scratch in both **C** and **Rust**, with no machine learning libraries. It is a 2-2-1 feedforward network trained with backpropagation on a toy binary classification problem.

The goal of this project is to understand the mechanics of a neural network — forward pass, loss, gradients, and weight updates — by writing every step by hand.

## Network architecture

```
inputs (2)          hidden layer (2)        output (1)
 x1  ──────────────►  h1  ────────────────┐
                                          ├──► o1 ──► y_pred
 x2  ──────────────►  h2  ────────────────┘
```

| Component       | Details                                  |
| --------------- | ---------------------------------------- |
| Activation      | Sigmoid                                  |
| Loss            | Mean squared error                       |
| Optimizer       | Stochastic gradient descent              |
| Learning rate   | `0.1`                                    |
| Epochs          | `1000`                                   |
| Initial weights | `[0, 1]` per neuron, bias `0`            |

Each neuron computes `sigmoid(w · x + b)`. Gradients are derived by hand through the sigmoid derivative and the chain rule, then applied per-sample.

## Files

| File           | Description                                     |
| -------------- | ----------------------------------------------- |
| `main.c`       | C implementation                                |
| `nn.rs`        | Rust implementation (no external crates)         |
| `.clang-format`| Formatting config for the C source               |

Both implementations are intentionally kept structurally similar so the two languages can be compared side by side.

## Dataset

The network is trained on four points labeled by a simple rule, with one sample per class repeated:

| x1   | x2  | Label |
| ---- | --- | ----- |
| -2   | -1  | 1     |
| 25   | 6   | 0     |
| 17   | 4   | 0     |
| -15  | -6  | 1     |

It then predicts on two unseen points, `(-7, -3)` and `(20, 2)`.

## Build and run

### C

Requires `gcc` (or any C compiler) and the math library:

```sh
gcc -O2 -o main main.c -lm
./main
```

### Rust

Requires the Rust toolchain:

```sh
rustc -O -o nn nn.rs
./nn
```

### Example output

```
0.967763
0.038712
```

The first value is the prediction for `(-7, -3)` and the second for `(20, 2)`. Values close to `1` and `0` respectively mean the network learned the pattern.

## Notes

This is an educational implementation, not a general-purpose library. It is hardcoded for a 2-2-1 topology, a single output, and a fixed dataset. Natural next steps would be configurable layer sizes, proper weight initialization (for example Xavier/He), and batching instead of per-sample updates.
