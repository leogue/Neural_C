//
// Created by Leo Guerin on 12/04/2026.
//

#ifndef NEURAL_C_LAYER_H
#define NEURAL_C_LAYER_H

#include <stdint.h>
#include "matrix.h"

typedef enum {
    SIGMOID,
    RELU,
    TANH
} ActivationType;

typedef struct {
    float (*forward)(float);
    float (*derivative)(float);
} ActivationPair;

typedef struct {
    uint32_t in_size;   // input size
    uint32_t out_size;  // output size (number of neurons)

    Matrix *W;      // weights
    Matrix *B;      // biases
    Matrix *Z;      // pre-activation: Z = W * X + B
    Matrix *A;      // activation: A = f(Z)

    Matrix *dW;     // weight gradients
    Matrix *dB;     // bias gradients
    Matrix *dZ;     // activation derivative
    Matrix *delta;  // error term for backpropagation

    float (*activation)(float);
    float (*activation_prime)(float);
} Layer;

Layer *layer_create(uint32_t in_size, uint32_t out_size, ActivationType type);
void layer_free(Layer **layer);
int layer_forward(Layer *layer, Matrix *input);

ActivationPair get_activation(ActivationType type);

#endif //NEURAL_C_LAYER_H
