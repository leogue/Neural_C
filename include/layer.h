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
    uint32_t in_size;  // input size
    uint32_t out_size; // neurone number

    Matrix* W;   // weight matrix
    Matrix* B;   // bias matrix
    Matrix* Z;   // Z = W.X+B
    Matrix* A;   // activation matrix

    Matrix* dW;
    Matrix* dB;
    Matrix* dZ;
    Matrix* delta;

    float (*activation)(float);
    float (*activation_prime)(float);

} Layer;

Layer* layer_create(uint32_t in_size, uint32_t out_size, ActivationType activation_type);
int layer_forward(Layer *l, Matrix *input);
void layer_free(Layer **l);

ActivationPair get_activation(ActivationType type);

#endif //NEURAL_C_LAYER_H
