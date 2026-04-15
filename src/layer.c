//
// Created by Leo Guerin on 12/04/2026.
//

#include "layer.h"

#include <math.h>
#include <stdlib.h>

// Activation functions

static float sigmoid_forward(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float sigmoid_derivative(float a) {
    return a * (1.0f - a);
}

static float relu_forward(float x) {
    return x > 0.0f ? x : 0.0f;
}

static float relu_derivative(float a) {
    return a > 0.0f ? 1.0f : 0.0f;
}

static float tanh_forward(float x) {
    return tanhf(x);
}

static float tanh_derivative(float a) {
    return 1.0f - (a * a);
}

ActivationPair get_activation(ActivationType type) {
    switch (type) {
        case RELU:
            return (ActivationPair){relu_forward, relu_derivative};
        case TANH:
            return (ActivationPair){tanh_forward, tanh_derivative};
        case SIGMOID:
        default:
            return (ActivationPair){sigmoid_forward, sigmoid_derivative};
    }
}

Layer *layer_create(uint32_t in_size, uint32_t out_size, uint32_t batch_size, ActivationType type) {
    if (in_size == 0 || out_size == 0 || batch_size == 0) return NULL;

    Layer *layer = malloc(sizeof(Layer));
    if (!layer) return NULL;

    *layer = (Layer){0};
    layer->in_size = in_size;
    layer->out_size = out_size;
    layer->batch_size = batch_size;

    // Weights and biases
    layer->W = matrix_create(out_size, in_size);
    if (!layer->W) { layer_free(&layer); return NULL; }
    if (matrix_randomize(layer->W) != 0) { layer_free(&layer); return NULL; }

    layer->B = matrix_create(out_size, 1);
    if (!layer->B) { layer_free(&layer); return NULL; }
    if (matrix_randomize(layer->B) != 0) { layer_free(&layer); return NULL; }

    // Forward pass storage
    layer->Z = matrix_create(out_size, layer->batch_size);
    if (!layer->Z) { layer_free(&layer); return NULL; }
    if (matrix_fill(layer->Z, 0.0f) != 0) { layer_free(&layer); return NULL; }

    layer->A = matrix_create(out_size, layer->batch_size);
    if (!layer->A) { layer_free(&layer); return NULL; }
    if (matrix_fill(layer->A, 0.0f) != 0) { layer_free(&layer); return NULL; }

    // Gradients
    layer->dW = matrix_create(out_size, in_size);
    if (!layer->dW) { layer_free(&layer); return NULL; }

    layer->dB = matrix_create(out_size, 1);
    if (!layer->dB) { layer_free(&layer); return NULL; }

    layer->dZ = matrix_create(out_size, layer->batch_size);
    if (!layer->dZ) { layer_free(&layer); return NULL; }

    layer->delta = matrix_create(out_size, layer->batch_size);
    if (!layer->delta) { layer_free(&layer); return NULL; }

    // Activation functions
    ActivationPair pair = get_activation(type);
    layer->activation = pair.forward;
    layer->activation_prime = pair.derivative;

    return layer;
}

void layer_free(Layer **layer) {
    if (!layer || !*layer) return;

    matrix_free(&(*layer)->W);
    matrix_free(&(*layer)->B);
    matrix_free(&(*layer)->Z);
    matrix_free(&(*layer)->A);
    matrix_free(&(*layer)->dW);
    matrix_free(&(*layer)->dB);
    matrix_free(&(*layer)->dZ);
    matrix_free(&(*layer)->delta);

    free(*layer);
    *layer = NULL;
}

int layer_forward(Layer *layer, Matrix *input) {
    if (!layer || !input) return -1;
    if (!layer->W || !layer->B || !layer->Z || !layer->A || !layer->activation) return -1;
    if (input->rows != layer->in_size || input->cols != layer->batch_size) return -1;

    // Z = W * input + B
    if (matrix_dot(layer->W, 0, input, 0, layer->Z) != 0) return -2;
    if (matrix_add(layer->Z, layer->B, layer->Z) != 0) return -3;

    // A = activation(Z)
    if (matrix_apply(layer->Z, layer->activation, layer->A) != 0) return -4;

    return 0;
}
