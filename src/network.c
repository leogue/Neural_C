//
// Created by Leo Guerin on 13/04/2026.
//

#include "network.h"

#include <stdlib.h>

Network *network_create(uint32_t *sizes, uint32_t count, ActivationType *types) {
    if (!sizes || !types || count < 2) return NULL;

    Network *net = malloc(sizeof(Network));
    if (!net) return NULL;

    net->layer_count = count - 1;
    net->layers = malloc(net->layer_count * sizeof(Layer *));
    if (!net->layers) {
        free(net);
        return NULL;
    }

    for (uint32_t i = 0; i < net->layer_count; ++i) {
        net->layers[i] = layer_create(sizes[i], sizes[i + 1], types[i]);
        if (!net->layers[i]) {
            for (uint32_t j = 0; j < i; ++j) {
                layer_free(&net->layers[j]);
            }
            free(net->layers);
            free(net);
            return NULL;
        }
    }

    return net;
}

void network_free(Network **net) {
    if (!net || !*net) return;

    for (uint32_t i = 0; i < (*net)->layer_count; ++i) {
        layer_free(&(*net)->layers[i]);
    }

    free((*net)->layers);
    free(*net);
    *net = NULL;
}

Matrix *network_predict(Network *net, Matrix *input) {
    if (!net || !input || !net->layers) return NULL;

    Matrix *current = input;
    for (uint32_t i = 0; i < net->layer_count; ++i) {
        if (layer_forward(net->layers[i], current) != 0) return NULL;
        current = net->layers[i]->A;
    }

    return current;
}

int network_backward(Network *net, Matrix *input, Matrix *target) {
    if (!net || !input || !target || !net->layers || net->layer_count == 0) return -1;

    // Output layer gradient: delta = (A - target) * activation'(A)
    Layer *output = net->layers[net->layer_count - 1];
    matrix_sub(output->A, target, output->delta);
    matrix_apply(output->A, output->activation_prime, output->dZ);
    matrix_hadamard(output->delta, output->dZ, output->delta);

    // Backpropagate through all layers
    for (int i = (int)net->layer_count - 1; i >= 0; --i) {
        Layer *layer = net->layers[i];

        // dB = delta
        matrix_fill(layer->dB, 0.0f);
        matrix_add(layer->dB, layer->delta, layer->dB);

        // dW = delta * prev_A^T
        Matrix *prev_activation = (i == 0) ? input : net->layers[i - 1]->A;
        matrix_dot(layer->delta, 0, prev_activation, 1, layer->dW);

        // Propagate error to previous layer
        if (i > 0) {
            Layer *prev = net->layers[i - 1];
            matrix_dot(layer->W, 1, layer->delta, 0, prev->delta);
            matrix_apply(prev->A, prev->activation_prime, prev->dZ);
            matrix_hadamard(prev->delta, prev->dZ, prev->delta);
        }
    }

    return 0;
}

int network_update(Network *net, float lr) {
    if (!net || !net->layers) return -1;

    // W = W - lr * dW
    // B = B - lr * dB
    for (uint32_t i = 0; i < net->layer_count; ++i) {
        Layer *layer = net->layers[i];
        matrix_add_scaled(layer->W, layer->dW, -lr, layer->W);
        matrix_add_scaled(layer->B, layer->dB, -lr, layer->B);
    }

    return 0;
}
