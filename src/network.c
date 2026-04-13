//
// Created by Leo Guerin on 13/04/2026.
//

#include <stdlib.h>
#include "network.h"


Network *network_create(uint32_t *sizes, uint32_t count, ActivationType *types) {
    Network *n = malloc(sizeof(Network));
    if (!n) return NULL;

    n->layer_count = count - 1;

    n->layers = malloc(n->layer_count * sizeof(Layer *));
    if (!n->layers) {
        free(n);
        return NULL;
    }

    for (size_t i = 0; i < n->layer_count; ++i) {
        n->layers[i] = layer_create(sizes[i], sizes[i + 1], types[i]);
    }


    return n;
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
    Matrix *current_input = input;

    for (size_t i = 0; i < net->layer_count; ++i) {
        layer_forward(net->layers[i], current_input);
        current_input = net->layers[i]->A;
    }

    return current_input;
}


void network_backward(Network *net, Matrix *input, Matrix *target) {
    Layer *out_l = net->layers[net->layer_count - 1];

    matrix_sub(out_l->A, target, out_l->delta);
    matrix_apply(out_l->A, out_l->activation_prime, out_l->dZ);
    matrix_hadamard(out_l->delta, out_l->dZ, out_l->delta);


    for (int i = net->layer_count - 1; i >= 0; --i) {
        Layer *l = net->layers[i];

        matrix_fill(l->dB, 0);
        matrix_add(l->dB, l->delta, l->dB);

        Matrix *prev_A = (i == 0) ? input : net->layers[i - 1]->A;
        matrix_dot(l->delta, 0, prev_A, 1, l->dW);

        if (i > 0) {
            Layer *prev_l = net->layers[i - 1];
            matrix_dot(l->W, 1, l->delta, 0, prev_l->delta);
            matrix_apply(prev_l->A, prev_l->activation_prime, prev_l->dZ);
            matrix_hadamard(prev_l->delta, prev_l->dZ, prev_l->delta);
        }
    }
}


void network_update(Network *net, float lr) {
    for (size_t i = 0; i < net->layer_count; ++i) {
        Layer *l = net->layers[i];

        matrix_add_scaled(l->W, l->dW, -lr, l->W);
        matrix_add_scaled(l->B, l->dB, -lr, l->B);
    }
}
