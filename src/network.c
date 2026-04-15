//
// Created by Leo Guerin on 13/04/2026.
//

#include "network.h"

#include <math.h>
#include <stdlib.h>


// MSE: 0.5 * (a - y)^2
static float mse_forward(float a, float y) {
    return 0.5f * (a - y) * (a - y);
}
static float mse_derivative(float a, float y) {
    return a - y;
}

// Log Loss: -[y*ln(a) + (1-y)*ln(1-a)]
static float log_loss_forward(float a, float y) {
    return -(y * logf(a + 1e-7f) + (1.0f - y) * logf(1.0f - a + 1e-7f));
}
static float log_loss_derivative(float a, float y) {
    return (a - y) / ((a * (1.0f - a)) + 1e-7f);
}

LossPair get_loss(LossType type) {
    switch (type) {
        case LOG_LOSS:
            return (LossPair){log_loss_forward, log_loss_derivative};
        case MSE:
        default:
            return (LossPair){mse_forward, mse_derivative};
    }
}


Network *network_create(uint32_t *sizes, uint32_t count, uint32_t batch_size, ActivationType *types) {
    if (!sizes || !types || count < 2 || batch_size == 0) return NULL;

    Network *net = malloc(sizeof(Network));
    if (!net) return NULL;

    net->layer_count = count - 1;
    net->batch_size = batch_size;
    net->layers = malloc(net->layer_count * sizeof(Layer *));
    if (!net->layers) {
        free(net);
        return NULL;
    }

    for (uint32_t i = 0; i < net->layer_count; ++i) {
        net->layers[i] = layer_create(sizes[i], sizes[i + 1],net->batch_size, types[i]);
        if (!net->layers[i]) {
            for (uint32_t j = 0; j < i; ++j) {
                layer_free(&net->layers[j]);
            }
            free(net->layers);
            free(net);
            return NULL;
        }
    }

    net->loss_type = MSE;

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

    LossPair loss = get_loss(net->loss_type);
    Layer *output = net->layers[net->layer_count - 1];
    if (input->rows != net->layers[0]->in_size || input->cols != net->batch_size) return -1;
    if (target->rows != output->out_size || target->cols != net->batch_size) return -1;
    if (output->A->rows != output->out_size || output->A->cols != net->batch_size) return -1;

    // Output layer gradient: delta = (A - target) * activation'(A)
    for (uint32_t k = 0; k < output->delta->rows * output->delta->cols; ++k) {
        output->delta->data[k] =
            loss.loss_derivative(output->A->data[k], target->data[k]);
    }
    if (matrix_apply(output->A, output->activation_prime, output->dZ) != 0) return -1;
    if (matrix_hadamard(output->delta, output->dZ, output->delta) != 0) return -1;

    // Backpropagate through all layers
    for (int i = (int)net->layer_count - 1; i >= 0; --i) {
        Layer *layer = net->layers[i];

        // dB = mean over batch of delta
        if (matrix_sum_columns(layer->delta, layer->dB) != 0) return -1;
        if (matrix_scale(layer->dB, 1.0f / (float)layer->batch_size) != 0) return -1;

        // dW = mean over batch of delta * prev_A^T
        Matrix *prev_activation = (i == 0) ? input : net->layers[i - 1]->A;
        if (matrix_dot(layer->delta, 0, prev_activation, 1, layer->dW) != 0) return -1;
        if (matrix_scale(layer->dW, 1.0f / (float)layer->batch_size) != 0) return -1;

        // Propagate error to previous layer
        if (i > 0) {
            Layer *prev = net->layers[i - 1];
            if (matrix_dot(layer->W, 1, layer->delta, 0, prev->delta) != 0) return -1;
            if (matrix_apply(prev->A, prev->activation_prime, prev->dZ) != 0) return -1;
            if (matrix_hadamard(prev->delta, prev->dZ, prev->delta) != 0) return -1;
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
