//
// Created by Leo Guerin on 12/04/2026.
//

#include <math.h>
#include <stdlib.h>

#include "matrix.h"
#include "layer.h"

float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
float sigmoid_d(float a) { return a * (1.0f - a); }

float relu_f(float x) { return x > 0 ? x : 0; }
float relu_d(float a) { return a > 0 ? 1 : 0; }

float tanh_f(float x) { return tanhf(x); }
float tanh_d(float a) { return 1.0f - (a * a); }

ActivationPair get_activation(ActivationType type) {
    switch (type) {
        case RELU: return (ActivationPair){relu_f, relu_d};
        case TANH: return (ActivationPair){tanh_f, tanh_d};
        case SIGMOID:
        default: return (ActivationPair){sigmoid_f, sigmoid_d};
    }
}


Layer* layer_create(uint32_t in_size, uint32_t out_size, ActivationType activation_type) {
    Layer* layer = malloc(sizeof(Layer));
    ActivationPair pair;

    if (in_size == 0 || out_size == 0) return NULL;
    if (!layer) return NULL;

    *layer = (Layer){0};
    layer->in_size = in_size;
    layer->out_size = out_size;

    layer->W = matrix_create(layer->out_size, layer->in_size);
    if (!layer->W) {
        layer_free(&layer);
        return NULL;
    }
    if (matrix_randomize(layer->W) != 0) {
        layer_free(&layer);
        return NULL;
    }

    layer->B = matrix_create(layer->out_size, 1);
    if (!layer->B) {
        layer_free(&layer);
        return NULL;
    }
    if (matrix_randomize(layer->B) != 0) {
        layer_free(&layer);
        return NULL;
    }

    layer->Z = matrix_create(layer->out_size, 1);
    if (!layer->Z) {
        layer_free(&layer);
        return NULL;
    }
    if (matrix_fill(layer->Z, 0.0f) != 0) {
        layer_free(&layer);
        return NULL;
    }

    layer->A = matrix_create(layer->out_size, 1);
    if (!layer->A) {
        layer_free(&layer);
        return NULL;
    }
    if (matrix_fill(layer->A, 0.0f) != 0) {
        layer_free(&layer);
        return NULL;
    }

    layer->dW = matrix_create(layer->out_size, layer->in_size);
    if (!layer->dW) {
        layer_free(&layer);
        return NULL;
    }

    layer->dB = matrix_create(layer->out_size, 1);
    if (!layer->dB) {
        layer_free(&layer);
        return NULL;
    }

    layer->dZ= matrix_create(layer->out_size, 1);
    if (!layer->dZ) {
        layer_free(&layer);
        return NULL;
    }

    layer->delta = matrix_create(layer->out_size, 1);
    if (!layer->delta) {
        layer_free(&layer);
        return NULL;
    }

    pair = get_activation(activation_type);
    layer->activation = pair.forward;
    layer->activation_prime = pair.derivative;

    return layer;
}

void layer_free(Layer **l) {
    if (!l || !*l) return;
    matrix_free(&(*l)->W);
    matrix_free(&(*l)->B);
    matrix_free(&(*l)->Z);
    matrix_free(&(*l)->A);
    matrix_free(&(*l)->dW);
    matrix_free(&(*l)->dB);
    matrix_free(&(*l)->dZ);
    matrix_free(&(*l)->delta);
    free(*l);
    *l = NULL;
}


int layer_forward(Layer *l, Matrix *input) {
    if (!l || !input || !l->W || !l->B || !l->Z || !l->A || !l->activation) return -1;
    if (input->rows != l->in_size || input->cols != 1) return -1;
    if (l->Z->rows != l->out_size || l->Z->cols != 1) return -1;
    if (l->A->rows != l->out_size || l->A->cols != 1) return -1;

    matrix_dot(l->W, 0, input, 0, l->Z);
    matrix_add(l->Z, l->B, l->Z);
    matrix_apply(l->Z, l->activation, l->A);

    return 0;
}
