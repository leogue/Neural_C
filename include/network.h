//
// Created by Leo Guerin on 12/04/2026.
//

#ifndef NEURAL_C_NETWORK_H
#define NEURAL_C_NETWORK_H

#include <stdint.h>
#include "layer.h"

typedef enum {
    MSE,
    LOG_LOSS
} LossType;

typedef struct {
    float (*loss_func)(float, float);
    float (*loss_derivative)(float, float);
} LossPair;

typedef struct {
    Layer **layers;
    uint32_t layer_count;
    LossType loss_type;
} Network;

LossPair get_loss(LossType type);

Network *network_create(uint32_t *sizes, uint32_t count, ActivationType *types);
void network_free(Network **net);

Matrix *network_predict(Network *net, Matrix *input);
int network_backward(Network *net, Matrix *input, Matrix *target);
int network_update(Network *net, float lr);

#endif //NEURAL_C_NETWORK_H
