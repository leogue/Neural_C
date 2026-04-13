//
// Created by Leo Guerin on 12/04/2026.
//

#ifndef NEURAL_C_NETWORK_H
#define NEURAL_C_NETWORK_H

#include <stdint.h>
#include "layer.h"

typedef struct {
    Layer** layers;
    uint32_t layer_count;
} Network;

Network* network_create(uint32_t *sizes, uint32_t count, ActivationType *types);
Matrix* network_predict(Network *net, Matrix *input);
void network_free(Network **net);
void network_backward(Network *net, Matrix *input, Matrix *target);
void network_update(Network *net, float lr);

#endif //NEURAL_C_NETWORK_H
