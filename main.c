#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"

static int xor_converged(Network *net, Matrix *input, float inputs_data[4][2], float targets_data[4][1], float threshold) {
    for (int i = 0; i < 4; i++) {
        input->data[0] = inputs_data[i][0];
        input->data[1] = inputs_data[i][1];

        Matrix *res = network_predict(net, input);
        float output = res->data[0];
        float target = targets_data[i][0];

        if (target == 1.0f) {
            if (output < 1.0f - threshold) return 0;
        } else {
            if (output > threshold) return 0;
        }
    }

    return 1;
}

int main(void) {
    const int max_epochs = 200000;
    const int log_every = 5000;
    const float threshold = 0.3f;
    clock_t train_start;
    clock_t train_end;
    double train_time;
    int converged_epoch = -1;

    // srand(time(NULL));
    srand(42);

    uint32_t sizes[] = {2, 4, 4, 4, 4, 1};
    ActivationType types[] = {RELU, RELU, RELU, RELU, SIGMOID};
    Network *net = network_create(sizes, 6, types);

    float inputs_data[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float targets_data[4][1] = {{0}, {1}, {1}, {0}};

    Matrix *input = matrix_create(2, 1);
    Matrix *target = matrix_create(1, 1);

    printf("Training ... \n");

    train_start = clock();

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        for (int i = 0; i < 4; i++) {
            input->data[0] = inputs_data[i][0];
            input->data[1] = inputs_data[i][1];
            target->data[0] = targets_data[i][0];

            network_predict(net, input);
            network_backward(net, input, target);
            network_update(net, 0.1f);
        }

        if (xor_converged(net, input, inputs_data, targets_data, threshold)) {
            converged_epoch = epoch + 1;
            break;
        }

        if (epoch % log_every == 0) printf("Epoch %d end...\n", epoch);
    }

    train_end = clock();
    train_time = (double)(train_end - train_start) / CLOCKS_PER_SEC;

    printf("\n--- Resultats Finaux ---\n");
    for (int i = 0; i < 4; i++) {
        input->data[0] = inputs_data[i][0];
        input->data[1] = inputs_data[i][1];

        Matrix *res = network_predict(net, input);
        printf("In: %.0f,%.0f -> Out: %f (Attendu: %.0f)\n",
                input->data[0], input->data[1], res->data[0], targets_data[i][0]);
    }

    if (converged_epoch >= 0) {
        printf("\nConvergence atteinte en %d epochs\n", converged_epoch);
    } else {
        printf("\nConvergence non atteinte apres %d epochs\n", max_epochs);
    }
    printf("Training time: %.6f s\n", train_time);

    matrix_free(&input);
    matrix_free(&target);
    network_free(&net);

    return 0;
}
