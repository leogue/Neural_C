#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "network.h"

#define FLOAT_TOL 1e-5f

static int tests_run = 0;

static void fail_test(const char *message, const char *file, int line) {
    fprintf(stderr, "Test failure at %s:%d: %s\n", file, line, message);
    exit(1);
}

#define ASSERT_TRUE(condition, message) \
    do { \
        if (!(condition)) { \
            fail_test(message, __FILE__, __LINE__); \
        } \
    } while (0)

#define ASSERT_INT_EQ(actual, expected, message) \
    do { \
        if ((actual) != (expected)) { \
            fail_test(message, __FILE__, __LINE__); \
        } \
    } while (0)

static void assert_float_close(float actual, float expected, const char *message) {
    if (fabsf(actual - expected) > FLOAT_TOL) {
        fail_test(message, __FILE__, __LINE__);
    }
}

static void test_network_create_initializes_all_layers(void) {
    uint32_t sizes[] = {3, 4, 2};
    ActivationType types[] = {RELU, SIGMOID};
    Network *net = network_create(sizes, 3, types);

    ASSERT_TRUE(net != NULL, "network_create should succeed");
    ASSERT_INT_EQ((int)net->layer_count, 2, "layer_count should be count - 1");
    ASSERT_TRUE(net->layers != NULL, "layers array should be allocated");
    ASSERT_TRUE(net->layers[0] != NULL, "first layer should be allocated");
    ASSERT_TRUE(net->layers[1] != NULL, "second layer should be allocated");

    ASSERT_INT_EQ((int)net->layers[0]->in_size, 3, "first layer in_size should match");
    ASSERT_INT_EQ((int)net->layers[0]->out_size, 4, "first layer out_size should match");
    ASSERT_INT_EQ((int)net->layers[1]->in_size, 4, "second layer in_size should match");
    ASSERT_INT_EQ((int)net->layers[1]->out_size, 2, "second layer out_size should match");

    network_free(&net);
    ASSERT_TRUE(net == NULL, "network_free should null the pointer");
    tests_run++;
}

static void test_network_create_rejects_null_sizes(void) {
    ActivationType types[] = {SIGMOID};
    ASSERT_TRUE(network_create(NULL, 2, types) == NULL, "network_create should reject NULL sizes");
    tests_run++;
}

static void test_network_create_rejects_null_types(void) {
    uint32_t sizes[] = {2, 3};
    ASSERT_TRUE(network_create(sizes, 2, NULL) == NULL, "network_create should reject NULL types");
    tests_run++;
}

static void test_network_create_rejects_count_less_than_two(void) {
    uint32_t sizes[] = {2, 3};
    ActivationType types[] = {SIGMOID};
    ASSERT_TRUE(network_create(sizes, 1, types) == NULL, "network_create should reject count < 2");
    ASSERT_TRUE(network_create(sizes, 0, types) == NULL, "network_create should reject count = 0");
    tests_run++;
}

static void test_network_free_is_null_safe(void) {
    Network *net = NULL;
    network_free(&net);
    network_free(NULL);
    tests_run++;
}

static void test_network_predict_rejects_null_network(void) {
    Matrix *input = matrix_create(2, 1);
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");
    ASSERT_TRUE(network_predict(NULL, input) == NULL, "network_predict should reject NULL network");
    matrix_free(&input);
    tests_run++;
}

static void test_network_predict_rejects_null_input(void) {
    uint32_t sizes[] = {2, 2};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, types);
    ASSERT_TRUE(net != NULL, "network_create should succeed");
    ASSERT_TRUE(network_predict(net, NULL) == NULL, "network_predict should reject NULL input");
    network_free(&net);
    tests_run++;
}

static void test_network_predict_returns_last_layer_activation(void) {
    uint32_t sizes[] = {2, 3, 1};
    ActivationType types[] = {RELU, SIGMOID};
    Network *net = network_create(sizes, 3, types);
    Matrix *input = matrix_create(2, 1);
    Matrix *output;

    ASSERT_TRUE(net != NULL, "network_create should succeed");
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");

    input->data[0] = 1.0f;
    input->data[1] = 2.0f;

    output = network_predict(net, input);
    ASSERT_TRUE(output != NULL, "network_predict should return a matrix");
    ASSERT_TRUE(output == net->layers[1]->A, "network_predict should return last layer activation");

    matrix_free(&input);
    network_free(&net);
    tests_run++;
}

static void test_network_predict_computes_correct_output(void) {
    uint32_t sizes[] = {2, 1};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, types);
    Matrix *input = matrix_create(2, 1);
    Matrix *output;
    float expected_z;
    float expected_a;

    ASSERT_TRUE(net != NULL, "network_create should succeed");
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");

    // Set known weights and biases
    net->layers[0]->W->data[0] = 0.5f;
    net->layers[0]->W->data[1] = -0.5f;
    net->layers[0]->B->data[0] = 0.1f;

    input->data[0] = 1.0f;
    input->data[1] = 2.0f;

    output = network_predict(net, input);
    ASSERT_TRUE(output != NULL, "network_predict should succeed");

    // Z = W * X + B = 0.5*1 + (-0.5)*2 + 0.1 = -0.4
    expected_z = -0.4f;
    expected_a = 1.0f / (1.0f + expf(-expected_z));

    assert_float_close(net->layers[0]->Z->data[0], expected_z, "Z should match W*X+B");
    assert_float_close(output->data[0], expected_a, "output should match sigmoid(Z)");

    matrix_free(&input);
    network_free(&net);
    tests_run++;
}

static void test_network_backward_rejects_null_network(void) {
    Matrix *input = matrix_create(2, 1);
    Matrix *target = matrix_create(2, 1);
    ASSERT_INT_EQ(network_backward(NULL, input, target), -1, "network_backward should reject NULL network");
    matrix_free(&input);
    matrix_free(&target);
    tests_run++;
}

static void test_network_backward_rejects_null_input(void) {
    uint32_t sizes[] = {2, 2};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, types);
    Matrix *target = matrix_create(2, 1);

    ASSERT_TRUE(net != NULL, "network_create should succeed");
    ASSERT_INT_EQ(network_backward(net, NULL, target), -1, "network_backward should reject NULL input");

    matrix_free(&target);
    network_free(&net);
    tests_run++;
}

static void test_network_backward_rejects_null_target(void) {
    uint32_t sizes[] = {2, 2};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, types);
    Matrix *input = matrix_create(2, 1);

    ASSERT_TRUE(net != NULL, "network_create should succeed");
    ASSERT_INT_EQ(network_backward(net, input, NULL), -1, "network_backward should reject NULL target");

    matrix_free(&input);
    network_free(&net);
    tests_run++;
}

static void test_network_backward_computes_output_gradients(void) {
    uint32_t sizes[] = {1, 1};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, types);
    Matrix *input = matrix_create(1, 1);
    Matrix *target = matrix_create(1, 1);
    Layer *layer;

    ASSERT_TRUE(net != NULL, "network_create should succeed");

    layer = net->layers[0];
    layer->W->data[0] = 0.0f;
    layer->B->data[0] = 0.0f;
    input->data[0] = 1.0f;
    target->data[0] = 0.0f;

    network_predict(net, input);
    ASSERT_INT_EQ(network_backward(net, input, target), 0, "network_backward should succeed");

    // With W=0, B=0, input=1: Z=0, A=sigmoid(0)=0.5
    // delta = (A - target) * sigmoid'(A) = 0.5 * 0.25 = 0.125
    assert_float_close(layer->A->data[0], 0.5f, "activation should be 0.5");
    assert_float_close(layer->dZ->data[0], 0.25f, "dZ should be sigmoid'(0.5)");
    assert_float_close(layer->delta->data[0], 0.125f, "delta should be (A-target)*dZ");
    assert_float_close(layer->dW->data[0], 0.125f, "dW should be delta * input");
    assert_float_close(layer->dB->data[0], 0.125f, "dB should be delta");

    matrix_free(&input);
    matrix_free(&target);
    network_free(&net);
    tests_run++;
}

static void test_network_backward_propagates_to_hidden_layer(void) {
    uint32_t sizes[] = {1, 1, 1};
    ActivationType types[] = {TANH, SIGMOID};
    Network *net = network_create(sizes, 3, types);
    Matrix *input = matrix_create(1, 1);
    Matrix *target = matrix_create(1, 1);
    Layer *hidden;
    Layer *output;
    float hidden_a;
    float output_a;
    float output_dz;
    float output_delta;
    float hidden_dz;
    float hidden_delta;

    ASSERT_TRUE(net != NULL, "network_create should succeed");

    hidden = net->layers[0];
    output = net->layers[1];

    hidden->W->data[0] = 1.0f;
    hidden->B->data[0] = 0.0f;
    output->W->data[0] = 1.0f;
    output->B->data[0] = 0.0f;
    input->data[0] = 0.5f;
    target->data[0] = 0.0f;

    network_predict(net, input);
    network_backward(net, input, target);

    hidden_a = tanhf(0.5f);
    output_a = 1.0f / (1.0f + expf(-hidden_a));
    output_dz = output_a * (1.0f - output_a);
    output_delta = output_a * output_dz;
    hidden_dz = 1.0f - (hidden_a * hidden_a);
    hidden_delta = output_delta * hidden_dz;

    assert_float_close(hidden->A->data[0], hidden_a, "hidden activation should match tanh");
    assert_float_close(output->A->data[0], output_a, "output activation should match sigmoid");
    assert_float_close(hidden->dZ->data[0], hidden_dz, "hidden dZ should be tanh'(a)");
    assert_float_close(hidden->delta->data[0], hidden_delta, "hidden delta should propagate correctly");

    matrix_free(&input);
    matrix_free(&target);
    network_free(&net);
    tests_run++;
}

static void test_network_update_rejects_null_network(void) {
    ASSERT_INT_EQ(network_update(NULL, 0.1f), -1, "network_update should reject NULL network");
    tests_run++;
}

static void test_network_update_applies_gradient_descent(void) {
    uint32_t sizes[] = {1, 1};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, types);
    Matrix *input = matrix_create(1, 1);
    Matrix *target = matrix_create(1, 1);
    Layer *layer;
    float initial_w;
    float initial_b;
    float dw;
    float db;
    float lr = 0.1f;

    ASSERT_TRUE(net != NULL, "network_create should succeed");

    layer = net->layers[0];
    layer->W->data[0] = 0.5f;
    layer->B->data[0] = 0.1f;
    initial_w = layer->W->data[0];
    initial_b = layer->B->data[0];

    input->data[0] = 1.0f;
    target->data[0] = 1.0f;

    network_predict(net, input);
    network_backward(net, input, target);

    dw = layer->dW->data[0];
    db = layer->dB->data[0];

    ASSERT_INT_EQ(network_update(net, lr), 0, "network_update should succeed");

    // W = W - lr * dW
    assert_float_close(layer->W->data[0], initial_w - lr * dw, "weight should follow gradient descent");
    assert_float_close(layer->B->data[0], initial_b - lr * db, "bias should follow gradient descent");

    matrix_free(&input);
    matrix_free(&target);
    network_free(&net);
    tests_run++;
}

static void test_network_training_reduces_loss(void) {
    uint32_t sizes[] = {2, 4, 1};
    ActivationType types[] = {RELU, SIGMOID};
    Network *net = network_create(sizes, 3, types);
    Matrix *input = matrix_create(2, 1);
    Matrix *target = matrix_create(1, 1);
    Matrix *output;
    float initial_loss;
    float final_loss;
    float lr = 0.5f;

    ASSERT_TRUE(net != NULL, "network_create should succeed");

    srand(42);
    input->data[0] = 1.0f;
    input->data[1] = 0.0f;
    target->data[0] = 1.0f;

    // Compute initial loss
    output = network_predict(net, input);
    initial_loss = (output->data[0] - target->data[0]) * (output->data[0] - target->data[0]);

    // Train for several iterations
    for (int i = 0; i < 100; ++i) {
        network_predict(net, input);
        network_backward(net, input, target);
        network_update(net, lr);
    }

    // Compute final loss
    output = network_predict(net, input);
    final_loss = (output->data[0] - target->data[0]) * (output->data[0] - target->data[0]);

    ASSERT_TRUE(final_loss < initial_loss, "training should reduce loss");

    matrix_free(&input);
    matrix_free(&target);
    network_free(&net);
    tests_run++;
}

int main(void) {
    // network_create tests
    test_network_create_initializes_all_layers();
    test_network_create_rejects_null_sizes();
    test_network_create_rejects_null_types();
    test_network_create_rejects_count_less_than_two();

    // network_free tests
    test_network_free_is_null_safe();

    // network_predict tests
    test_network_predict_rejects_null_network();
    test_network_predict_rejects_null_input();
    test_network_predict_returns_last_layer_activation();
    test_network_predict_computes_correct_output();

    // network_backward tests
    test_network_backward_rejects_null_network();
    test_network_backward_rejects_null_input();
    test_network_backward_rejects_null_target();
    test_network_backward_computes_output_gradients();
    test_network_backward_propagates_to_hidden_layer();

    // network_update tests
    test_network_update_rejects_null_network();
    test_network_update_applies_gradient_descent();

    // Integration test
    test_network_training_reduces_loss();

    printf("All %d network tests passed.\n", tests_run);
    return 0;
}
