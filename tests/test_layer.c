#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
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

static void test_get_activation_supports_all_declared_types(void) {
    ActivationPair sigmoid = get_activation(SIGMOID);
    ActivationPair relu = get_activation(RELU);
    ActivationPair tanh_pair = get_activation(TANH);

    ASSERT_TRUE(sigmoid.forward != NULL, "SIGMOID forward must exist");
    ASSERT_TRUE(sigmoid.derivative != NULL, "SIGMOID derivative must exist");
    ASSERT_TRUE(relu.forward != NULL, "RELU forward must exist");
    ASSERT_TRUE(relu.derivative != NULL, "RELU derivative must exist");
    ASSERT_TRUE(tanh_pair.forward != NULL, "TANH forward must exist");
    ASSERT_TRUE(tanh_pair.derivative != NULL, "TANH derivative must exist");

    assert_float_close(sigmoid.forward(0.0f), 0.5f, "SIGMOID forward is incorrect");
    assert_float_close(relu.forward(-2.0f), 0.0f, "RELU forward is incorrect");
    assert_float_close(tanh_pair.forward(0.0f), 0.0f, "TANH forward is incorrect");
    assert_float_close(tanh_pair.derivative(0.0f), 1.0f, "TANH derivative is incorrect");
    tests_run++;
}

static void test_activation_derivatives_match_expected_values(void) {
    ActivationPair sigmoid = get_activation(SIGMOID);
    ActivationPair relu = get_activation(RELU);
    ActivationPair tanh_pair = get_activation(TANH);

    assert_float_close(sigmoid.derivative(0.5f), 0.25f, "SIGMOID derivative is incorrect");
    assert_float_close(relu.derivative(3.0f), 1.0f, "RELU derivative on positive input is incorrect");
    assert_float_close(relu.derivative(0.0f), 0.0f, "RELU derivative on zero input is incorrect");
    assert_float_close(tanh_pair.derivative(0.5f), 0.75f, "TANH derivative is incorrect");
    tests_run++;
}

static void test_layer_create_initializes_all_fields(void) {
    Layer *layer;

    srand(1234);
    layer = layer_create(3, 2, 1, TANH);

    ASSERT_TRUE(layer != NULL, "layer_create should allocate a layer");
    ASSERT_INT_EQ((int)layer->in_size, 3, "in_size should match");
    ASSERT_INT_EQ((int)layer->out_size, 2, "out_size should match");
    ASSERT_TRUE(layer->W != NULL, "weights must be allocated");
    ASSERT_TRUE(layer->B != NULL, "biases must be allocated");
    ASSERT_TRUE(layer->Z != NULL, "pre-activation buffer must be allocated");
    ASSERT_TRUE(layer->A != NULL, "activation buffer must be allocated");
    ASSERT_TRUE(layer->activation != NULL, "activation function must be set");
    ASSERT_TRUE(layer->activation_prime != NULL, "activation derivative must be set");

    ASSERT_INT_EQ((int)layer->W->rows, 2, "weight rows should match out_size");
    ASSERT_INT_EQ((int)layer->W->cols, 3, "weight cols should match in_size");
    ASSERT_INT_EQ((int)layer->B->rows, 2, "bias rows should match out_size");
    ASSERT_INT_EQ((int)layer->B->cols, 1, "bias cols should be one");
    ASSERT_INT_EQ((int)layer->Z->rows, 2, "Z rows should match out_size");
    ASSERT_INT_EQ((int)layer->Z->cols, 1, "Z cols should be one");
    ASSERT_INT_EQ((int)layer->A->rows, 2, "A rows should match out_size");
    ASSERT_INT_EQ((int)layer->A->cols, 1, "A cols should be one");
    assert_float_close(layer->Z->data[0], 0.0f, "Z should start at zero");
    assert_float_close(layer->A->data[0], 0.0f, "A should start at zero");

    layer_free(&layer);
    ASSERT_TRUE(layer == NULL, "layer_free should null the pointer");
    tests_run++;
}

static void test_layer_create_rejects_zero_dimensions(void) {
    ASSERT_TRUE(layer_create(0, 2, 1, SIGMOID) == NULL, "layer_create should reject zero input size");
    ASSERT_TRUE(layer_create(2, 0, 1, SIGMOID) == NULL, "layer_create should reject zero output size");
    ASSERT_TRUE(layer_create(2, 2, 0, SIGMOID) == NULL, "layer_create should reject zero batch size");
    tests_run++;
}

static void test_layer_forward_computes_expected_sigmoid_output(void) {
    Layer *layer = layer_create(2, 2, 1, SIGMOID);
    Matrix *input = matrix_create(2, 1);

    ASSERT_TRUE(layer != NULL, "layer_create should succeed");
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");

    layer->W->data[0] = 1.0f;
    layer->W->data[1] = 2.0f;
    layer->W->data[2] = -1.0f;
    layer->W->data[3] = 0.5f;
    layer->B->data[0] = 0.5f;
    layer->B->data[1] = -0.5f;
    input->data[0] = 3.0f;
    input->data[1] = -1.0f;

    ASSERT_INT_EQ(layer_forward(layer, input), 0, "layer_forward should succeed");
    assert_float_close(layer->Z->data[0], 1.5f, "Z[0] should match W.X+B");
    assert_float_close(layer->Z->data[1], -4.0f, "Z[1] should match W.X+B");
    assert_float_close(layer->A->data[0], 1.0f / (1.0f + expf(-1.5f)), "A[0] should match sigmoid(Z[0])");
    assert_float_close(layer->A->data[1], 1.0f / (1.0f + expf(4.0f)), "A[1] should match sigmoid(Z[1])");

    matrix_free(&input);
    layer_free(&layer);
    tests_run++;
}

static void test_layer_forward_computes_expected_relu_output(void) {
    Layer *layer = layer_create(2, 2, 1, RELU);
    Matrix *input = matrix_create(2, 1);

    ASSERT_TRUE(layer != NULL, "layer_create should succeed");
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");

    layer->W->data[0] = 1.0f;
    layer->W->data[1] = -2.0f;
    layer->W->data[2] = 0.5f;
    layer->W->data[3] = 1.0f;
    layer->B->data[0] = -0.5f;
    layer->B->data[1] = 1.0f;
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;

    ASSERT_INT_EQ(layer_forward(layer, input), 0, "layer_forward should succeed with RELU");
    assert_float_close(layer->Z->data[0], -3.5f, "RELU Z[0] should match W.X+B");
    assert_float_close(layer->Z->data[1], 3.5f, "RELU Z[1] should match W.X+B");
    assert_float_close(layer->A->data[0], 0.0f, "RELU A[0] should clamp negatives to zero");
    assert_float_close(layer->A->data[1], 3.5f, "RELU A[1] should keep positives");

    matrix_free(&input);
    layer_free(&layer);
    tests_run++;
}

static void test_layer_forward_computes_expected_tanh_output(void) {
    Layer *layer = layer_create(2, 1, 1, TANH);
    Matrix *input = matrix_create(2, 1);
    float expected_z = 0.75f;

    ASSERT_TRUE(layer != NULL, "layer_create should succeed");
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");

    layer->W->data[0] = 0.5f;
    layer->W->data[1] = -1.0f;
    layer->B->data[0] = 0.25f;
    input->data[0] = 1.5f;
    input->data[1] = 0.25f;

    ASSERT_INT_EQ(layer_forward(layer, input), 0, "layer_forward should succeed with TANH");
    assert_float_close(layer->Z->data[0], expected_z, "TANH Z should match W.X+B");
    assert_float_close(layer->A->data[0], tanhf(expected_z), "TANH A should match tanh(Z)");

    matrix_free(&input);
    layer_free(&layer);
    tests_run++;
}

static void test_layer_forward_rejects_invalid_input_shape(void) {
    Layer *layer = layer_create(3, 2, 1, RELU);
    Matrix *bad_input = matrix_create(2, 1);

    ASSERT_TRUE(layer != NULL, "layer_create should succeed");
    ASSERT_TRUE(bad_input != NULL, "matrix_create should succeed");
    ASSERT_INT_EQ(layer_forward(layer, bad_input), -1, "layer_forward should reject incompatible input shapes");

    matrix_free(&bad_input);
    layer_free(&layer);
    tests_run++;
}

static void test_layer_forward_rejects_invalid_input_columns(void) {
    Layer *layer = layer_create(2, 2, 1, SIGMOID);
    Matrix *bad_input = matrix_create(2, 2);

    ASSERT_TRUE(layer != NULL, "layer_create should succeed");
    ASSERT_TRUE(bad_input != NULL, "matrix_create should succeed");
    ASSERT_INT_EQ(layer_forward(layer, bad_input), -1, "layer_forward should reject non-column-vector inputs");

    matrix_free(&bad_input);
    layer_free(&layer);
    tests_run++;
}

static void test_layer_forward_rejects_null_arguments(void) {
    Layer *layer = layer_create(2, 2, 1, SIGMOID);
    Matrix *input = matrix_create(2, 1);

    ASSERT_TRUE(layer != NULL, "layer_create should succeed");
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");
    ASSERT_INT_EQ(layer_forward(NULL, input), -1, "layer_forward should reject NULL layer");
    ASSERT_INT_EQ(layer_forward(layer, NULL), -1, "layer_forward should reject NULL input");

    matrix_free(&input);
    layer_free(&layer);
    tests_run++;
}

static void test_layer_free_is_null_safe(void) {
    Layer *layer = NULL;

    layer_free(&layer);
    layer_free(NULL);
    tests_run++;
}

static void test_network_backward_uses_activation_output_for_sigmoid_derivative(void) {
    uint32_t sizes[] = {1, 1};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, 1, types);
    Matrix *input = matrix_create(1, 1);
    Matrix *target = matrix_create(1, 1);
    Layer *out_l;

    ASSERT_TRUE(net != NULL, "network_create should succeed");
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");
    ASSERT_TRUE(target != NULL, "matrix_create should succeed");

    out_l = net->layers[0];
    out_l->W->data[0] = 0.0f;
    out_l->B->data[0] = 0.0f;
    input->data[0] = 1.0f;
    target->data[0] = 0.0f;

    ASSERT_TRUE(network_predict(net, input) == out_l->A, "network_predict should return the last activation");
    network_backward(net, input, target);

    assert_float_close(out_l->A->data[0], 0.5f, "sigmoid output should be 0.5");
    assert_float_close(out_l->dZ->data[0], 0.25f, "sigmoid derivative should be computed from activation output");
    assert_float_close(out_l->delta->data[0], 0.125f, "output delta should use sigmoid'(a)");
    assert_float_close(out_l->dW->data[0], 0.125f, "weight gradient should match delta * input");
    assert_float_close(out_l->dB->data[0], 0.125f, "bias gradient should match delta");

    matrix_free(&input);
    matrix_free(&target);
    network_free(&net);
    tests_run++;
}

static void test_network_backward_uses_activation_output_for_hidden_tanh_derivative(void) {
    uint32_t sizes[] = {1, 1, 1};
    ActivationType types[] = {TANH, SIGMOID};
    Network *net = network_create(sizes, 3, 1, types);
    Matrix *input = matrix_create(1, 1);
    Matrix *target = matrix_create(1, 1);
    Layer *hidden_l;
    Layer *out_l;
    float hidden_a;
    float output_a;
    float expected_out_dz;
    float expected_out_delta;
    float expected_hidden_backprop;
    float expected_hidden_dz;
    float expected_hidden_delta;

    ASSERT_TRUE(net != NULL, "network_create should succeed");
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");
    ASSERT_TRUE(target != NULL, "matrix_create should succeed");

    hidden_l = net->layers[0];
    out_l = net->layers[1];

    hidden_l->W->data[0] = 1.0f;
    hidden_l->B->data[0] = 0.0f;
    out_l->W->data[0] = 1.0f;
    out_l->B->data[0] = 0.0f;
    input->data[0] = 0.5f;
    target->data[0] = 0.0f;

    network_predict(net, input);
    network_backward(net, input, target);

    hidden_a = tanhf(0.5f);
    output_a = 1.0f / (1.0f + expf(-hidden_a));
    expected_out_dz = output_a * (1.0f - output_a);
    expected_out_delta = output_a * expected_out_dz;
    expected_hidden_backprop = expected_out_delta;
    expected_hidden_dz = 1.0f - (hidden_a * hidden_a);
    expected_hidden_delta = expected_hidden_backprop * expected_hidden_dz;

    assert_float_close(hidden_l->A->data[0], hidden_a, "hidden activation should match tanh");
    assert_float_close(out_l->A->data[0], output_a, "output activation should match sigmoid");
    assert_float_close(hidden_l->dZ->data[0], expected_hidden_dz, "hidden tanh derivative should be computed from activation output");
    assert_float_close(hidden_l->delta->data[0], expected_hidden_delta, "hidden delta should use tanh'(a)");

    matrix_free(&input);
    matrix_free(&target);
    network_free(&net);
    tests_run++;
}

static void test_network_create_initializes_all_layers(void) {
    uint32_t sizes[] = {3, 4, 2};
    ActivationType types[] = {RELU, SIGMOID};
    Network *net = network_create(sizes, 3, 1, types);

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

static void test_network_create_rejects_invalid_parameters(void) {
    uint32_t sizes[] = {2, 3};
    ActivationType types[] = {SIGMOID};

    ASSERT_TRUE(network_create(NULL, 2, 1, types) == NULL, "network_create should reject NULL sizes");
    ASSERT_TRUE(network_create(sizes, 2, 1, NULL) == NULL, "network_create should reject NULL types");
    ASSERT_TRUE(network_create(sizes, 1, 1, types) == NULL, "network_create should reject count < 2");
    ASSERT_TRUE(network_create(sizes, 0, 1, types) == NULL, "network_create should reject count = 0");
    ASSERT_TRUE(network_create(sizes, 2, 0, types) == NULL, "network_create should reject zero batch size");
    tests_run++;
}

static void test_network_free_is_null_safe(void) {
    Network *net = NULL;
    network_free(&net);
    network_free(NULL);
    tests_run++;
}

static void test_network_predict_rejects_null_arguments(void) {
    uint32_t sizes[] = {2, 2};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, 1, types);
    Matrix *input = matrix_create(2, 1);

    ASSERT_TRUE(net != NULL, "network_create should succeed");
    ASSERT_TRUE(input != NULL, "matrix_create should succeed");

    ASSERT_TRUE(network_predict(NULL, input) == NULL, "network_predict should reject NULL network");
    ASSERT_TRUE(network_predict(net, NULL) == NULL, "network_predict should reject NULL input");

    matrix_free(&input);
    network_free(&net);
    tests_run++;
}

static void test_network_predict_returns_last_layer_activation(void) {
    uint32_t sizes[] = {2, 3, 1};
    ActivationType types[] = {RELU, SIGMOID};
    Network *net = network_create(sizes, 3, 1, types);
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

static void test_network_backward_rejects_null_arguments(void) {
    uint32_t sizes[] = {2, 2};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, 1, types);
    Matrix *input = matrix_create(2, 1);
    Matrix *target = matrix_create(2, 1);

    ASSERT_TRUE(net != NULL, "network_create should succeed");

    ASSERT_INT_EQ(network_backward(NULL, input, target), -1, "network_backward should reject NULL network");
    ASSERT_INT_EQ(network_backward(net, NULL, target), -1, "network_backward should reject NULL input");
    ASSERT_INT_EQ(network_backward(net, input, NULL), -1, "network_backward should reject NULL target");

    matrix_free(&input);
    matrix_free(&target);
    network_free(&net);
    tests_run++;
}

static void test_network_update_applies_gradient_descent(void) {
    uint32_t sizes[] = {1, 1};
    ActivationType types[] = {SIGMOID};
    Network *net = network_create(sizes, 2, 1, types);
    Matrix *input = matrix_create(1, 1);
    Matrix *target = matrix_create(1, 1);
    float initial_w;
    float initial_b;
    float lr = 0.1f;

    ASSERT_TRUE(net != NULL, "network_create should succeed");

    Layer *layer = net->layers[0];
    layer->W->data[0] = 0.5f;
    layer->B->data[0] = 0.1f;
    initial_w = layer->W->data[0];
    initial_b = layer->B->data[0];

    input->data[0] = 1.0f;
    target->data[0] = 1.0f;

    network_predict(net, input);
    network_backward(net, input, target);
    ASSERT_INT_EQ(network_update(net, lr), 0, "network_update should succeed");

    // W and B should have changed
    ASSERT_TRUE(fabsf(layer->W->data[0] - initial_w) > FLOAT_TOL, "weights should be updated");
    ASSERT_TRUE(fabsf(layer->B->data[0] - initial_b) > FLOAT_TOL, "biases should be updated");

    // Verify gradient descent formula: W = W - lr * dW
    float expected_w = initial_w - lr * layer->dW->data[0];
    float expected_b = initial_b - lr * layer->dB->data[0];
    assert_float_close(layer->W->data[0], expected_w, "weight update should follow gradient descent");
    assert_float_close(layer->B->data[0], expected_b, "bias update should follow gradient descent");

    matrix_free(&input);
    matrix_free(&target);
    network_free(&net);
    tests_run++;
}

static void test_network_update_rejects_null_arguments(void) {
    ASSERT_INT_EQ(network_update(NULL, 0.1f), -1, "network_update should reject NULL network");
    tests_run++;
}

int main(void) {
    // Layer tests
    test_get_activation_supports_all_declared_types();
    test_activation_derivatives_match_expected_values();
    test_layer_create_initializes_all_fields();
    test_layer_create_rejects_zero_dimensions();
    test_layer_forward_computes_expected_sigmoid_output();
    test_layer_forward_computes_expected_relu_output();
    test_layer_forward_computes_expected_tanh_output();
    test_layer_forward_rejects_invalid_input_shape();
    test_layer_forward_rejects_invalid_input_columns();
    test_layer_forward_rejects_null_arguments();
    test_layer_free_is_null_safe();

    // Network tests
    test_network_create_initializes_all_layers();
    test_network_create_rejects_invalid_parameters();
    test_network_free_is_null_safe();
    test_network_predict_rejects_null_arguments();
    test_network_predict_returns_last_layer_activation();
    test_network_backward_rejects_null_arguments();
    test_network_backward_uses_activation_output_for_sigmoid_derivative();
    test_network_backward_uses_activation_output_for_hidden_tanh_derivative();
    test_network_update_applies_gradient_descent();
    test_network_update_rejects_null_arguments();

    printf("All %d layer/network tests passed.\n", tests_run);
    return 0;
}
