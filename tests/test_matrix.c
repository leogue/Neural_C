#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "matrix.h"

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

static Matrix *create_matrix_from_array(uint32_t rows, uint32_t cols, const float *values) {
    Matrix *m = matrix_create(rows, cols);
    ASSERT_TRUE(m != NULL, "matrix_create should succeed");

    for (uint32_t i = 0; i < rows * cols; i++) {
        m->data[i] = values[i];
    }

    return m;
}

static void assert_matrix_equals(Matrix *m, const float *expected, uint32_t size, const char *message) {
    ASSERT_TRUE(m != NULL, "matrix must not be NULL");

    for (uint32_t i = 0; i < size; i++) {
        if (fabsf(m->data[i] - expected[i]) > FLOAT_TOL) {
            fail_test(message, __FILE__, __LINE__);
        }
    }
}

static float square_plus_one(float x) {
    return x * x + 1.0f;
}

static void test_matrix_create_and_fill(void) {
    Matrix *m = matrix_create(2, 3);
    ASSERT_TRUE(m != NULL, "matrix_create should allocate the matrix");
    ASSERT_INT_EQ((int)m->rows, 2, "rows should match");
    ASSERT_INT_EQ((int)m->cols, 3, "cols should match");

    ASSERT_INT_EQ(matrix_fill(m, 4.5f), 0, "matrix_fill should succeed");
    for (uint32_t i = 0; i < 6; i++) {
        assert_float_close(m->data[i], 4.5f, "matrix_fill should write every element");
    }

    matrix_free(&m);
    ASSERT_TRUE(m == NULL, "matrix_free should null the pointer");
    tests_run++;
}

static void test_matrix_create_rejects_zero_dimensions(void) {
    Matrix *m1 = matrix_create(0, 3);
    Matrix *m2 = matrix_create(3, 0);

    ASSERT_TRUE(m1 == NULL, "matrix_create should reject zero rows");
    ASSERT_TRUE(m2 == NULL, "matrix_create should reject zero cols");
    tests_run++;
}

static void test_matrix_free_is_null_safe(void) {
    Matrix *m = NULL;
    matrix_free(&m);
    matrix_free(NULL);
    tests_run++;
}

static void test_matrix_dot_basic(void) {
    const float a_values[] = {1, 2, 3, 4, 5, 6};
    const float b_values[] = {7, 8, 9, 10, 11, 12};
    const float expected[] = {58, 64, 139, 154};

    Matrix *A = create_matrix_from_array(2, 3, a_values);
    Matrix *B = create_matrix_from_array(3, 2, b_values);
    Matrix *out = matrix_create(2, 2);

    ASSERT_INT_EQ(matrix_dot(A, 0, B, 0, out), 0, "matrix_dot should succeed");
    assert_matrix_equals(out, expected, 4, "matrix_dot should produce the expected product");

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&out);
    tests_run++;
}

static void test_matrix_dot_with_transpose_flags(void) {
    const float a_values[] = {1, 2, 3, 4, 5, 6};
    const float b_values[] = {7, 8, 9, 10, 11, 12};
    const float expected_at_b[] = {89, 98, 116, 128};
    const float expected_a_bt[] = {23, 29, 35, 53, 67, 81, 83, 105, 127};

    Matrix *A = create_matrix_from_array(3, 2, a_values);
    Matrix *B = create_matrix_from_array(3, 2, b_values);
    Matrix *out_at_b = matrix_create(2, 2);
    Matrix *out_a_bt = matrix_create(3, 3);

    ASSERT_INT_EQ(matrix_dot(A, 1, B, 0, out_at_b), 0, "matrix_dot should support A^T * B");
    ASSERT_INT_EQ(matrix_dot(A, 0, B, 1, out_a_bt), 0, "matrix_dot should support A * B^T");

    assert_matrix_equals(out_at_b, expected_at_b, 4, "matrix_dot transpose on A is incorrect");
    assert_matrix_equals(out_a_bt, expected_a_bt, 9, "matrix_dot transpose on B is incorrect");

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&out_at_b);
    matrix_free(&out_a_bt);
    tests_run++;
}

static void test_matrix_dot_rejects_invalid_shapes(void) {
    Matrix *A = matrix_create(2, 3);
    Matrix *B = matrix_create(4, 2);
    Matrix *out = matrix_create(2, 2);

    ASSERT_INT_EQ(matrix_dot(A, 0, B, 0, out), -1, "matrix_dot should reject incompatible shapes");
    ASSERT_INT_EQ(matrix_dot(NULL, 0, B, 0, out), -1, "matrix_dot should reject NULL matrices");

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&out);
    tests_run++;
}

static void test_matrix_hadamard(void) {
    const float a_values[] = {1, 2, 3, 4};
    const float b_values[] = {5, 6, 7, 8};
    const float expected[] = {5, 12, 21, 32};

    Matrix *A = create_matrix_from_array(2, 2, a_values);
    Matrix *B = create_matrix_from_array(2, 2, b_values);
    Matrix *out = matrix_create(2, 2);

    ASSERT_INT_EQ(matrix_hadamard(A, B, out), 0, "matrix_hadamard should succeed");
    assert_matrix_equals(out, expected, 4, "matrix_hadamard should multiply element-wise");

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&out);
    tests_run++;
}

static void test_matrix_hadamard_rejects_invalid_shapes(void) {
    Matrix *A = matrix_create(2, 2);
    Matrix *B = matrix_create(2, 3);
    Matrix *out = matrix_create(2, 2);

    ASSERT_INT_EQ(matrix_hadamard(A, B, out), -2, "matrix_hadamard should reject incompatible inputs");

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&out);
    tests_run++;
}

static void test_matrix_apply(void) {
    const float values[] = {1, 2, 3, 4};
    const float expected[] = {2, 5, 10, 17};

    Matrix *A = create_matrix_from_array(2, 2, values);
    Matrix *out = matrix_create(2, 2);

    ASSERT_INT_EQ(matrix_apply(A, square_plus_one, out), 0, "matrix_apply should succeed");
    assert_matrix_equals(out, expected, 4, "matrix_apply should transform every element");

    matrix_free(&A);
    matrix_free(&out);
    tests_run++;
}

static void test_matrix_apply_rejects_invalid_shapes(void) {
    Matrix *A = matrix_create(2, 2);
    Matrix *out = matrix_create(3, 2);

    ASSERT_INT_EQ(matrix_apply(A, square_plus_one, out), -2, "matrix_apply should reject incompatible shapes");
    ASSERT_INT_EQ(matrix_apply(NULL, square_plus_one, out), -1, "matrix_apply should reject NULL input");

    matrix_free(&A);
    matrix_free(&out);
    tests_run++;
}

static void test_matrix_randomize_range(void) {
    Matrix *m = matrix_create(10, 10);
    ASSERT_TRUE(m != NULL, "matrix_create should allocate the matrix");

    srand(1234);
    ASSERT_INT_EQ(matrix_randomize(m), 0, "matrix_randomize should succeed");

    for (uint32_t i = 0; i < 100; i++) {
        ASSERT_TRUE(m->data[i] >= -1.0f && m->data[i] <= 1.0f, "matrix_randomize should stay within [-1, 1]");
    }

    ASSERT_INT_EQ(matrix_randomize(NULL), -1, "matrix_randomize should reject NULL");

    matrix_free(&m);
    tests_run++;
}

static void test_matrix_add_standard(void) {
    const float a_values[] = {1, 2, 3, 4};
    const float b_values[] = {5, 6, 7, 8};
    const float expected[] = {6, 8, 10, 12};

    Matrix *A = create_matrix_from_array(2, 2, a_values);
    Matrix *B = create_matrix_from_array(2, 2, b_values);
    Matrix *out = matrix_create(2, 2);

    ASSERT_INT_EQ(matrix_add(A, B, out), 0, "matrix_add should support standard addition");
    assert_matrix_equals(out, expected, 4, "matrix_add standard addition is incorrect");

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&out);
    tests_run++;
}

static void test_matrix_add_broadcast_column_vector(void) {
    const float a_values[] = {1, 2, 3, 4, 5, 6};
    const float b_values[] = {10, 20};
    const float expected[] = {11, 12, 13, 24, 25, 26};

    Matrix *A = create_matrix_from_array(2, 3, a_values);
    Matrix *B = create_matrix_from_array(2, 1, b_values);
    Matrix *out = matrix_create(2, 3);

    ASSERT_INT_EQ(matrix_add(A, B, out), 0, "matrix_add should support column broadcasting");
    assert_matrix_equals(out, expected, 6, "matrix_add broadcast result is incorrect");

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&out);
    tests_run++;
}

static void test_matrix_add_rejects_invalid_shapes(void) {
    Matrix *A = matrix_create(2, 3);
    Matrix *B = matrix_create(3, 1);
    Matrix *out = matrix_create(2, 3);

    ASSERT_INT_EQ(matrix_add(A, B, out), -4, "matrix_add should reject unsupported broadcasting");
    ASSERT_INT_EQ(matrix_add(NULL, B, out), -1, "matrix_add should reject NULL input");

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&out);
    tests_run++;
}

static void test_matrix_print_output(void) {
    const float values[] = {1.0f, 2.5f, -3.0f, 4.25f};
    char buffer[256];
    FILE *capture = tmpfile();
    int saved_stdout;
    Matrix *A;

    ASSERT_TRUE(capture != NULL, "tmpfile should succeed");

    saved_stdout = dup(fileno(stdout));
    ASSERT_TRUE(saved_stdout >= 0, "dup should succeed");

    fflush(stdout);
    ASSERT_TRUE(dup2(fileno(capture), fileno(stdout)) >= 0, "dup2 should redirect stdout");

    A = create_matrix_from_array(2, 2, values);
    matrix_print(A);

    fflush(stdout);
    ASSERT_TRUE(dup2(saved_stdout, fileno(stdout)) >= 0, "stdout should be restored");
    close(saved_stdout);

    rewind(capture);
    memset(buffer, 0, sizeof(buffer));
    ASSERT_TRUE(fgets(buffer, sizeof(buffer), capture) != NULL, "matrix_print should write at least one line");
    ASSERT_TRUE(strstr(buffer, "[") != NULL, "matrix_print output should contain brackets");

    fclose(capture);
    matrix_free(&A);
    tests_run++;
}

static void test_matrix_print_null(void) {
    matrix_print(NULL);
    tests_run++;
}

int main(void) {
    test_matrix_create_and_fill();
    test_matrix_create_rejects_zero_dimensions();
    test_matrix_free_is_null_safe();
    test_matrix_dot_basic();
    test_matrix_dot_with_transpose_flags();
    test_matrix_dot_rejects_invalid_shapes();
    test_matrix_hadamard();
    test_matrix_hadamard_rejects_invalid_shapes();
    test_matrix_apply();
    test_matrix_apply_rejects_invalid_shapes();
    test_matrix_randomize_range();
    test_matrix_add_standard();
    test_matrix_add_broadcast_column_vector();
    test_matrix_add_rejects_invalid_shapes();
    test_matrix_print_output();
    test_matrix_print_null();

    printf("All %d matrix tests passed.\n", tests_run);
    return 0;
}
