//
// Created by Leo Guerin on 12/04/2026.
//

#ifndef NEURAL_C_MATRIX_H
#define NEURAL_C_MATRIX_H
#include <stdint.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    float* data;
} Matrix;

Matrix* matrix_create(uint32_t rows, uint32_t cols);
void matrix_free(Matrix** matrix);
// Matrix* matrix_transpose(Matrix *B);
int matrix_dot(Matrix *A, int ta, Matrix *B, int tb, Matrix *out);
int matrix_hadamard(Matrix *A, Matrix *B, Matrix *out);
int matrix_apply(Matrix *A, float (*sigma)(float) ,Matrix *out);
int matrix_fill(Matrix *m, float value);
int matrix_randomize(Matrix *m);
int matrix_add(Matrix *A, Matrix *B, Matrix *out);
int matrix_sub(Matrix *A, Matrix *B, Matrix *out);
int matrix_add_scaled(Matrix *A, Matrix *B, float scalar, Matrix *out);
void matrix_print(Matrix *m);

#endif //NEURAL_C_MATRIX_H
