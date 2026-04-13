//
// Created by Leo Guerin on 12/04/2026.
//

#ifndef NEURAL_C_MATRIX_H
#define NEURAL_C_MATRIX_H

#include <stdint.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    float *data;
} Matrix;

Matrix *matrix_create(uint32_t rows, uint32_t cols);
void matrix_free(Matrix **m);

int matrix_dot(Matrix *a, int transpose_a, Matrix *b, int transpose_b, Matrix *out);
int matrix_hadamard(Matrix *a, Matrix *b, Matrix *out);
int matrix_add(Matrix *a, Matrix *b, Matrix *out);
int matrix_sub(Matrix *a, Matrix *b, Matrix *out);
int matrix_add_scaled(Matrix *a, Matrix *b, float scalar, Matrix *out);

int matrix_apply(Matrix *m, float (*fn)(float), Matrix *out);
int matrix_fill(Matrix *m, float value);
int matrix_randomize(Matrix *m);

void matrix_print(Matrix *m);

#endif //NEURAL_C_MATRIX_H
