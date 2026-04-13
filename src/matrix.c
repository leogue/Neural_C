//
// Created by Leo Guerin on 12/04/2026.
//

#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


Matrix *matrix_create(uint32_t rows, uint32_t cols) {
    if (rows == 0 || cols == 0) return NULL;

    Matrix *M = malloc(sizeof(Matrix));
    if (!M) return NULL;

    M->rows = rows;
    M->cols = cols;

    M->data = malloc(M->rows * M->cols * sizeof(float));
    if (!M->data) {
        free(M);
        return NULL;
    }

    return M;
}

void matrix_free(Matrix **matrix) {
    if (!matrix || !*matrix) return;

    if ((*matrix)->data) {
        free((*matrix)->data);
        (*matrix)->data = NULL;
    }

    free(*matrix);
    *matrix = NULL;
}


// Matrix *matrix_transpose(Matrix *B) {
//     Matrix *T = matrix_create(B->cols, B->rows);
//     if (!T) return NULL;
//
//     for (size_t i = 0; i < B->rows; i++) {
//         for (size_t j = 0; j < B->cols; j++) {
//             T->data[j * T->cols + i] = B->data[i * B->cols + j];
//         }
//     }
//
//     return T;
// }


int matrix_dot(Matrix *A, int ta, Matrix *B, int tb, Matrix *out) {
    if (!A || !B || !out) return -1;

    uint32_t a_rows = ta ? A->cols : A->rows;
    uint32_t a_cols = ta ? A->rows : A->cols;
    uint32_t b_rows = tb ? B->cols : B->rows;
    uint32_t b_cols = tb ? B->rows : B->cols;

    if (a_cols != b_rows || out->rows != a_rows || out->cols != b_cols) {
        return -1;
    }

    for (uint32_t i = 0; i < a_rows; i++) {
        for (uint32_t j = 0; j < b_cols; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < a_cols; k++) {
                float valA = ta ? A->data[k * A->cols + i] : A->data[i * A->cols + k];
                float valB = tb ? B->data[j * B->cols + k] : B->data[k * B->cols + j];
                sum += valA * valB;
            }
            out->data[i * out->cols + j] = sum;
        }
    }
    return 0;
}


int matrix_hadamard(Matrix *A, Matrix *B, Matrix *out) {
    if (!A || !B || !out) return -1;
    if (A->cols != B->cols || A->rows != B->rows) return -2;
    if (out->cols != A->cols || out->rows != A->rows) return -3;

    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; j++) {
            out->data[i * out->cols + j] = A->data[i * out->cols + j] * B->data[i * out->cols + j];
        }
    }

    return 0;
}


int matrix_apply(Matrix *A, float (*sigma)(float), Matrix *out) {
    if (!A || !out) return -1;
    if (A->cols != out->cols || A->rows != out->rows) return -2;

    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            out->data[i * out->cols + j] = sigma(A->data[i * out->cols + j]);
        }
    }

    return 0;
}



int matrix_fill(Matrix *m, float value) {
    if (!m) return -1;

    for (size_t i=0; i<m->cols * m->rows; ++i) {
        m->data[i] = value;
    }

    return 0;
}


int matrix_randomize(Matrix *m) {
    if (!m) return -1;

    for (size_t i=0; i<m->cols * m->rows; ++i) {
        m->data[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f ;
    }

    return 0;
}


int matrix_add(Matrix *A, Matrix *B, Matrix *out) {
    if (!A || !B || !out) return -1;

    // Standard addition for matrices with identical shapes.
    if (A->rows == B->rows && A->cols == B->cols) {
        if (out->rows != A->rows || out->cols != A->cols) return -2;

        for (size_t i = 0; i < A->rows * A->cols; i++) {
            out->data[i] = A->data[i] + B->data[i];
        }
        return 0;
    }

    // Broadcasting case: add a column vector B to every column of A.
    if (A->rows == B->rows && B->cols == 1) {
        if (out->rows != A->rows || out->cols != A->cols) return -3;

        for (uint32_t i = 0; i < A->rows; i++) {
            for (uint32_t j = 0; j < A->cols; j++) {
                out->data[i * A->cols + j] = A->data[i * A->cols + j] + B->data[i];
            }
        }
        return 0;
    }

    return -4; // Incompatible dimensions.
}



int matrix_sub(Matrix *A, Matrix *B, Matrix *out) {
    if (!A || !B || !out) return -1;

    // Standard addition for matrices with identical shapes.
    if (A->rows == B->rows && A->cols == B->cols) {
        if (out->rows != A->rows || out->cols != A->cols) return -2;

        for (size_t i = 0; i < A->rows * A->cols; i++) {
            out->data[i] = A->data[i] - B->data[i];
        }
        return 0;
    }

    // Broadcasting case: add a column vector B to every column of A.
    if (A->rows == B->rows && B->cols == 1) {
        if (out->rows != A->rows || out->cols != A->cols) return -3;

        for (uint32_t i = 0; i < A->rows; i++) {
            for (uint32_t j = 0; j < A->cols; j++) {
                out->data[i * A->cols + j] = A->data[i * A->cols + j] - B->data[i];
            }
        }
        return 0;
    }

    return -4; // Incompatible dimensions.
}


int matrix_add_scaled(Matrix *A, Matrix *B, float scalar, Matrix *out) {
    // A + xB

    if (!A || !B || !out) return -1;
    // TODO : add size check

    for (size_t i=0; i<out->rows; ++i) {
        for (size_t j=0; j<out->cols; ++j) {
            out->data[i * out->cols + j] = A->data[i * out->cols + j] + scalar * B->data[i * out->cols + j];
        }
    }

    return 0;

}

void matrix_print(Matrix *m) {
    if (!m) {
        printf("(null matrix)\n");
        return;
    }

    for (uint32_t i = 0; i < m->rows; i++) {
        printf("[ ");
        for (uint32_t j = 0; j < m->cols; j++) {
            printf("%8.4f ", m->data[i * m->cols + j]);
        }
        printf("]\n");
    }
}
