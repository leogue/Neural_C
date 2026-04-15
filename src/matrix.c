//
// Created by Leo Guerin on 12/04/2026.
//

#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Matrix *matrix_create(uint32_t rows, uint32_t cols) {
    if (rows == 0 || cols == 0) return NULL;

    Matrix *m = malloc(sizeof(Matrix));
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * cols * sizeof(float));
    if (!m->data) {
        free(m);
        return NULL;
    }

    return m;
}

void matrix_free(Matrix **m) {
    if (!m || !*m) return;

    if ((*m)->data) {
        free((*m)->data);
        (*m)->data = NULL;
    }

    free(*m);
    *m = NULL;
}

int matrix_dot(Matrix *a, int transpose_a, Matrix *b, int transpose_b, Matrix *out) {
    if (!a || !b || !out) return -1;

    uint32_t a_rows = transpose_a ? a->cols : a->rows;
    uint32_t a_cols = transpose_a ? a->rows : a->cols;
    uint32_t b_rows = transpose_b ? b->cols : b->rows;
    uint32_t b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows || out->rows != a_rows || out->cols != b_cols) {
        return -1;
    }

    for (uint32_t i = 0; i < a_rows; ++i) {
        for (uint32_t j = 0; j < b_cols; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < a_cols; ++k) {
                float val_a = transpose_a ? a->data[k * a->cols + i] : a->data[i * a->cols + k];
                float val_b = transpose_b ? b->data[j * b->cols + k] : b->data[k * b->cols + j];
                sum += val_a * val_b;
            }
            out->data[i * out->cols + j] = sum;
        }
    }

    return 0;
}

int matrix_hadamard(Matrix *a, Matrix *b, Matrix *out) {
    if (!a || !b || !out) return -1;
    if (a->rows != b->rows || a->cols != b->cols) return -2;
    if (out->rows != a->rows || out->cols != a->cols) return -3;

    for (uint32_t i = 0; i < out->rows; ++i) {
        for (uint32_t j = 0; j < out->cols; ++j) {
            out->data[i * out->cols + j] = a->data[i * out->cols + j] * b->data[i * out->cols + j];
        }
    }

    return 0;
}

int matrix_apply(Matrix *m, float (*fn)(float), Matrix *out) {
    if (!m || !out || !fn) return -1;
    if (m->rows != out->rows || m->cols != out->cols) return -2;

    for (uint32_t i = 0; i < out->rows; ++i) {
        for (uint32_t j = 0; j < out->cols; ++j) {
            out->data[i * out->cols + j] = fn(m->data[i * out->cols + j]);
        }
    }

    return 0;
}

int matrix_fill(Matrix *m, float value) {
    if (!m) return -1;

    for (uint32_t i = 0; i < m->rows * m->cols; ++i) {
        m->data[i] = value;
    }

    return 0;
}

int matrix_randomize(Matrix *m) {
    if (!m) return -1;

    for (uint32_t i = 0; i < m->rows * m->cols; ++i) {
        m->data[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
    }

    return 0;
}

int matrix_add(Matrix *a, Matrix *b, Matrix *out) {
    if (!a || !b || !out) return -1;

    // Standard addition for matrices with identical shapes
    if (a->rows == b->rows && a->cols == b->cols) {
        if (out->rows != a->rows || out->cols != a->cols) return -2;

        for (uint32_t i = 0; i < a->rows * a->cols; ++i) {
            out->data[i] = a->data[i] + b->data[i];
        }
        return 0;
    }

    // Broadcasting: add column vector b to each column of a
    if (a->rows == b->rows && b->cols == 1) {
        if (out->rows != a->rows || out->cols != a->cols) return -3;

        for (uint32_t i = 0; i < a->rows; ++i) {
            for (uint32_t j = 0; j < a->cols; ++j) {
                out->data[i * a->cols + j] = a->data[i * a->cols + j] + b->data[i];
            }
        }
        return 0;
    }

    return -4; // Incompatible dimensions
}

int matrix_sub(Matrix *a, Matrix *b, Matrix *out) {
    if (!a || !b || !out) return -1;

    // Standard subtraction for matrices with identical shapes
    if (a->rows == b->rows && a->cols == b->cols) {
        if (out->rows != a->rows || out->cols != a->cols) return -2;

        for (uint32_t i = 0; i < a->rows * a->cols; ++i) {
            out->data[i] = a->data[i] - b->data[i];
        }
        return 0;
    }

    // Broadcasting: subtract column vector b from each column of a
    if (a->rows == b->rows && b->cols == 1) {
        if (out->rows != a->rows || out->cols != a->cols) return -3;

        for (uint32_t i = 0; i < a->rows; ++i) {
            for (uint32_t j = 0; j < a->cols; ++j) {
                out->data[i * a->cols + j] = a->data[i * a->cols + j] - b->data[i];
            }
        }
        return 0;
    }

    return -4; // Incompatible dimensions
}

int matrix_add_scaled(Matrix *a, Matrix *b, float scalar, Matrix *out) {
    if (!a || !b || !out) return -1;
    if (a->rows != b->rows || a->cols != b->cols) return -2;
    if (out->rows != a->rows || out->cols != a->cols) return -3;

    // out = a + scalar * b
    for (uint32_t i = 0; i < out->rows; ++i) {
        for (uint32_t j = 0; j < out->cols; ++j) {
            out->data[i * out->cols + j] = a->data[i * out->cols + j] + scalar * b->data[i * out->cols + j];
        }
    }

    return 0;
}

int matrix_scale(Matrix *a, float s) {
    if (!a ) return -1;

    for (int32_t i = 0; i < a->rows; ++i) {
        for (int32_t j = 0; j < a->cols; ++j) {
            a->data[i * a->cols + j] *= s;
        }
    }

    return 0;
}

int matrix_sum_columns(Matrix *a, Matrix *out) {
    if (!a || !out) return -1;
    if (out->rows != a->rows || out->cols != 1) return -2;

    for (uint32_t i = 0; i < a->rows; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < a->cols; ++j) {
            sum += a->data[i * a->cols + j];
        }
        out->data[i] = sum;
    }

    return 0;
}

void matrix_print(Matrix *m) {
    if (!m) {
        printf("(null matrix)\n");
        return;
    }

    for (uint32_t i = 0; i < m->rows; ++i) {
        printf("[ ");
        for (uint32_t j = 0; j < m->cols; ++j) {
            printf("%8.4f ", m->data[i * m->cols + j]);
        }
        printf("]\n");
    }
}
