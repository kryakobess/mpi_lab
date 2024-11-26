#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int rows = 4;
int cols = 6;

void matrix_vector_multiply(double *local_matrix, double *vector, double *partial_result, int local_rows, int local_cols, const int result_index_offset) {
    for (int i = 0; i < local_rows; i++) {
        partial_result[i] = 0.0;
        for (int j = 0; j < local_cols; j++) {
            partial_result[i + result_index_offset * local_rows] += local_matrix[i * local_cols + j] * vector[j];
            printf("result_index_offset:%d, i:%d j=%d partial_result=%f, cur_el=%f * vect_el=%f\n", (i + result_index_offset * local_rows), i, j, partial_result[i + result_index_offset * local_rows], local_matrix[i * local_cols + j], vector[j]);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int block_size = (int) sqrt(size);
    if ((block_size * block_size != size) || ((rows * cols) % size != 0)) {
        if (rank == 0) {
            printf("Error: matrix rows and cols should be divisible by number of processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double *matrix = NULL;
    double *vector = NULL;
    double *result = NULL;
    int *result_indeces = NULL;
    double *block_matrix = NULL;

    int local_rows = rows / block_size; //2
    int local_cols = cols / block_size; //3

    if (rank == 0) {
        matrix = malloc(rows * cols * sizeof(double));
        block_matrix = malloc(rows * cols * sizeof(double));
        result_indeces = malloc(size * sizeof(int));
        vector = malloc(cols * local_rows * sizeof(double));
        result = calloc(rows, sizeof(double));

        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = i + 1;
        }

        int block_index = 0;
        for (int block_row = 0; block_row < block_size; block_row++) {
            for (int block_col = 0; block_col < block_size; block_col++) {
                for (int r = 0; r < local_rows; r++) {
                    for (int c = 0; c < local_cols; c++) {
                        int global_row = block_row * local_rows + r;
                        int global_col = block_col * local_cols + c;
                        int global_index = global_row * cols + global_col;

                        block_matrix[block_index++] = matrix[global_index];
                    }
                }
            }
        }
        for (int i = 0; i < rows * cols; i++) {
            printf("%f ", block_matrix[i]);
        }
        printf("\n");

        for (int i = 0; i < cols * local_rows; i++) {
            vector[i] = i + 1.0;
            if (i >= cols) {
                vector[i] = vector[i % cols];
            }
        }
        
        int row_block_count = size / local_rows;
        for (int i = 0; i < size; ++i) {
            result_indeces[i] = (int) (i / row_block_count);
        }
    }

    // Распределение памяти для локальной матрицы, вектора и результата
    double *local_matrix = malloc(local_rows * local_cols * sizeof(double));
    double *partial_result = malloc(rows * sizeof(double));
    double *local_vector = malloc(local_cols * sizeof(double));
    int *cur_index = malloc(sizeof(int));
    if (rank == 0) {
        MPI_Scatter(block_matrix, local_rows * local_cols, MPI_DOUBLE,
                    local_matrix, local_rows * local_cols, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(NULL, local_rows * local_cols, MPI_DOUBLE,
                    local_matrix, local_rows * local_cols, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Scatter(vector, local_cols, MPI_DOUBLE, local_vector, local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(NULL, local_cols, MPI_DOUBLE, local_vector, local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Scatter(result_indeces, 1, MPI_INT, cur_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(NULL, 1, MPI_INT, cur_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    matrix_vector_multiply(local_matrix, local_vector, partial_result, local_rows, local_cols, *cur_index);

    int code = MPI_Reduce(partial_result, result, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Вывод результата
        printf("Resulting vector:\n");
        for (int i = 0; i < rows; i++) {
            printf("%f\n", result[i]);
        }

        // Очистка памяти
        free(matrix);
        free(vector);
        free(result);
        free(result_indeces);
    }

    // Очистка памяти локальных массивов
    free(local_matrix);
    free(local_vector);
    free(partial_result);
    free(cur_index);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
