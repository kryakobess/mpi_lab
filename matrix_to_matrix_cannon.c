#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int rows;
int cols;

void matrix_vector_multiply(double *local_matrix, double *vector, double *partial_result, int local_rows, int local_cols, const int result_index_offset) {
    for (int i = 0; i < local_rows; i++) {
        partial_result[i] = 0.0;
        for (int j = 0; j < local_cols; j++) {
            partial_result[i + result_index_offset * local_rows] += local_matrix[i * local_cols + j] * vector[j];
            printf("result_index_offset:%d, i:%d j=%d partial_result=%f, cur_el=%f * vect_el=%f\n", (i + result_index_offset * local_rows), i, j, partial_result[i + result_index_offset * local_rows], local_matrix[i * local_cols + j], vector[j]);
        }
    }
}

void divide_to_block_matrix(int rows, int cols, int q, const double* matrix, double* block_matrix) {
    int block_index = 0;
    int block_rows = rows / q;
    int block_cols = cols / q;
    for (int block_row = 0; block_row < q; block_row++) {
        for (int block_col = 0; block_col < q; block_col++) {
            for (int r = 0; r < block_rows; r++) {
                for (int c = 0; c < block_cols; c++) {
                    int global_row = block_row * block_rows + r;
                    int global_col = block_col * block_cols + c;
                    int global_index = global_row * cols + global_col;

                    block_matrix[block_index++] = matrix[global_index];
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int q = (int) sqrt(size);    
    if (rank == 0) {
        printf("Enter number of rows: \n");
        scanf("%d", &rows);
        printf("Enter number of columns: \n");
        scanf("%d", &cols);

        if ((q * q != size) || ((rows * cols) % size != 0)) {
            printf("Error: number of rows should be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *block_matrix_A = NULL;
    double *block_matrix_B = NULL;

    int block_rows = rows / q;
    int block_cols = cols / q;

    if (rank == 0) {
        A = malloc(rows * cols * sizeof(double));
        B = malloc(rows * cols * sizeof(double));
        C = malloc(rows * cols * sizeof(double));
        block_matrix_A = malloc(block_rows * block_cols * sizeof(double));
        block_matrix_B = malloc(block_rows * block_cols * sizeof(double));

        for (int i = 0; i < rows * cols; i++) {
            A[i] = i + 1;
            B[i] = i + 1;
        }

        divide_to_block_matrix(rows, cols, q, A, block_matrix_A);
        divide_to_block_matrix(cols, rows, q, B, block_matrix_B);
        for (int i = 0; i < rows * cols; i++) {
            printf("%f ", block_matrix_B[i]);
        }
        printf("\n");
    }

    // // Распределение памяти для локальной матрицы, вектора и результата
    // double *local_matrix = malloc(local_rows * local_cols * sizeof(double));
    // double *partial_result = malloc(rows * sizeof(double));
    // double *local_vector = malloc(local_cols * sizeof(double));
    // int *cur_index = malloc(sizeof(int));
    // if (rank == 0) {
    //     MPI_Scatter(block_matrix, local_rows * local_cols, MPI_DOUBLE,
    //                 local_matrix, local_rows * local_cols, MPI_DOUBLE,
    //                 0, MPI_COMM_WORLD);
    // } else {
    //     MPI_Scatter(NULL, local_rows * local_cols, MPI_DOUBLE,
    //                 local_matrix, local_rows * local_cols, MPI_DOUBLE,
    //                 0, MPI_COMM_WORLD);
    // }

    // if (rank == 0) {
    //     MPI_Scatter(vector, local_cols, MPI_DOUBLE, local_vector, local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // } else {
    //     MPI_Scatter(NULL, local_cols, MPI_DOUBLE, local_vector, local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // }

    // if (rank == 0) {
    //     MPI_Scatter(result_indeces, 1, MPI_INT, cur_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // } else {
    //     MPI_Scatter(NULL, 1, MPI_INT, cur_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // }

    // matrix_vector_multiply(local_matrix, local_vector, partial_result, local_rows, local_cols, *cur_index);

    // int code = MPI_Reduce(partial_result, result, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // // Вывод результата
        // printf("Resulting vector:\n");
        // for (int i = 0; i < rows; i++) {
        //     printf("%f\n", result[i]);
        // }

        // Очистка памяти
        free(A);
        free(B);
        free(C);
        free(block_matrix_A);
        free(block_matrix_B);
    }

    // // Очистка памяти локальных массивов
    // free(local_matrix);
    // free(local_vector);
    // free(partial_result);
    // free(cur_index);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
