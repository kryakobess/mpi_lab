#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


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

    int rows;
    int cols;

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
        block_matrix_A = malloc(rows * cols * sizeof(double));
        block_matrix_B = malloc(rows * cols * sizeof(double));

        for (int i = 0; i < rows * cols; i++) {
            A[i] = i + 1;
            B[i] = i + rows * cols + 1;
        }

        divide_to_block_matrix(rows, cols, q, A, block_matrix_A);

        for (int i = 0; i < rows * cols; ++i) {
            printf("%f ", block_matrix_A[i]);
        }
        printf("\n");
        divide_to_block_matrix(cols, rows, q, B, block_matrix_B);
    }

    MPI_Comm cartCommunicator;
    int dim[2] = {q, q};
    int periodic[2] = {1, 1}; 
    int coords[2];
    int left = -1;
    int right = -1;
    int up = -1;
    int down = -1;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periodic, 1, &cartCommunicator);

    // Распределение памяти для локальной матрицы, вектора и результата
    double *block_A = malloc(block_rows * block_cols * sizeof(double));
    double *block_B = malloc(block_rows * block_cols * sizeof(double));
    double *block_C = malloc(block_rows * block_cols * sizeof(double));

    MPI_Scatter(block_matrix_A, block_rows * block_cols, MPI_DOUBLE, block_A, block_rows * block_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(block_matrix_B, block_rows * block_cols, MPI_DOUBLE, block_B, block_rows * block_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Cart_coords(cartCommunicator, rank, 2, coords);
    //printf("rank: %d, left: %d, right: %d, coord[0]: %d, coord[1]: %d, block_rows = %d, block_cols = %d\n", rank, left, right, coords[0], coords[1], block_rows, block_cols);
    MPI_Cart_shift(cartCommunicator, 1, coords[0], &left, &right);
    //printf("shifted. rank: %d, left: %d, right: %d, coord[0]: %d, coord[1]: %d\n", rank, left, right, coords[0], coords[1]);
	MPI_Sendrecv_replace(block_A, block_rows * block_cols, MPI_DOUBLE, left, 1, right, 1, cartCommunicator, MPI_STATUS_IGNORE);
    
	MPI_Cart_shift(cartCommunicator, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(block_B, block_rows * block_cols, MPI_DOUBLE, up, 1, down, 1, cartCommunicator, MPI_STATUS_IGNORE);

    // for (int i = 0; i < block_rows * block_cols; ++i) {
    //     printf("rank: %d mat: %f\n", rank, block_A[i]);
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
    free(block_A);
    free(block_B);
    free(block_C);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
