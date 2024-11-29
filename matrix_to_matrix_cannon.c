#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#define PRINT_RESULT


void multiply_matrix(double* A, double* B, int rowsA, int colsArowsB, int colsB, double* C) {
    for (int i = 0; i < rowsA * colsB; i++) {
        C[i] = 0.0;
    }

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsArowsB; k++) {
                C[i * colsB + j] += A[i * colsArowsB + k] * B[k * colsB + j];
            }
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

    int rowsA;
    int colsArowsB;
    int colsB;

    int q = (int) sqrt(size);    
    if (rank == 0) {
        printf("Enter number of A rows: \n");
        scanf("%d", &rowsA);
        printf("Enter number of A columns and B rows: \n");
        scanf("%d", &colsArowsB);
        printf("Enter number of B columns: \n");
        scanf("%d", &colsB);

        if ((q * q != size) || ((rowsA * colsArowsB) % size != 0) || ((colsArowsB * colsB) % size != 0)) {
            printf("Error: number of rows should be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsArowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *block_matrix_A = NULL;
    double *block_matrix_B = NULL;
    double *gathered_C = NULL;

    int block_rowsA = rowsA / q;
    int block_colsA_rowsB = colsArowsB / q;
    int block_colsB = colsB / q;

    if (rank == 0) {
        A = malloc(rowsA * colsArowsB * sizeof(double));
        B = malloc(colsArowsB * colsB * sizeof(double));
        C = calloc(rowsA * colsB, sizeof(double));
        gathered_C = calloc(rowsA * colsB, sizeof(double));
        block_matrix_A = malloc(rowsA * colsArowsB * sizeof(double));
        block_matrix_B = malloc(colsArowsB * colsB * sizeof(double));

        for (int i = 0; i < rowsA * colsArowsB; i++) {
            A[i] = i + 1;
        }

        for (int i = 0; i < colsArowsB * colsB; i++)
        {
            B[i] = i + rowsA * colsArowsB + 1;
        }        

        divide_to_block_matrix(rowsA, colsArowsB, q, A, block_matrix_A);
        divide_to_block_matrix(colsArowsB, colsB, q, B, block_matrix_B);
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
    double *block_A = malloc(block_rowsA * block_colsA_rowsB * sizeof(double));
    double *block_B = malloc(block_colsB * block_colsA_rowsB * sizeof(double));
    double *block_C = calloc(block_rowsA * block_colsB, sizeof(double));
    double *partitial_block_C = calloc(block_rowsA * block_colsB, sizeof(double));

    MPI_Scatter(block_matrix_A, block_rowsA * block_colsA_rowsB, MPI_DOUBLE, block_A, block_rowsA * block_colsA_rowsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(block_matrix_B, block_colsA_rowsB * block_colsB, MPI_DOUBLE, block_B, block_colsA_rowsB * block_colsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    MPI_Cart_coords(cartCommunicator, rank, 2, coords);

    MPI_Cart_shift(cartCommunicator, 1, coords[0], &left, &right);
	MPI_Sendrecv_replace(block_A, block_rowsA * block_colsA_rowsB, MPI_DOUBLE, left, 1, right, 1, cartCommunicator, MPI_STATUS_IGNORE);
    
	MPI_Cart_shift(cartCommunicator, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(block_B, block_colsA_rowsB * block_colsB, MPI_DOUBLE, up, 1, down, 1, cartCommunicator, MPI_STATUS_IGNORE);

    for (int i = 0; i < q; ++i) {
        multiply_matrix(block_A, block_B, block_rowsA, block_colsA_rowsB, block_colsB, partitial_block_C);

        for (int j = 0; j < block_rowsA * block_colsB; ++j) {
            block_C[j] += partitial_block_C[j];
        }

        MPI_Cart_shift(cartCommunicator, 1, 1, &left, &right);
        MPI_Cart_shift(cartCommunicator, 0, 1, &up, &down);
        MPI_Sendrecv_replace(block_A, block_rowsA * block_colsA_rowsB, MPI_DOUBLE, left, 1, right, 1, cartCommunicator, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(block_B, block_colsA_rowsB * block_colsB, MPI_DOUBLE, up, 1, down, 1, cartCommunicator, MPI_STATUS_IGNORE);
    }

    MPI_Gather(block_C, block_rowsA * block_colsB, MPI_DOUBLE, gathered_C, block_rowsA * block_colsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double work_time = (end_time - start_time) * 1000;

    if (rank == 0) {
        divide_to_block_matrix(rowsA, colsB, q, gathered_C, C);
    #  ifdef PRINT_RESULT
        printf("Result matrix: \n");
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                printf("%f ", C[i * colsB + j]);
            }
            printf("\n");
        }
    #  endif
        printf("Total work time in ms: %f\n", work_time);

        free(A);
        free(B);
        free(C);
        free(block_matrix_A);
        free(gathered_C);
        free(block_matrix_B);
    }

    free(block_A);
    free(block_B);
    free(block_C);
    free(partitial_block_C);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
