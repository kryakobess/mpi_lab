#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void local_matrix_vector_multiply(double *local_matrix, double *local_vector, double *partial_result, int local_cols, int rows) {
    for (int i = 0; i < rows; i++) {
        partial_result[i] = 0.0;
        for (int j = 0; j < local_cols; j++) {
            partial_result[i] += local_matrix[j*rows + i] * local_vector[j];
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows, cols;

    if (rank == 0) {
        printf("Enter number of rows: \n");
        scanf("%d", &rows);
        printf("Enter number of columns: \n");
        scanf("%d", &cols);

        if (cols % size != 0) {
            printf("Error: number of cols should be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *matrix = NULL;
    double *col_matrix = NULL;
    double *vector = NULL;
    double *result = NULL;
    int local_cols = cols / size;
    double *local_matrix = malloc(rows * local_cols * sizeof(double));
    double *local_vector = malloc(local_cols * sizeof(double));
    double *partial_result = malloc(rows * sizeof(double));

    if (rank == 0) {
        matrix = malloc(rows * cols * sizeof(double));
        col_matrix = malloc(rows * cols * sizeof(double));
        vector = malloc(cols * sizeof(double));
        result = calloc(rows, sizeof(double));

        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = i + 1;
        }

        int counter = 0;
        for (int i = 0; i< cols; i++) {
            for (int j = 0; j < rows; ++j) {
                col_matrix[counter++] = matrix[i+j*cols];
            }
        }

        for (int i = 0; i < cols; i++) {
            vector[i] = i + 1;
        }
    }

    MPI_Scatter(col_matrix, rows * local_cols, MPI_DOUBLE, local_matrix, rows * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vector, local_cols, MPI_DOUBLE, local_vector, local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    local_matrix_vector_multiply(local_matrix, local_vector, partial_result, local_cols, rows);
    MPI_Reduce(partial_result, result, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double work_time = (end_time - start_time) * 1000;

    if (rank == 0) {
        #ifdef PRINT_RESULT
        printf("Resulting vector:\n");
        for (int i = 0; i < rows; i++) {
            printf("%f\n", result[i]);
        }
        #endif
        
        printf("Total work time in ms: %f\n", work_time);

        free(matrix);
        free(vector);
        free(result);
    }

    free(local_matrix);
    free(local_vector);
    free(partial_result);

    MPI_Finalize();
    return 0;
}
