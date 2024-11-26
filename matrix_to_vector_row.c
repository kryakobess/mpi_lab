#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void matrix_vector_multiply(double *local_matrix, double *vector, double *local_result, int local_rows, int cols) {
    for (int i = 0; i < local_rows; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            local_result[i] += local_matrix[i * cols + j] * vector[j];
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows, cols;

    if (rank == 0) {
        printf("Enter number of rows: \n");
        scanf("%d", &rows);
        printf("Enter number of columns: \n");
        scanf("%d", &cols);

        if (rows % size != 0) {
            printf("Error: number of rows should be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *matrix = NULL;
    double *vector = NULL;
    double *result = NULL;

    int local_rows = rows / size;
    double *local_matrix = malloc(local_rows * cols * sizeof(double));
    double *local_result = malloc(local_rows * sizeof(double));

    if (rank == 0) {
        matrix = malloc(rows * cols * sizeof(double));
        vector = malloc(cols * sizeof(double));
        result = malloc(rows * sizeof(double));

        // Заполнение матрицы и вектора
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = i + 1; // Пример: матрица заполняется числами 1, 2, 3, ...
        }
        for (int i = 0; i < cols; i++) {
            vector[i] = i + 1; // Пример: вектор заполняется единицами
        }
    }

    MPI_Scatter(matrix, local_rows * cols, MPI_DOUBLE, local_matrix, local_rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        MPI_Bcast(vector, cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        vector = malloc(cols * sizeof(double));
        MPI_Bcast(vector, cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    matrix_vector_multiply(local_matrix, vector, local_result, local_rows, cols);

    MPI_Gather(local_result, local_rows, MPI_DOUBLE, result, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Resulting vector:\n");
        for (int i = 0; i < rows; i++) {
            printf("%f\n", result[i]);
        }

        free(matrix);
        free(vector);
        free(result);
    }

    // Очистка памяти локальных массивов
    free(local_matrix);
    free(local_result);
    if (rank != 0) {
        free(vector);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}