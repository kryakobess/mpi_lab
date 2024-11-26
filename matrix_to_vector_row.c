#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int rows = 6; // Общее количество строк матрицы
int cols = 2; // Количество столбцов матрицы

void matrix_vector_multiply(double *local_matrix, double *vector, double *local_result, int local_rows) {
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

    double *matrix = NULL;
    double *vector = NULL;
    double *result = NULL;

    // Определение локального числа строк для каждого процесса
    if (rows % size != 0 && rank == 0) {
        printf("Error: matrix rows should be divisible by number of processes.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Инициализация матрицы и вектора только на процессе 0
    if (rank == 0) {
        matrix = malloc(rows * cols * sizeof(double));
        vector = malloc(cols * sizeof(double));
        result = malloc(rows * sizeof(double));

        // Заполнение матрицы и вектора
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = i + 1; // Пример: матрица заполняется числами 1, 2, 3, ...
        }
        for (int i = 0; i < cols; i++) {
            vector[i] = 1.0; // Пример: вектор заполняется единицами
        }
    }

    // Распределение памяти для локальной матрицы, вектора и результата
    int local_rows = rows / size;
    double *local_matrix = malloc(local_rows * cols * sizeof(double));
    double *local_result = malloc(local_rows * sizeof(double));
    if (rank == 0) {
        // Рассылка строк матрицы всем процессам
        MPI_Scatter(matrix, local_rows * cols, MPI_DOUBLE,
                    local_matrix, local_rows * cols, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(NULL, local_rows * cols, MPI_DOUBLE,
                    local_matrix, local_rows * cols, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Рассылка вектора всем процессам
    if (rank == 0) {
        MPI_Bcast(vector, cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        vector = malloc(cols * sizeof(double));
        MPI_Bcast(vector, cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Локальное умножение
    matrix_vector_multiply(local_matrix, vector, local_result, local_rows);

    // Сбор результата обратно в процесс 0
    MPI_Gather(local_result, local_rows, MPI_DOUBLE,
               result, local_rows, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

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
