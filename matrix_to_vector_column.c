#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


int rows = 4; // Число строк матрицы
int cols = 6; // Число столбцов матрицы

void local_matrix_vector_multiply(double *local_matrix, double *local_vector, double *partial_result, int local_cols) {
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

    if (cols % size != 0) {
        if (rank == 0) {
            printf("Число столбцов должно быть кратно числу процессов!\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double *matrix = NULL;
    double *col_matrix = NULL;
    double *vector = NULL;
    double *result = NULL;

    // Инициализация данных на процессе 0
    if (rank == 0) {
        matrix = malloc(rows * cols * sizeof(double));
        col_matrix = malloc(rows * cols * sizeof(double));
        vector = malloc(cols * sizeof(double));
        result = malloc(rows * sizeof(double));

        // Заполнение матрицы и вектора
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = i + 1;
            printf("%f ",matrix[i]);
        }
        printf("\n");

        int counter = 0;
        for (int i = 0; i< cols; i++) {
            for (int j = 0; j < rows; ++j) {
                col_matrix[counter++] = matrix[i+j*cols];
                printf("%f ",col_matrix[counter-1]);
            }
        }
        printf("\n");

        for (int i = 0; i < cols; i++) {
            vector[i] = i + 1; // Пример вектора
        }
    }

    // Распределение памяти для локальной матрицы, локального вектора и частичного результата
    int local_cols = cols / size; // Число столбцов на процесс
    double *local_matrix = malloc(rows * local_cols * sizeof(double));
    double *local_vector = malloc(local_cols * sizeof(double));
    double *partial_result = malloc(rows * sizeof(double));

    if (rank == 0) {
        MPI_Scatter(col_matrix, rows * local_cols, MPI_DOUBLE,
                    local_matrix, rows * local_cols, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(NULL, rows * local_cols, MPI_DOUBLE,
                    local_matrix, rows * local_cols, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Рассылка части вектора
    MPI_Scatter(vector, local_cols, MPI_DOUBLE,
                local_vector, local_cols, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Локальное умножение матрицы на часть вектора
    local_matrix_vector_multiply(local_matrix, local_vector, partial_result, local_cols);

    // Сбор частичных результатов на процессе 0
    if (rank == 0) {
        result = calloc(rows, sizeof(double));
    }

    MPI_Reduce(partial_result, result, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Вывод результата на процессе 0
    if (rank == 0) {
        printf("Результат умножения матрицы на вектор:\n");
        for (int i = 0; i < rows; i++) {
            printf("%f\n", result[i]);
        }

        // Освобождение памяти
        free(matrix);
        free(vector);
        free(result);
    }

    // Освобождение памяти локальных массивов
    free(local_matrix);
    free(local_vector);
    free(partial_result);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
