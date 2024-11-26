#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int comm_sz; int my_rank;
    MPI_Init(NULL, NULL);
    comm_sz = 2;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
    MPI_Datatype type_indexed ;
    int array_of_blocklenghts [4] = {4, 3, 2, 1}; 
    int array_of_displscements [4] = {0, 5, 10, 15};
    MPI_Type_indexed(4 , array_of_blocklenghts , array_of_displscements , MPI_INT, &type_indexed);
    MPI_Type_commit(&type_indexed );
    int mat[16], mat1[10];
    if (my_rank == 0) {
        for (int i = 0; i < 16; ++i) {
            mat[i] = i;
        }
        MPI_Send(mat,1 ,type_indexed ,1 ,0 ,MPI_COMM_WORLD);
    } else {
        MPI_Recv(mat1, 10 ,MPI_INT,0 ,0 ,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        for (int i=0;i<10;++i){
            printf("%3d", mat1[i]) ; 
        }
        printf( "\n" ) ; 
    }
    MPI_Type_free(&type_indexed ) ;
    MPI_Finalize ( ); 
}