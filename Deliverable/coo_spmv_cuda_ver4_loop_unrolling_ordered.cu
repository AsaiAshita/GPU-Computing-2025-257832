#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

int WARM_UP = 20;
int EXECUTION = 250;

__global__ void sparse_matrix_dense_vector_mult(float result_vector[], float vector[], int row_index[], int column_index[], float value[], int nonzero){
    int idx = (blockIdx.x * blockDim.x + threadIdx.x)*4;
    if (idx < nonzero) {
        int row = row_index[idx-4];
        int row_1 = row_index[idx-3];
        int row_2 = row_index[idx-2];
        int row_3 = row_index[idx-1];
        int col = column_index[idx-4];
        int col_1 = column_index[idx-3];
        int col_2 = column_index[idx-2];
        int col_3 = column_index[idx-1];
        float val = value[idx-4];
        float val_1 = value[idx-3];
        float val_2 = value[idx-2];
        float val_3 = value[idx-1];
        float prod = val * vector[col];
        float prod_1 = val_1 * vector[col_1];
        float prod_2 = val_2 * vector[col_2];
        float prod_3 = val_3 * vector[col_3];
        result_vector[row] = result_vector[row] + prod;
        result_vector[row_1] = result_vector[row_1] + prod_1; 
        result_vector[row_2] = result_vector[row_2] + prod_2; 
        result_vector[row_3] = result_vector[row_3] + prod_3;  
    }
    else{
        int remaining_operations = idx-nonzero;
        while(remaining_operations != 0){
            int row = row_index[nonzero - remaining_operations];
            int col = column_index[nonzero - remaining_operations];
            float val = value[nonzero - remaining_operations];
            float prod = val * vector[col];
            result_vector[row] = result_vector[row] + prod; 
            remaining_operations--; 
        }
    }
}

int skip_comment_lines(FILE *file) {
    //Remember the current position and declare the char array to use to skip lines, its length and the value c used to
    //verify if the line starts with a "%" or if the file has reached EOF
    long pos;
    int c;
    char * line = NULL;
    size_t len;
    //We save the position at the start of the line, peek at the first character and see what it is. If it is the EoF, then
    //we return 0. If c is different than "%", this means we have finished reading the commented lines, thus we need to reposition
    //ourselves at the start of the line and return to the main program. Else, we are reading a comment line, so we can skip
    //it by reading it with getline.
    while (1) {
        pos = ftell(file);
        c = fgetc(file);
        if (c == EOF) {
            return 0;
        }
        if (c != '%') {
            fseek(file, pos, SEEK_SET);
            free(line);
            return 1;
        }
        getline(&line, &len, file);
    }
}

bool verify_results(float * gpu_results, float * vector, int * row_index, float * value_content, int * column_index, int row, int nonzero){
    float result_vector[row];
    for (int i=0; i<nonzero; i++){
        result_vector[row_index[i]] = result_vector[row_index[i]] + (value_content[i] * vector[column_index[i]]); 
    }
    for (int i=0; i<row; i++){
        if (abs(gpu_results[i]) - abs(result_vector[i]) > 1e-4){
            printf("WRONG RESULT: position %d, GPU value: %f  CPU value: %f\n", i, gpu_results[i], result_vector[i]);
            return 0;
        }
    }
    //printf("All values correct!\n");
    return 1;
}

void swap(int * row_index, int * column_index, float * value, int i, int j){
    //A simple and naive swap function that switches two given positions, keeping the value consistent
    //across all necessary arrays
    int temp_row, temp_column;
    float temp_value;
    temp_row = row_index[i];
    temp_column = column_index[i];
    temp_value = value[i];
    row_index[i] = row_index[j];
    column_index[i] = column_index[j];
    value[i] = value[j];
    row_index[j] = temp_row;
    column_index[j] = temp_column;
    value[j] = temp_value;
}

int main(int argc, char * argv[]){
    //We check that the number of arguments is right: if it is not, we exit the program
    if (argc != 2){
        printf("Usage: ./a.out <path/to/matrix.mtx>\n");
        exit(1);
    }

    //We open the file in read mode
    FILE* matrix;
    matrix = fopen(argv[1], "r");
    //We check that the file was effectively opened: else, we exit the program
    if (matrix == NULL) {
        printf("An error occured while opening the file");
        fclose(matrix);
        exit(2);
    }

    //We declare the necessary variables to read every single line of the file, knowing that the file is formatted as such:
    //% delimitates the comments, we can safely ignore such lines
    //the first line without % is composed of three numbers: the number of rows, of columns and of nonzero elements of the matrix
    //subsequent lines give the position of the nonzero elements in the following format: row_number column_number double_value
    int column, row, nonzero = 0;
    float value = 0.0;
    //We skip over all of the comment lines
    int res = skip_comment_lines(matrix);
    //if res == 0, we have reached the EoF: this means the file is made only of comments, thus we exit the program.
    if (res == 0){
        fprintf(stderr, "End of File reached before being able to read matrix dimension");
        fclose(matrix);
        exit(3);
    }
    //We get the matrix dimension plus the number of nonzero positions: if the format is incorrect, we exit the program.
    res = fscanf(matrix, "%d %d %d", &row, &column, &nonzero);
    if (res != 3) {
        printf("%d", res);
        printf("%d %d %d", row, column, nonzero);
        fprintf(stderr, "Error reading matrix size\n");
        fclose(matrix);
        exit(4);
    }
    //Remove comment to debug matrix dimension and number of nonzero elements
    printf("Matrix dimensions: %d x %d, Non-zeros: %d\n", row, column, nonzero);

    //We declare the arrays necessary to store the matrix in COO format and we read each line of the file, expecting the following:
    //row_number column_number double_value
    //If the following is not respected, the program exists and returns an error.
    int row_index[nonzero];
    int column_index[nonzero];
    float value_content[nonzero];
    int row_entry, column_entry = 0;  

    for (int i = 0; i < nonzero; i++) {
        if (fscanf(matrix, "%d %d %f\n", &row_entry, &column_entry, &value) != 3) {
            fprintf(stderr, "Error reading matrix entry %d\n", i);
            fclose(matrix);
            exit(5);
        }
        //printf("Entry %d: row %d, col %d, val %f\n", i + 1, row, column, value);
        row_index[i] = row_entry - 1; //-1 to simplify further calculations, so to make row 1 be equal to 0
        column_index[i] = column_entry - 1; //same as above
        value_content[i] = value;
    }

    for(int i=0; i<nonzero-1; i++){
        for(int j=1; j<nonzero; j++){
            if(row_index[i] > row_index[j]){
                swap(row_index, column_index, value_content, i, j);
            }
            else{
                if(column_index[i] > column_index[j]){
                swap(row_index, column_index, value_content, i, j);
                }
            }
        }
    }


    //We declare the dense vector as an array of double of size equal to the number of rows of the matrix
    //(in order to be able to perform a matrix vector multiplication) and we generate random values to populate it.
    //We limit the size of an array element to be less than 1000 just to avoid possible overflows.
    float vector[column];
    srandom(time(NULL));
    for(int i=0; i<column; i++){
        while (vector[i] == 0){
            vector[i] = ((float)rand()/(float)(RAND_MAX)) * 5.0;
            //printf("ELEMENT %d = %lf\n", i, vector[i]);
        }
    }

    //We execute the sparse matrix dense vector multiplication
    float result_vector[row];
    float timer[EXECUTION];
    float flops[EXECUTION];
    float bandwidth[EXECUTION];

    int *d_rows, *d_columns;
    float *d_value, *d_vector, *d_result;

    // Allocate memory on device
    cudaMalloc((void**)&d_rows, sizeof(row_index));
    cudaMalloc((void**)&d_columns, sizeof(column_index));
    cudaMalloc((void**)&d_value, sizeof(value_content));
    cudaMalloc((void**)&d_vector, sizeof(vector));
    cudaMalloc((void**)&d_result, sizeof(result_vector));

    // Copy data to device
    cudaMemcpy(d_rows, row_index, sizeof(row_index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, column_index, sizeof(column_index), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value_content, sizeof(value_content), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, sizeof(vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result_vector, sizeof(result_vector), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsSizes [] = {32,64,128,256,512,1024};
    for(int sizes = 0; sizes<6; sizes++){
        int threadsPerBlock = threadsSizes[sizes];
        int blocksPerGrid = (nonzero/4 + threadsPerBlock - 1) / threadsPerBlock;

        for(int j = -WARM_UP; j < EXECUTION; j++){
            //We execute the task
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            // Launch your CUDA kernel here
            sparse_matrix_dense_vector_mult<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_vector, d_rows, d_columns, d_value, nonzero);
            //Stop Event and print kernel time
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start,stop);
            //printf("Kernel Time: %f ms\n",milliseconds);
            float seconds = milliseconds / 1000.0f;
            //printf("SECONDS: %f\n", seconds);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaMemset(d_result, 0, sizeof(result_vector));
            if (j>=0){
                timer[j] = seconds;
                flops[j] = (2*nonzero*4)/seconds;
                bandwidth[j] = ((nonzero*24*4)/1e9)/seconds;
                cudaMemcpy(result_vector, d_result, sizeof(result_vector), cudaMemcpyDeviceToHost);
                if (verify_results(result_vector, vector, row_index, value_content, column_index, row, nonzero)){
                    continue;
                }
                else{
                    printf("Exiting program...");
                    exit(6);
                }
            }
        }
        float total_time = 0;
        float total_flops = 0;
        float effective_bandwidth = 0;
        float geometric_time = 0;
        float geometric_flops = 0;
        float geometric_bandwidth = 0;
        for (int i=0; i<EXECUTION; i++){
            total_time = total_time + timer[i];
            total_flops = total_flops + flops[i];
            effective_bandwidth = effective_bandwidth + bandwidth[i];
            geometric_time = geometric_time + log(timer[i]);
            geometric_flops = geometric_flops + log(flops[i]);
            geometric_bandwidth = geometric_bandwidth + log(bandwidth[i]);
        }
        printf("Av. Time: %f seconds for (%d thread, %d blocks)\n", total_time/EXECUTION, threadsPerBlock, blocksPerGrid);
        printf("Av. FLOPS: %f FLOPS/seconds for (%d thread, %d blocks)\n", total_flops/EXECUTION, threadsPerBlock, blocksPerGrid);
        printf("Av. Bandwidth: %f GB/seconds for (%d thread, %d blocks)\n", effective_bandwidth/EXECUTION, threadsPerBlock, blocksPerGrid);
        printf("Geo_mean Time: %f seconds for (%d thread, %d blocks)\n", exp(geometric_time / EXECUTION), threadsPerBlock, blocksPerGrid);
        printf("Geo_mean FLOPS: %f FLOPS/seconds for (%d thread, %d blocks)\n", exp(geometric_flops / EXECUTION), threadsPerBlock, blocksPerGrid);
        printf("Geo_mean Bandwidth: %f GB/seconds for (%d thread, %d blocks)\n", exp(geometric_bandwidth / EXECUTION), threadsPerBlock, blocksPerGrid);
    }
    fclose(matrix);
    return 0;
}