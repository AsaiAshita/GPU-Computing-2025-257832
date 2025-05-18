#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <time.h>

int WARM_UP = 20;
int EXECUTION = 250;

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

void swap(int * row_index, int * column_index, float * value_content, int i, int j){
    int temp_row = row_index[i];
    int temp_col = column_index[i];
    float temp_val = value_content[i];
    row_index[i] = row_index[j];
    column_index[i] = column_index[j];
    value_content[i] = value_content[j];
    row_index[j] = temp_row;
    column_index[j] = temp_row;
    value_content[j] = temp_val;
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

    //printf("First element is: row %d, column %d, content %f", row_index[0], column_index[0], value_content[0]);

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

    //printf("First element is: row %d, column %d, content %f", row_index[0], column_index[0], value_content[0]);

    //We declare the dense vector as an array of double of size equal to the number of rows of the matrix
    //(in order to be able to perform a matrix vector multiplication) and we generate random values to populate it.
    //We limit the size of an array element to be less than 1000 just to avoid possible overflows.
    float vector[column];
    srandom(time(NULL));
    for(int i=0; i<column; i++){
        vector[i] = ((float)rand()/(float)(RAND_MAX)) * 5.0;
        //printf("ELEMENT %d = %lf\n", i, vector[i]);
    }

    //We execute the sparse matrix dense vector multiplication
    float result_vector[row];
    double timer[EXECUTION];
    double flops[EXECUTION];
    double bandwidth[EXECUTION];
    
    for(int j = -WARM_UP; j < EXECUTION; j++){
        //We set up the required functions to measure the time needed by the task
        clock_t tic = clock();
        //We execute the task
        for (int i=0; i<nonzero; i++){
            int temp_sum = (value_content[i]*vector[column_index[i]]);
            int reduction = i+1;
            while(reduction < nonzero && row_index[i] == row_index[reduction]){
                temp_sum = temp_sum + (value_content[reduction]*vector[column_index[reduction]]);
                reduction++;
            }
            result_vector[row_index[i]] = temp_sum;
            i = reduction;
        }
        clock_t toc = clock();
        double ex_time = (double) (toc - tic) / CLOCKS_PER_SEC;
        //We get the time the task ended and compute the time it needed to complete. If we have already finished the warm up
        //runs, then we save the time in timer to then use to compute the various benchmarks.
        if (j>=0){
            timer[j] = ex_time;
            flops[j] = (2*nonzero)/ex_time;
            bandwidth[j] = ((nonzero*24)/1e9)/ex_time; //read 2 int and 3 floats, wrote 1 float.
        }
        //for (int i=0; i<row; i++){
        //    printf("RES %d = %lf\n", i+1, result_vector[i]);
        //}
        memset(result_vector, 0, sizeof result_vector); //we empty the vector so to reuse it in subsequent runs
    }

    
    double total_time = 0;
    double total_flops = 0;
    double effective_bandwidth = 0;
    double geometric_time = 0;
    double geometric_flops = 0;
    double geometric_bandwidth = 0;
    for (int i=0; i<EXECUTION; i++){
        total_time = total_time + timer[i];
        total_flops = total_flops + flops[i];
        effective_bandwidth = effective_bandwidth + bandwidth[i];
        geometric_time = geometric_time + log(timer[i]);
        geometric_flops = geometric_flops + log(flops[i]);
        geometric_bandwidth = geometric_bandwidth + log(bandwidth[i]);
    }
    printf("Average time taken by the application: %lf seconds\n", total_time/EXECUTION);
    printf("Average FLOPS of the application: %lf FLOPS/seconds\n", total_flops/EXECUTION);
    printf("Average effective bandwidth of the application: %f GB/seconds\n", effective_bandwidth/EXECUTION);
    printf("Geometric mean of the time taken by the application: %lf seconds\n", exp(geometric_time / EXECUTION));
    printf("Geometric mean of the FLOPS done by the application: %lf FLOPS/seconds\n", exp(geometric_flops / EXECUTION));
    printf("Geometric mean of the effective bandwidth of the application: %f GB/seconds\n", exp(geometric_bandwidth / EXECUTION));

    fclose(matrix);
    return 0;
}