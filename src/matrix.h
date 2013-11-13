#ifndef MATRIX_H
#define MATRIX_H
typedef struct matrix{
    int rows, cols;
    double **vals;
} matrix;

matrix make_matrix(int rows, int cols);
void free_matrix(matrix m);
void print_matrix(matrix m);

matrix csv_to_matrix(char *filename);
matrix hold_out_matrix(matrix *m, int n);

double *pop_column(matrix *m, int c);

#endif
