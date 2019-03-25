#include "table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

table make_table(int rows, int cols)
{
    table t;
    t.rows = rows;
    t.cols = cols;
    t.vals = malloc(t.rows * t.cols * sizeof(char *) * 1024);
    return t;
}

table copy_table(table t) 
{
    table c = {0};
    c.rows = t.rows;
    c.cols = t.cols;
    c.vals = malloc(t.rows * t.cols * sizeof(char *) * 1024);
    int i, j, p;
    for (i = 0; i < c.rows; ++i) {
        for (j = 0; j < c.cols; ++j) {
            p = i*c.rows + j;
            c.vals[p] = t.vals[p];
        } 
    }
    return c;
}

void table_to_csv(table t1, table t2, char *filename) 
{
    char *default_csv = "classifier_predictions.csv";
    char *csv_classes;
    
    if (!filename) {
        filename = default_csv;
        csv_classes = "classifier_classes.csv";
    } else {
        int k = strlen(filename) - 4;
        char *csv_ext = ".csv";
        csv_classes = calloc(strlen(filename), sizeof(char));
        if (k > 0) {
            int c = 0;
            char sub[5];
            while (c < 4) {
                sub[c] = filename[k + c];
                c++;
            }
            sub[c] = '\0';
            if (0!=strcmp(sub, csv_ext)) {
                strcpy(csv_classes, filename);
                strcat(filename, csv_ext);
            } else {
                strncpy(csv_classes, filename, k);
            }
        } else if (k == 0) {
            strcpy(csv_classes, filename);
            if (0!=strcmp(filename, csv_ext)) 
                strcat(filename, csv_ext);
        } else {
            strcpy(csv_classes, filename);
            strcat(filename, csv_ext);
        }
        
        strcat(csv_classes, "_classes.csv");
    }

    printf("\nCSV files...\n");
    save_csv(t1, filename);
    save_csv(t2, csv_classes);
}

void save_csv(table t, char *filename)
{
    FILE *fp;
    fp = fopen(filename, "w");

    int i, j;
    for (i = 0; i < t.rows; ++i) {
        for (j = 0; j < t.cols; ++j) {
            if (j > 0) fprintf(fp, ",");
            fprintf(fp, "%s", t.vals[ i*t.cols + j ]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("%s file created!\n", filename);
}

void print_table(table t)
{
    int i, j;
    for (i = 0; i < t.rows; ++i){
        for (j = 0; j < t.cols; ++j) {
            printf(" %s", t.vals[ i*t.cols + j ]);
        }
        printf("\n");
    }
}