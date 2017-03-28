#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include "list.h"

#define SECRET_NUM -1234
#define TWO_PI 6.2831853071795864769252866

#ifdef __cplusplus
extern "C" {
#endif

int *read_map(char *filename);
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
void free_ptrs(void **ptrs, int n);
char *basecfg(char *cfgfile);
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
void find_replace(char *str, char *orig, char *rep, char *output);
void error(const char *s);
void malloc_error();
void file_error(char *s);
void strip(char *s);
void strip_char(char *s, char bad);
void top_k(float *a, int n, int k, int *index);
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
float *parse_fields(char *line, int n);
void normalize_array(float *a, int n);
void scale_array(float *a, int n, float s);
void translate_array(float *a, int n, float s);
int max_index(float *a, int n);
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);
float mse_array(float *a, int n);
float rand_normal();
size_t rand_size_t();
float rand_uniform(float min, float max);
float rand_scale(float s);
int rand_int(int min, int max);
float sum_array(float *a, int n);
float mean_array(float *a, int n);
void mean_arrays(float **a, int n, int els, float *avg);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
int sample_array(float *a, int n);
void print_statistics(float *a, int n);


#ifdef __cplusplus
}
#endif

#endif

