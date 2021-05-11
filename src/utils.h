#ifndef UTILS_H
#define UTILS_H
#include "darknet.h"
#include "list.h"

#include <stdio.h>
#include <time.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#ifdef __cplusplus
extern "C" {
#endif

LIB_API void free_ptrs(void **ptrs, int n);
LIB_API void top_k(float *a, int n, int k, int *index);

void *xmalloc(size_t size);
void *xcalloc(size_t nmemb, size_t size);
void *xrealloc(void *ptr, size_t size);
double what_time_is_it_now();
int *read_map(char *filename);
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
char *basecfg(char *cfgfile);
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
LIB_API void find_replace(const char* str, char* orig, char* rep, char* output);
void replace_image_to_label(const char* input_path, char* output_path);
void error(const char *s);
void malloc_error();
void calloc_error();
void realloc_error();
void file_error(char *s);
void strip(char *s);
void strip_args(char *s);
void strip_char(char *s, char bad);
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
int top_max_index(float *a, int n, int k);
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
float mag_array_skip(float *a, int n, int * indices_to_skip);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
int sample_array(float *a, int n);
int sample_array_custom(float *a, int n);
void print_statistics(float *a, int n);
unsigned int random_gen_fast(void);
float random_float_fast();
int rand_int_fast(int min, int max);
unsigned int random_gen();
float random_float();
float rand_uniform_strong(float min, float max);
float rand_precalc_random(float min, float max, float random_part);
double double_rand(void);
unsigned int uint_rand(unsigned int less_than);
int check_array_is_nan(float *arr, int size);
int check_array_is_inf(float *arr, int size);
int int_index(int *a, int val, int n);
int *random_index_order(int min, int max);
int max_int_index(int *a, int n);
boxabs box_to_boxabs(const box* b, const int img_w, const int img_h, const int bounds_check);
int make_directory(char *path, int mode);
unsigned long custom_hash(char *str);
bool is_live_stream(const char * path);

#define max_val_cmp(a,b) (((a) > (b)) ? (a) : (b))
#define min_val_cmp(a,b) (((a) < (b)) ? (a) : (b))

#ifdef __cplusplus
}
#endif

#endif
