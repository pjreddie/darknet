// @file lowp_model.c
//
//  \date Created on: Oct 14, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//
#include "darknet.h"
#include "lowp_darknet.h"
#include "utils.h"
#include "convolutional_layer.h"
#include "connected_layer.h"
#include "batchnorm_layer.h"
#include "local_layer.h"

#include <assert.h>
#include <stdio.h>

extern void transpose_matrix(float *a, int rows, int cols);

void SaveConvolutionalWeightsLowp(layer l, int nbytes, FILE *fp) {
  assert(l.binary == 0);

#ifdef GPU
  if(gpu_index >= 0){
    pull_convolutional_layer(l);
  }
#endif
  LowpTensor bias = FloatToLowp(l.biases, l.n, nbytes, NULL);
  WriteLowpTensorToFile(bias, fp);
  FreeLowpTensor(bias);

  if (l.batch_normalize){
    LowpTensor scales = FloatToLowp(l.scales, l.n, nbytes, NULL);
    LowpTensor mean = FloatToLowp(l.rolling_mean, l.n, nbytes, NULL);
    LowpTensor var = FloatToLowp(l.rolling_variance, l.n, nbytes, NULL);
    WriteLowpTensorToFile(scales, fp);
    WriteLowpTensorToFile(mean, fp);
    WriteLowpTensorToFile(var, fp);
    FreeLowpTensor(scales);
    FreeLowpTensor(mean);
    FreeLowpTensor(var);
  }
  LowpTensor weights = FloatToLowp(l.weights, l.nweights, nbytes, NULL);
  WriteLowpTensorToFile(weights, fp);
  FreeLowpTensor(weights);
}

void SaveConnectedWeightsLowp(layer l, int nbytes, FILE *fp) {
#ifdef GPU
  if(gpu_index >= 0){
    pull_connected_layer(l);
  }
#endif
  LowpTensor bias = FloatToLowp(l.biases, l.n, nbytes, NULL);
  WriteLowpTensorToFile(bias, fp);
  FreeLowpTensor(bias);
  LowpTensor weights = FloatToLowp(l.weights, l.outputs*l.inputs, nbytes, NULL);
  WriteLowpTensorToFile(weights, fp);
  FreeLowpTensor(weights);

  if (l.batch_normalize){
    LowpTensor scales = FloatToLowp(l.scales, l.outputs, nbytes, NULL);
    LowpTensor mean = FloatToLowp(l.rolling_mean, l.outputs, nbytes, NULL);
    LowpTensor var = FloatToLowp(l.rolling_variance, l.outputs, nbytes, NULL);
    WriteLowpTensorToFile(scales, fp);
    WriteLowpTensorToFile(mean, fp);
    WriteLowpTensorToFile(var, fp);
    FreeLowpTensor(scales);
    FreeLowpTensor(mean);
    FreeLowpTensor(var);
  }
}

void SaveBatchnormWeightsLowp(layer l, int nbytes, FILE *fp) {
#ifdef GPU
  if(gpu_index >= 0){
    pull_batchnorm_layer(l);
  }
#endif
  LowpTensor scales = FloatToLowp(l.scales, l.c, nbytes, NULL);
  LowpTensor mean = FloatToLowp(l.rolling_mean, l.c, nbytes, NULL);
  LowpTensor var = FloatToLowp(l.rolling_variance, l.c, nbytes, NULL);
  WriteLowpTensorToFile(scales, fp);
  WriteLowpTensorToFile(mean, fp);
  WriteLowpTensorToFile(var, fp);
  FreeLowpTensor(scales);
  FreeLowpTensor(mean);
  FreeLowpTensor(var);
}

void SaveWeightsAsLowpModelUpto(network net, char *filename, int cutoff,
                                int bytes_per_element) {
#ifdef GPU
  if(net.gpu_index >= 0){
    cuda_set_device(net.gpu_index);
  }
#endif
  fprintf(stderr, "Saving weights in the low precision format to %s\n",
          filename);
  FILE *fp = fopen(filename, "wb");
  if(!fp) file_error(filename);

  int major = 0;
  int minor = 2;
  int revision = 0;
  fwrite(&major, sizeof(int), 1, fp);
  fwrite(&minor, sizeof(int), 1, fp);
  fwrite(&revision, sizeof(int), 1, fp);
  fwrite(net.seen, sizeof(size_t), 1, fp);
  fwrite(&bytes_per_element, sizeof(int), 1, fp);
  printf("Major : %d\tMinor : %d\t Rev : %d\tBytes per element : %d\n",
         major, minor, revision, bytes_per_element);
  int i;
  for(i = 0; i < net.n && i < cutoff; ++i){
    layer l = net.layers[i];
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
      SaveConvolutionalWeightsLowp(l, bytes_per_element, fp);
    } if(l.type == CONNECTED){
      SaveConnectedWeightsLowp(l, bytes_per_element, fp);
    } if(l.type == BATCHNORM){
      SaveBatchnormWeightsLowp(l, bytes_per_element, fp);
    } if(l.type == RNN){
      SaveConnectedWeightsLowp(*(l.input_layer), bytes_per_element, fp);
      SaveConnectedWeightsLowp(*(l.self_layer), bytes_per_element, fp);
      SaveConnectedWeightsLowp(*(l.output_layer), bytes_per_element, fp);
    } if (l.type == LSTM) {
      SaveConnectedWeightsLowp(*(l.wi), bytes_per_element, fp);
      SaveConnectedWeightsLowp(*(l.wf), bytes_per_element, fp);
      SaveConnectedWeightsLowp(*(l.wo), bytes_per_element, fp);
      SaveConnectedWeightsLowp(*(l.wg), bytes_per_element, fp);
      SaveConnectedWeightsLowp(*(l.ui), bytes_per_element, fp);
      SaveConnectedWeightsLowp(*(l.uf), bytes_per_element, fp);
      SaveConnectedWeightsLowp(*(l.uo), bytes_per_element, fp);
      SaveConnectedWeightsLowp(*(l.ug), bytes_per_element, fp);
    } if (l.type == GRU) {
      if(1){
        SaveConnectedWeightsLowp(*(l.wz), bytes_per_element, fp);
        SaveConnectedWeightsLowp(*(l.wr), bytes_per_element, fp);
        SaveConnectedWeightsLowp(*(l.wh), bytes_per_element, fp);
        SaveConnectedWeightsLowp(*(l.uz), bytes_per_element, fp);
        SaveConnectedWeightsLowp(*(l.ur), bytes_per_element, fp);
        SaveConnectedWeightsLowp(*(l.uh), bytes_per_element, fp);
      }else{
        SaveConnectedWeightsLowp(*(l.reset_layer), bytes_per_element, fp);
        SaveConnectedWeightsLowp(*(l.update_layer), bytes_per_element, fp);
        SaveConnectedWeightsLowp(*(l.state_layer), bytes_per_element, fp);
      }
    } if (l.type == CRNN){
      SaveConvolutionalWeightsLowp(*(l.input_layer), bytes_per_element, fp);
      SaveConvolutionalWeightsLowp(*(l.self_layer), bytes_per_element, fp);
      SaveConvolutionalWeightsLowp(*(l.output_layer), bytes_per_element, fp);
    } if(l.type == LOCAL){
#ifdef GPU
      if(gpu_index >= 0){
        pull_local_layer(l);
      }
#endif
      int locations = l.out_w*l.out_h;
      int size = l.size*l.size*l.c*l.n*locations;
      LowpTensor bias = FloatToLowp(l.biases, l.outputs, bytes_per_element,
                                    NULL);
      WriteLowpTensorToFile(bias, fp);
      FreeLowpTensor(bias);
      LowpTensor weights = FloatToLowp(l.weights, size, bytes_per_element,
                                       NULL);
      WriteLowpTensorToFile(weights, fp);
      FreeLowpTensor(weights);
    }
  }
  fclose(fp);
}

void LoadConvolutionalWeightsLowpAsFloat(layer l, int nbytes, FILE *fp) {
  assert(l.binary == 0);

  LoadLowpTensorAsFloatFromFile(l.biases, fp, nbytes, l.n);

  if (l.batch_normalize && (!l.dontloadscales)){
    LoadLowpTensorAsFloatFromFile(l.scales, fp, nbytes, l.n);
    LoadLowpTensorAsFloatFromFile(l.rolling_mean, fp, nbytes, l.n);
    LoadLowpTensorAsFloatFromFile(l.rolling_variance, fp, nbytes, l.n);
  }
  LoadLowpTensorAsFloatFromFile(l.weights, fp, nbytes, l.nweights);

  if (l.flipped) {
    transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
  }
#ifdef GPU
  if(gpu_index >= 0){
    push_convolutional_layer(l);
  }
#endif
}

void LoadConnectedWeightsLowpAsFloat(layer l, int nbytes, FILE *fp,
                                     int transpose) {
  LoadLowpTensorAsFloatFromFile(l.biases, fp, nbytes, l.outputs);
  LoadLowpTensorAsFloatFromFile(l.weights, fp, nbytes, l.outputs*l.inputs);
  if(transpose){
    transpose_matrix(l.weights, l.inputs, l.outputs);
  }
  if (l.batch_normalize && (!l.dontloadscales)){
    LoadLowpTensorAsFloatFromFile(l.scales, fp, nbytes, l.outputs);
    LoadLowpTensorAsFloatFromFile(l.rolling_mean, fp, nbytes, l.outputs);
    LoadLowpTensorAsFloatFromFile(l.rolling_variance, fp, nbytes, l.outputs);
  }
#ifdef GPU
  if(gpu_index >= 0){
    push_connected_layer(l);
  }
#endif
}

void LoadBatchnormWeights(layer l, int nbytes, FILE *fp) {
  LoadLowpTensorAsFloatFromFile(l.scales, fp, nbytes, l.c);
  LoadLowpTensorAsFloatFromFile(l.rolling_mean, fp, nbytes, l.c);
  LoadLowpTensorAsFloatFromFile(l.rolling_variance, fp, nbytes, l.c);

#ifdef GPU
  if(gpu_index >= 0){
    push_batchnorm_layer(l);
  }
#endif
}

void LoadLowpWeightsAsFloatUpto(network *net, char *filename, int start,
                                int end) {
#ifdef GPU
  if(net->gpu_index >= 0){
    cuda_set_device(net->gpu_index);
  }
#endif
  fprintf(stderr, "Loading low precision model from %s...\n", filename);
  fflush(stdout);
  FILE *fp = fopen(filename, "rb");
  if(!fp) file_error(filename);

  int major;
  int minor;
  int revision;
  int bpe;
  fread(&major, sizeof(int), 1, fp);
  fread(&minor, sizeof(int), 1, fp);
  fread(&revision, sizeof(int), 1, fp);

  if ((major*10 + minor) >= 2){
    fread(net->seen, sizeof(size_t), 1, fp);
  } else {
    int iseen = 0;
    fread(&iseen, sizeof(int), 1, fp);
    *net->seen = iseen;
  }
  fread(&bpe, sizeof(int), 1, fp);
  printf("Major : %d\tMinor : %d\t Rev : %d\tBytes per element : %d\n",
         major, minor, revision, bpe);
  assert(bpe > 0 && bpe <= 4);
  int transpose = (major > 1000) || (minor > 1000);

  int i;
  for(i = start; i < net->n && i < end; ++i){
    layer l = net->layers[i];
    if (l.dontload) continue;
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
      LoadConvolutionalWeightsLowpAsFloat(l, bpe, fp);
    }
    if(l.type == CONNECTED){
      LoadConnectedWeightsLowpAsFloat(l, bpe, fp, transpose);
    }
    if(l.type == BATCHNORM){
      LoadBatchnormWeights(l, bpe, fp);
    }
    if(l.type == CRNN){
      LoadConvolutionalWeightsLowpAsFloat(*(l.input_layer), bpe, fp);
      LoadConvolutionalWeightsLowpAsFloat(*(l.self_layer), bpe, fp);
      LoadConvolutionalWeightsLowpAsFloat(*(l.output_layer), bpe, fp);
    }
    if(l.type == RNN){
      LoadConnectedWeightsLowpAsFloat(*(l.input_layer), bpe, fp, transpose);
      LoadConnectedWeightsLowpAsFloat(*(l.self_layer), bpe, fp, transpose);
      LoadConnectedWeightsLowpAsFloat(*(l.output_layer), bpe, fp, transpose);
    }
    if (l.type == LSTM) {
      LoadConnectedWeightsLowpAsFloat(*(l.wi), bpe, fp, transpose);
      LoadConnectedWeightsLowpAsFloat(*(l.wf), bpe, fp, transpose);
      LoadConnectedWeightsLowpAsFloat(*(l.wo), bpe, fp, transpose);
      LoadConnectedWeightsLowpAsFloat(*(l.wg), bpe, fp, transpose);
      LoadConnectedWeightsLowpAsFloat(*(l.ui), bpe, fp, transpose);
      LoadConnectedWeightsLowpAsFloat(*(l.uf), bpe, fp, transpose);
      LoadConnectedWeightsLowpAsFloat(*(l.uo), bpe, fp, transpose);
      LoadConnectedWeightsLowpAsFloat(*(l.ug), bpe, fp, transpose);
    }
    if (l.type == GRU) {
      if(1){
        LoadConnectedWeightsLowpAsFloat(*(l.wz), bpe, fp, transpose);
        LoadConnectedWeightsLowpAsFloat(*(l.wr), bpe, fp, transpose);
        LoadConnectedWeightsLowpAsFloat(*(l.wh), bpe, fp, transpose);
        LoadConnectedWeightsLowpAsFloat(*(l.uz), bpe, fp, transpose);
        LoadConnectedWeightsLowpAsFloat(*(l.ur), bpe, fp, transpose);
        LoadConnectedWeightsLowpAsFloat(*(l.uh), bpe, fp, transpose);
      }else{
        LoadConnectedWeightsLowpAsFloat(*(l.reset_layer), bpe, fp, transpose);
        LoadConnectedWeightsLowpAsFloat(*(l.update_layer), bpe, fp, transpose);
        LoadConnectedWeightsLowpAsFloat(*(l.state_layer), bpe, fp, transpose);
      }
    }
    if(l.type == LOCAL){
      int locations = l.out_w*l.out_h;
      int size = l.size*l.size*l.c*l.n*locations;
      LoadLowpTensorAsFloatFromFile(l.biases, fp, bpe, l.outputs);
      LoadLowpTensorAsFloatFromFile(l.weights, fp, bpe, size);
#ifdef GPU
      if(gpu_index >= 0){
        push_local_layer(l);
      }
#endif
    }
  }
  fprintf(stderr, "Done!\n");
  fclose(fp);
}

void CreateLowpModel(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "Usage : %s %s [cfg] [weights] [-bytes <bytes per element>]"
        "[-out <output model file>(optional)]\n", argv[0], argv[1]);
    return;
  }
  int bytes_per_ele = find_int_arg(argc, argv, "-bytes", 1);
  char *outfile = find_char_arg(argc, argv, "-out", 0);
  char *cfg = argv[2];
  char *model = argv[3];

  char lowp_model[256];
  if (outfile) {
    sprintf(lowp_model, "%s", outfile);
  } else {
    char *base = basecfg(cfg);
    sprintf(lowp_model, "%s_%dbit.weights", base, bytes_per_ele*8);
  }

  assert(bytes_per_ele > 0 && bytes_per_ele <= 4);
  // Init network so that all weight buffers are allocated.
  network net = parse_network_cfg(cfg);

  // Load weights into floating point buffers
  load_weights(&net, model);

  // Convert to low precision model and write to file
  SaveWeightsAsLowpModelUpto(net, lowp_model, net.n, bytes_per_ele);

  // Free the network
  free_network(net);
}


