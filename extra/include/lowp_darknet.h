// @file lowp_darknet.h
//
//  \date Created on: Oct 14, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#ifndef EXTRA_LOWP_DARKNET_H_
#define EXTRA_LOWP_DARKNET_H_
#include <stdbool.h>
#include "darknet.h"

typedef struct {
  void *data;
  int bytes_per_ele;
  float max;
  float min;
  int count;
  bool alloc;
} LowpTensor;

void *LowpToFloat(LowpTensor *lowp_tensor, float *f32_data);

LowpTensor FloatToLowp(const float *f32_data, const int no_elements,
                       const int bytes_per_element, void *lowp_data);

void WriteLowpTensorToFile(LowpTensor t, FILE *fp);

void LoadLowpTensorAsFloatFromFile(float *f32_data, FILE *fp,
                                   int bytes_per_ele, size_t count);

void FreeLowpTensor(LowpTensor t);

LowpTensor CreateLowpTensor(int bytes_per_ele, size_t count);

// API to load low precision weight file into the network. The weights are
// unpacked as floating point values for regular float inference. The main
// use of this API is to reduce the space of the disk to store the model. Memory
// requirement and the compute requirements remain the same. Init time will
// increase since the conversion process from low precision to float takes
// time.
//
// Call parse_network_cfg(filename) before calling this API as this API expects
// all model weight buffers to be allocated by the parse_network_cfg() API.
void LoadLowpWeightsAsFloatUpto(network *net, char *filename, int start,
                                int end);

// API to convert the floating point weights into low precision weights and
// store into the model file. The model file contains one extra integer at
// the beginning of the model to represent no_bytes used to represent a single
// element. This number comes after net.seen. Also, the model contains 2 floats
// at the beginning of each array representing max and min of the array being
// stored in the low precision format.
void SaveWeightsAsLowpModelUpto(network net, char *filename, int cutoff,
                                int bytes_per_element);


// API to convert floating point model to low precision model
void CreateLowpModel(int argc, char **argv);

#endif  // EXTRA_LOWP_DARKNET_H_
