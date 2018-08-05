
#define CPU_ONLY

#include <iostream>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/net.hpp"

using std::string;
using namespace caffe;

extern "C" void *load_caffe_model(const char *cfg, const char *weights) {
    ::google::InitGoogleLogging("caffe.log");

    std::string caffe_cfg = cfg;
    std::string caffe_weights = weights;
    std::cout << "in caffe layer impl" << std::endl;
    std::cout << caffe_cfg << std::endl;
    std::cout << caffe_weights << std::endl;

    Caffe::set_mode(Caffe::CPU);
    Net<float> *caffe_net = new Net<float>(caffe_cfg, TEST);

    caffe_net->CopyTrainedLayersFrom(caffe_weights);

    return (void *)caffe_net;
}

extern "C" void run_caffe_model(void *net, float *input) {
    Net<float> *caffe_net = (Net<float>*)net;
    Blob<float>* input_layer = caffe_net->input_blobs()[0];
    float* input_data = input_layer->mutable_cpu_data();

    size_t input_size = (input_layer->width() * input_layer->height() * input_layer->channels());

    for (int i = 0; i < input_size; i++) {
        input_data[i] = input[i];
    }

    caffe_net->Forward();
}

extern "C" float* get_output(void *net, int index) {
    std::vector<float> output;
    Net<float> *caffe_net = (Net<float>*)net;
    Blob<float>* output_layer = caffe_net->output_blobs()[index];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    output = std::vector<float>(begin, end);
    return (float *)begin;
}

extern "C" int num_inputs(void *net) {
    Net<float> *caffe_net = (Net<float>*)net;
    return caffe_net->num_inputs();
}

extern "C" int num_outputs(void *net) {
    Net<float> *caffe_net = (Net<float>*)net;
    return caffe_net->num_outputs();
}

extern "C" int input_channels(void *net, int index) {
    Net<float> *caffe_net = (Net<float>*)net;
    Blob<float>* input_layer = caffe_net->input_blobs()[index];
    return input_layer->channels();
}

extern "C" int input_height(void *net, int index) {
    Net<float> *caffe_net = (Net<float>*)net;
    Blob<float>* input_layer = caffe_net->input_blobs()[index];
    return input_layer->height();
}

extern "C" int input_width(void *net, int index) {
    Net<float> *caffe_net = (Net<float>*)net;
    Blob<float>* input_layer = caffe_net->input_blobs()[index];
    return input_layer->width();
}

extern "C" int output_channels(void *net, int index) {
    Net<float> *caffe_net = (Net<float>*)net;
    Blob<float>* output_layer = caffe_net->output_blobs()[index];
    return output_layer->channels();
}

extern "C" int output_height(void *net, int index) {
    Net<float> *caffe_net = (Net<float>*)net;
    Blob<float>* output_layer = caffe_net->output_blobs()[index];
    return output_layer->height();
}

extern "C" int output_width(void *net, int index) {
    Net<float> *caffe_net = (Net<float>*)net;
    Blob<float>* output_layer = caffe_net->output_blobs()[index];
    return output_layer->width();
}
