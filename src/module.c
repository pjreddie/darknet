#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "blas.h"
#include "detection.h"

detection* predict(network * net,image im,float thresh,char* nameslist){
      int j;
	  char ** names = get_labels(nameslist);
      image sized = letterbox_image(im, net->w, net->h);
      float hier_thresh = thresh;
      float nms=.3;
      //image sized = resize_image(im, net->w, net->h);
      //image sized2 = resize_max(im, net->w);
      //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
      //resize_network(&net, sized.w, sized.h);
      layer l = net->layers[net->n-1];

      box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
      float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
      for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));
      float **masks = 0;
      if (l.coords > 4){
          masks = calloc(l.w*l.h*l.n, sizeof(float*));
          for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = calloc(l.coords-4, sizeof(float *));
      }

      float *X = sized.data;
      network_predict(*net, X);
      get_region_boxes(l, im.w, im.h, net->w, net->h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
      if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

      return get_detections(l.w * l.h * l.n, thresh, boxes, probs, names,l.classes);


}




