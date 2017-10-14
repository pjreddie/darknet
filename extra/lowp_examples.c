// @file lowp_examples.c
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

void RunLowpDetector(char *datacfg, char *cfgfile, char *weightfile,
                     char *filename, char *outfile, float thresh) {
  list *options = read_data_cfg(datacfg);
  char *name_list = option_find_str(options, "names", "data/names.list");
  char **names = get_labels(name_list);

  image **alphabet = load_alphabet();
  network net = parse_network_cfg(cfgfile);
  if(weightfile){
    LoadLowpWeightsAsFloatUpto(&net, weightfile, 0, net.n);
  }
  set_batch_network(&net, 1);
  srand(2222222);
  double time;
  char buff[256];
  char *input = buff;
  int j;
  float nms=.3;

  image im = load_image_color(filename, 0, 0);
  image sized = letterbox_image(im, net.w, net.h);
  layer l = net.layers[net.n-1];

  box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
  float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
  for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));
  float **masks = 0;
  if (l.coords > 4){
      masks = calloc(l.w*l.h*l.n, sizeof(float*));
      for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = calloc(l.coords-4, sizeof(float *));
  }

  float *X = sized.data;
  time=what_time_is_it_now();
  network_predict(net, X);
  printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
  get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, masks, 0, 0, 0.5, 1);
  if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

  draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
  if(outfile){
    save_image(im, outfile);
  }
  else{
    save_image(im, "predictions");
#ifdef OPENCV
    cvNamedWindow("predictions", CV_WINDOW_NORMAL);
    if(1){
      cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN,
                          CV_WINDOW_FULLSCREEN);
    }
    show_image(im, "predictions");
    cvWaitKey(0);
    cvDestroyAllWindows();
#endif
  }

  free_image(im);
  free_image(sized);
  free(boxes);
  free_ptrs((void **)probs, l.w*l.h*l.n);
}

void LowpDetector(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "Usage : %s %s [data_cfg] [cfg] [weights] [image file]\n",
            argv[0], argv[1]);
    return;
  }
  char *datacfg = argv[2];
  char *cfg = argv[3];
  char *model = argv[4];
  char *image_path = argv[5];
  float thresh = find_float_arg(argc, argv, "-thresh", .24);
  RunLowpDetector(datacfg, cfg, model, image_path, NULL, thresh);
}
