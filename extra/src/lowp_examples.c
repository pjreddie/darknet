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

extern void print_detector_detections(FILE **fps, char *id, box *boxes,
                                      float **probs, int total, int classes,
                                      int w, int h);
extern void print_cocos(FILE *fp, int image_id, box *boxes, float **probs,
                        int num_boxes, int classes, int w, int h);
extern void print_imagenet_detections(FILE *fp, int id, box *boxes,
                                      float **probs, int total, int classes,
                                      int w, int h);

// Using functions from examples/detector.c with minor changes so as to keep
// the original Darknet source files untouched.
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

void LowpValidateDetector(char *datacfg, char *cfgfile, char *weightfile,
                          char *outfile) {
  int j;
  list *options = read_data_cfg(datacfg);
  char *valid_images = option_find_str(options, "valid", "data/train.list");
  char *name_list = option_find_str(options, "names", "data/names.list");
  char *prefix = option_find_str(options, "results", "results");
  char **names = get_labels(name_list);
  char *mapf = option_find_str(options, "map", 0);
  int *map = 0;
  if (mapf) map = read_map(mapf);

  network net = parse_network_cfg(cfgfile);
  if(weightfile){
    LoadLowpWeightsAsFloatUpto(&net, weightfile, 0, net.n);
  }
  set_batch_network(&net, 1);
  fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
          net.learning_rate, net.momentum, net.decay);
  srand(time(0));

  list *plist = get_paths(valid_images);
  char **paths = (char **)list_to_array(plist);

  layer l = net.layers[net.n-1];
  int classes = l.classes;

  char buff[1024];
  char *type = option_find_str(options, "eval", "voc");
  FILE *fp = 0;
  FILE **fps = 0;
  int coco = 0;
  int imagenet = 0;
  if(0==strcmp(type, "coco")){
    if(!outfile) outfile = "coco_results";
    snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
    fp = fopen(buff, "w");
    fprintf(fp, "[\n");
    coco = 1;
  } else if(0==strcmp(type, "imagenet")){
    if(!outfile) outfile = "imagenet-detection";
    snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
    fp = fopen(buff, "w");
    imagenet = 1;
    classes = 200;
  } else {
    if(!outfile) outfile = "comp4_det_test_";
    fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
        fps[j] = fopen(buff, "w");
    }
  }

  box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
  float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
  for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes+1, sizeof(float *));

  int m = plist->size;
  int i=0;
  int t;

  float thresh = .005;
  float nms = .45;

  int nthreads = 4;
  image *val = calloc(nthreads, sizeof(image));
  image *val_resized = calloc(nthreads, sizeof(image));
  image *buf = calloc(nthreads, sizeof(image));
  image *buf_resized = calloc(nthreads, sizeof(image));
  pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

  load_args args = {0};
  args.w = net.w;
  args.h = net.h;
  args.type = LETTERBOX_DATA;

  for(t = 0; t < nthreads; ++t) {
    args.path = paths[i+t];
    args.im = &buf[t];
    args.resized = &buf_resized[t];
    thr[t] = load_data_in_thread(args);
  }
  double start = what_time_is_it_now();
  for(i = nthreads; i < m+nthreads; i += nthreads) {
    fprintf(stderr, "%d\n", i);
    for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
      pthread_join(thr[t], 0);
      val[t] = buf[t];
      val_resized[t] = buf_resized[t];
    }
    for(t = 0; t < nthreads && i+t < m; ++t){
      args.path = paths[i+t];
      args.im = &buf[t];
      args.resized = &buf_resized[t];
      thr[t] = load_data_in_thread(args);
    }
    for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
      char *path = paths[i+t-nthreads];
      char *id = basecfg(path);
      float *X = val_resized[t].data;
      network_predict(net, X);
      int w = val[t].w;
      int h = val[t].h;
      get_region_boxes(l, w, h, net.w, net.h, thresh, probs, boxes, 0, 0, map, .5, 0);
      if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);
      if (coco){
        print_cocos(fp, path, boxes, probs, l.w*l.h*l.n, classes, w, h);
      } else if (imagenet){
        print_imagenet_detections(fp, i+t-nthreads+1, boxes, probs, l.w*l.h*l.n, classes, w, h);
      } else {
        print_detector_detections(fps, id, boxes, probs, l.w*l.h*l.n, classes, w, h);
      }
      free(id);
      free_image(val[t]);
      free_image(val_resized[t]);
    }
  }
  for(j = 0; j < classes; ++j){
    if(fps) fclose(fps[j]);
  }
  if(coco){
    fseek(fp, -2, SEEK_CUR);
    fprintf(fp, "\n]\n");
    fclose(fp);
  }
  fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}

void LowpDetector(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "Usage : %s %s [test/valid] [data_cfg] [cfg] [weights]"
        "[image file]\n", argv[0], argv[1]);
    return;
  }
  char *outfile = find_char_arg(argc, argv, "-out", 0);
  char *datacfg = argv[3];
  char *cfg = argv[4];
  char *model = argv[5];
  char *image_path = argc > 6?argv[6] : 0;

  float thresh = find_float_arg(argc, argv, "-thresh", .5);
  if (0 == strcmp(argv[2], "test")) {
    RunLowpDetector(datacfg, cfg, model, image_path, NULL, thresh);
  } else if (0 == strcmp(argv[2], "valid")) {
    LowpValidateDetector(datacfg, cfg, model, outfile);
  } else {
    printf("Invalid detector function\n");
  }
}
