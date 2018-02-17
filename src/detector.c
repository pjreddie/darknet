#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/version.hpp"

#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)""CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#else
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)""CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif

#endif

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

	int init_w = net.w;
	int init_h = net.h;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
	args.small_object = l.small_object;
    args.d = &buffer;
    args.type = DETECTION_DATA;
	args.threads = 4;// 8;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data(args);
    clock_t time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
		if(l.random && count++%10 == 0){
            printf("Resizing\n");
			int dim = (rand() % 12 + (init_w/32 - 5)) * 32;	// +-160
            //int dim = (rand() % 10 + 10) * 32;
            //if (get_current_batch(net)+100 > net.max_batches) dim = 544;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets + i, dim, dim);
            }
            net = nets[0];
        }
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
           image im = float_to_image(448, 448, 3, train.X.vals[10]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
           draw_bbox(im, b, 8, 1,0,0);
           }
           save_image(im, "truth11");
         */

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
		//if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
		if (i % 100 == 0) {
#ifdef GPU
			if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '_');
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, box *boxes, float **probs, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, probs[i][j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (probs[i][class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, probs[i][class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network net = parse_network_cfg_custom(cfgfile, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "comp4_det_test_";
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
        snprintf(buff, 1024, "%s/coco_results.json", prefix);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        snprintf(buff, 1024, "%s/imagenet-detection.txt", prefix);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, base, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

	int detection_count = 0;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
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
            get_region_boxes(l, w, h, thresh, probs, boxes, 0, map);
            if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);

			int x, y;
			for (x = 0; x < (l.w*l.h*l.n); ++x) {
				for (y = 0; y < classes; ++y) 
				{
					if (probs[x][y]) ++detection_count;
				}
			}

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
	printf("\n detection_count = %d \n", detection_count);
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg_custom(cfgfile, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.txt");
    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j, k;
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

	float thresh = .001;// .001;	// .2;
    float iou_thresh = .5;
    float nms = .4;

	int detection_count = 0, truth_count = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 1, 0);
        if (nms) do_nms(boxes, probs, l.w*l.h*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);
	find_replace(labelpath, ".png", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
		truth_count += num_labels;
        for(k = 0; k < l.w*l.h*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
		for (j = 0; j < num_labels; ++j) {
			++total;
			box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
			float best_iou = 0;
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				float iou = box_iou(boxes[k], t);
				if (probs[k][0] > thresh && iou > best_iou) {
					best_iou = iou;
				}
			}
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
	printf("\n truth_count = %d \n", truth_count);
}

typedef struct {
	box b;
	float p;
	int class_id;
	int image_index;
	int truth_flag;
	int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
	box_prob a = *(box_prob *)pa;
	box_prob b = *(box_prob *)pb;
	float diff = a.p - b.p;
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

void validate_detector_map(char *datacfg, char *cfgfile, char *weightfile)
{
	int j;
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.txt");
	char *difficult_valid_images = option_find_str(options, "difficult", NULL);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	char *mapf = option_find_str(options, "map", 0);
	int *map = 0;
	if (mapf) map = read_map(mapf);

	network net = parse_network_cfg_custom(cfgfile, 1);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	list *plist = get_paths(valid_images);
	char **paths = (char **)list_to_array(plist);

	char **paths_dif = NULL;
	if (difficult_valid_images) {
		list *plist_dif = get_paths(difficult_valid_images);
		paths_dif = (char **)list_to_array(plist_dif);
	}
	

	layer l = net.layers[net.n - 1];
	int classes = l.classes;

	box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
	float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

	int m = plist->size;
	int i = 0;
	int t;

	const float thresh = .005;
	const float nms = .45;
	const float iou_thresh = 0.5;

	int nthreads = 4;
	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.type = IMAGE_DATA;

	const float thresh_calc_avg_iou = 0.24;
	float avg_iou = 0;
	int tp_for_thresh = 0;
	int fp_for_thresh = 0;

	box_prob *detections = calloc(1, sizeof(box_prob));
	int detections_count = 0;
	int unique_truth_count = 0;

	int *truth_classes_count = calloc(classes, sizeof(int));

	for (t = 0; t < nthreads; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	time_t start = time(0);
	for (i = nthreads; i < m + nthreads; i += nthreads) {
		fprintf(stderr, "%d\n", i);
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for (t = 0; t < nthreads && i + t < m; ++t) {
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			const int image_index = i + t - nthreads;
			char *path = paths[image_index];
			char *id = basecfg(path);
			float *X = val_resized[t].data;
			network_predict(net, X);
			get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, map);
			if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);

			char labelpath[4096];
			find_replace(path, "images", "labels", labelpath);
			find_replace(labelpath, "JPEGImages", "labels", labelpath);
			find_replace(labelpath, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".JPEG", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label *truth = read_boxes(labelpath, &num_labels);
			int i, j;
			for (j = 0; j < num_labels; ++j) {
				truth_classes_count[truth[j].id]++;
			}

			// difficult
			box_label *truth_dif = NULL;
			int num_labels_dif = 0;
			if (paths_dif)
			{
				char *path_dif = paths_dif[image_index];

				char labelpath_dif[4096];
				find_replace(path_dif, "images", "labels", labelpath_dif);
				find_replace(labelpath_dif, "JPEGImages", "labels", labelpath_dif);
				find_replace(labelpath_dif, ".jpg", ".txt", labelpath_dif);
				find_replace(labelpath_dif, ".JPEG", ".txt", labelpath_dif);
				find_replace(labelpath_dif, ".png", ".txt", labelpath_dif);				
				truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
			}

			for (i = 0; i < (l.w*l.h*l.n); ++i) {

				int class_id;
				for (class_id = 0; class_id < classes; ++class_id) {
					float prob = probs[i][class_id];
					if (prob > 0) {
						detections_count++;
						detections = realloc(detections, detections_count * sizeof(box_prob));
						detections[detections_count - 1].b = boxes[i];
						detections[detections_count - 1].p = prob;
						detections[detections_count - 1].image_index = image_index;
						detections[detections_count - 1].class_id = class_id;
						detections[detections_count - 1].truth_flag = 0;
						detections[detections_count - 1].unique_truth_index = -1;

						int truth_index = -1;
						float max_iou = 0;
						for (j = 0; j < num_labels; ++j)
						{
							box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
							//printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n", 
							//	box_iou(boxes[i], t), prob, class_id, truth[j].id);
							float current_iou = box_iou(boxes[i], t);
							if (current_iou > iou_thresh && class_id == truth[j].id) {
								if (current_iou > max_iou) {
									max_iou = current_iou;
									truth_index = unique_truth_count + j;
								}
							}
						}

						// best IoU
						if (truth_index > -1) {
							detections[detections_count - 1].truth_flag = 1;
							detections[detections_count - 1].unique_truth_index = truth_index;
						}
						else {
							// if object is difficult then remove detection
							for (j = 0; j < num_labels_dif; ++j) {
								box t = { truth_dif[j].x, truth_dif[j].y, truth_dif[j].w, truth_dif[j].h };
								float current_iou = box_iou(boxes[i], t);
								if (current_iou > iou_thresh && class_id == truth_dif[j].id) {
									--detections_count;
									break;
								}
							}
						}

						// calc avg IoU, true-positives, false-positives for required Threshold
						if (prob > thresh_calc_avg_iou) {
							if (truth_index > -1) {
								avg_iou += max_iou;
								++tp_for_thresh;
							}
							else
								fp_for_thresh++;
						}
					}
				}
			}
			
			unique_truth_count += num_labels;

			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}

	avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

	
	// SORT(detections)
	qsort(detections, detections_count, sizeof(box_prob), detections_comparator);
	
	typedef struct {
		double precision;
		double recall;
		int tp, fp, fn;
	} pr_t;

	// for PR-curve
	pr_t **pr = calloc(classes, sizeof(pr_t*));
	for (i = 0; i < classes; ++i) {
		pr[i] = calloc(detections_count, sizeof(pr_t));
	}
	printf("detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


	int *truth_flags = calloc(unique_truth_count, sizeof(int));

	int rank;
	for (rank = 0; rank < detections_count; ++rank) {
		if(rank % 100 == 0)
			printf(" rank = %d of ranks = %d \r", rank, detections_count);

		if (rank > 0) {
			int class_id;
			for (class_id = 0; class_id < classes; ++class_id) {
				pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
				pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
			}
		}

		box_prob d = detections[rank];
		// if (detected && isn't detected before)
		if (d.truth_flag == 1) {
			if (truth_flags[d.unique_truth_index] == 0) 
			{
				truth_flags[d.unique_truth_index] = 1;
				pr[d.class_id][rank].tp++;	// true-positive
			}
		}
		else {
			pr[d.class_id][rank].fp++;	// false-positive
		}

		for (i = 0; i < classes; ++i) 
		{
			const int tp = pr[i][rank].tp;
			const int fp = pr[i][rank].fp;
			const int fn = truth_classes_count[i] - tp;	// false-negative = objects - true-positive
			pr[i][rank].fn = fn;

			if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
			else pr[i][rank].precision = 0;

			if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
			else pr[i][rank].recall = 0;
		}
	}

	free(truth_flags);
	
	
	double mean_average_precision = 0;

	for (i = 0; i < classes; ++i) {
		double avg_precision = 0;
		int point;
		for (point = 0; point < 11; ++point) {
			double cur_recall = point * 0.1;
			double cur_precision = 0;
			for (rank = 0; rank < detections_count; ++rank)
			{
				if (pr[i][rank].recall >= cur_recall) {	// > or >=
					if (pr[i][rank].precision > cur_precision) {
						cur_precision = pr[i][rank].precision;
					}
				}
			}
			//printf("point = %d, cur_recall = %.4f, cur_precision = %.4f \n", point, cur_recall, cur_precision);

			avg_precision += cur_precision;
		}
		avg_precision = avg_precision / 11;
		printf("class = %d, name = %s, \t ap = %2.2f %% \n", i, names[i], avg_precision*100);
		mean_average_precision += avg_precision;
	}
	
	printf(" for thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n", 
		thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

	mean_average_precision = mean_average_precision / classes;
	printf("\n mean average precision (mAP) = %f, or %2.2f %% \n", mean_average_precision, mean_average_precision*100);


	for (i = 0; i < classes; ++i) {
		free(pr[i]);
	}
	free(pr);
	free(detections);
	free(truth_classes_count);

	fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
			if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
        if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
        save_image(im, "predictions");
        show_image(im, "predictions");

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void run_detector(int argc, char **argv)
{
	int http_stream_port = find_int_arg(argc, argv, "-http_port", -1);
	char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .24);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
	if(weights)
		if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights);
	else if(0==strcmp(argv[2], "map")) validate_detector_map(datacfg, cfg, weights);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
		if(filename)
			if (filename[strlen(filename) - 1] == 0x0d) filename[strlen(filename) - 1] = 0;
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, out_filename, http_stream_port);
    }
}
