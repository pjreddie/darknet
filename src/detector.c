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
//#include "opencv2/core/core.hpp"
#include "opencv2/core/version.hpp"
#include "opencv2/imgproc/imgproc_c.h"

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

IplImage* draw_train_chart(float max_img_loss, int max_batches, int number_of_lines, int img_size);
void draw_train_loss(IplImage* img, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches);
#endif	// OPENCV

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show)
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
	int iter_save;
	iter_save = get_current_batch(net);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.flip = net.flip;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
	args.small_object = net.small_object;
    args.d = &buffer;
    args.type = DETECTION_DATA;
	args.threads = 16;	// 64

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

#ifdef OPENCV
	args.threads = 3;
	IplImage* img = NULL;
	float max_img_loss = 5;
	int number_of_lines = 100;
	int img_size = 1000;
	if (!dont_show)
		img = draw_train_chart(max_img_loss, net.max_batches, number_of_lines, img_size);
#endif	//OPENCV

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
		if(l.random && count++%10 == 0){
            printf("Resizing\n");
			int dim = (rand() % 12 + (init_w/32 - 5)) * 32;	// +-160
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
        time=what_time_is_it_now();
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

        printf("Loaded: %lf seconds\n", (what_time_is_it_now()-time));

        time=what_time_is_it_now();
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
        if (avg_loss < 0 || avg_loss != avg_loss) avg_loss = loss;	// if(-inf or nan)
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("\n %d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), (what_time_is_it_now()-time), i*imgs);

#ifdef OPENCV
		if(!dont_show)
			draw_train_loss(img, img_size, avg_loss, max_img_loss, i, net.max_batches);
#endif	// OPENCV

		//if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
		//if (i % 100 == 0) {
		if(i >= (iter_save + 100)) {
			iter_save = i;
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

	//cvReleaseImage(&img);
	//cvDestroyAllWindows();
}


static int get_coco_image_id(char *filename)
{
	char *p = strrchr(filename, '/');
	char *c = strrchr(filename, '_');
	if (c) p = c;
	return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
	int i, j;
	int image_id = get_coco_image_id(image_path);
	for (i = 0; i < num_boxes; ++i) {
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		float bx = xmin;
		float by = ymin;
		float bw = xmax - xmin;
		float bh = ymax - ymin;

		for (j = 0; j < classes; ++j) {
			if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
		}
	}
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
	int i, j;
	for (i = 0; i < total; ++i) {
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

		if (xmin < 1) xmin = 1;
		if (ymin < 1) ymin = 1;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j) {
			if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
				xmin, ymin, xmax, ymax);
		}
	}
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
	int i, j;
	for (i = 0; i < total; ++i) {
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j) {
			int class = j;
			if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[class],
				xmin, ymin, xmax, ymax);
		}
	}
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
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

	network net = parse_network_cfg_custom(cfgfile, 1);	// set batch=1
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	//set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	list *plist = get_paths(valid_images);
	char **paths = (char **)list_to_array(plist);

	layer l = net.layers[net.n - 1];
	int classes = l.classes;

	char buff[1024];
	char *type = option_find_str(options, "eval", "voc");
	FILE *fp = 0;
	FILE **fps = 0;
	int coco = 0;
	int imagenet = 0;
	if (0 == strcmp(type, "coco")) {
		if (!outfile) outfile = "coco_results";
		snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
		fp = fopen(buff, "w");
		fprintf(fp, "[\n");
		coco = 1;
	}
	else if (0 == strcmp(type, "imagenet")) {
		if (!outfile) outfile = "imagenet-detection";
		snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
		fp = fopen(buff, "w");
		imagenet = 1;
		classes = 200;
	}
	else {
		if (!outfile) outfile = "comp4_det_test_";
		fps = calloc(classes, sizeof(FILE *));
		for (j = 0; j < classes; ++j) {
			snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
			fps[j] = fopen(buff, "w");
		}
	}


	int m = plist->size;
	int i = 0;
	int t;

	float thresh = .005;
	float nms = .45;

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
	//args.type = LETTERBOX_DATA;

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
			char *path = paths[i + t - nthreads];
			char *id = basecfg(path);
			float *X = val_resized[t].data;
			network_predict(net, X);
			int w = val[t].w;
			int h = val[t].h;
			int nboxes = 0;
			int letterbox = (args.type == LETTERBOX_DATA);
			detection *dets = get_network_boxes(&net, w, h, thresh, .5, map, 0, &nboxes, letterbox);
			if (nms) do_nms_sort(dets, nboxes, classes, nms);
			if (coco) {
				print_cocos(fp, path, dets, nboxes, classes, w, h);
			}
			else if (imagenet) {
				print_imagenet_detections(fp, i + t - nthreads + 1, dets, nboxes, classes, w, h);
			}
			else {
				print_detector_detections(fps, id, dets, nboxes, classes, w, h);
			}
			free_detections(dets, nboxes);
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}
	for (j = 0; j < classes; ++j) {
		if (fps) fclose(fps[j]);
	}
	if (coco) {
		fseek(fp, -2, SEEK_CUR);
		fprintf(fp, "\n]\n");
		fclose(fp);
	}
	fprintf(stderr, "Total Detection Time: %f Seconds\n", time(0) - start);
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile)
{
	network net = parse_network_cfg_custom(cfgfile, 1);	// set batch=1
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	//set_batch_network(&net, 1);
	fuse_conv_batchnorm(net);
	srand(time(0));

	//list *plist = get_paths("data/coco_val_5k.list");
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.txt");
	list *plist = get_paths(valid_images);
	char **paths = (char **)list_to_array(plist);

	layer l = net.layers[net.n - 1];

	int j, k;

	int m = plist->size;
	int i = 0;

	float thresh = .001;
	float iou_thresh = .5;
	float nms = .4;

	int total = 0;
	int correct = 0;
	int proposals = 0;
	float avg_iou = 0;

	for (i = 0; i < m; ++i) {
		char *path = paths[i];
		image orig = load_image_color(path, 0, 0);
		image sized = resize_image(orig, net.w, net.h);
		char *id = basecfg(path);
		network_predict(net, sized.data);
		int nboxes = 0;
		int letterbox = 0;
		detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes, letterbox);
		if (nms) do_nms_obj(dets, nboxes, 1, nms);

		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".png", ".txt", labelpath);
		find_replace(labelpath, ".bmp", ".txt", labelpath);
		find_replace(labelpath, ".JPG", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		for (k = 0; k < nboxes; ++k) {
			if (dets[k].objectness > thresh) {
				++proposals;
			}
		}
		for (j = 0; j < num_labels; ++j) {
			++total;
			box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
			float best_iou = 0;
			for (k = 0; k < nboxes; ++k) {
				float iou = box_iou(dets[k].bbox, t);
				if (dets[k].objectness > thresh && iou > best_iou) {
					best_iou = iou;
				}
			}
			avg_iou += best_iou;
			if (best_iou > iou_thresh) {
				++correct;
			}
		}
		//fprintf(stderr, " %s - %s - ", paths[i], labelpath);
		fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
		free(id);
		free_image(orig);
		free_image(sized);
	}
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

void validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou)
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

	network net = parse_network_cfg_custom(cfgfile, 1);	// set batch=1
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	//set_batch_network(&net, 1);
	fuse_conv_batchnorm(net);
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
	//args.type = LETTERBOX_DATA;

	//const float thresh_calc_avg_iou = 0.24;
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

			int nboxes = 0;
			int letterbox = (args.type == LETTERBOX_DATA);
			float hier_thresh = 0;
			detection *dets = get_network_boxes(&net, 1, 1, thresh, hier_thresh, 0, 0, &nboxes, letterbox);
			//detection *dets = get_network_boxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0, 1, &nboxes, letterbox); // for letterbox=1
			if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

			char labelpath[4096];
			find_replace(path, "images", "labels", labelpath);
			find_replace(labelpath, "JPEGImages", "labels", labelpath);
			find_replace(labelpath, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			find_replace(labelpath, ".bmp", ".txt", labelpath);
			find_replace(labelpath, ".JPG", ".txt", labelpath);
			find_replace(labelpath, ".JPEG", ".txt", labelpath);
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

			const int checkpoint_detections_count = detections_count;

			for (i = 0; i < nboxes; ++i) {

				int class_id;
				for (class_id = 0; class_id < classes; ++class_id) {
					float prob = dets[i].prob[class_id];
					if (prob > 0) {
						detections_count++;
						detections = realloc(detections, detections_count * sizeof(box_prob));
						detections[detections_count - 1].b = dets[i].bbox;
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
							//	box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
							float current_iou = box_iou(dets[i].bbox, t);
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
								float current_iou = box_iou(dets[i].bbox, t);
								if (current_iou > iou_thresh && class_id == truth_dif[j].id) {
									--detections_count;
									break;
								}
							}
						}

						// calc avg IoU, true-positives, false-positives for required Threshold
						if (prob > thresh_calc_avg_iou) {
							int z, found = 0;
							for (z = checkpoint_detections_count; z < detections_count-1; ++z)
								if (detections[z].unique_truth_index == truth_index) {
									found = 1; break;
								}

							if(truth_index > -1 && found == 0) {
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

			free_detections(dets, nboxes);
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}

	if((tp_for_thresh + fp_for_thresh) > 0)
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
			//printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

			avg_precision += cur_precision;
		}
		avg_precision = avg_precision / 11;
		printf("class_id = %d, name = %s, \t ap = %2.2f %% \n", i, names[i], avg_precision*100);
		mean_average_precision += avg_precision;
	}
	
	const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
	const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
	const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
	printf(" for thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
		thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

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

#ifdef OPENCV
typedef struct {
	float w, h;
} anchors_t;

int anchors_comparator(const void *pa, const void *pb)
{
	anchors_t a = *(anchors_t *)pa;
	anchors_t b = *(anchors_t *)pb;
	float diff = b.w*b.h - a.w*a.h;
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
	printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);
	if (width < 0 || height < 0) {
		printf("Usage: darknet detector calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416 \n");
		printf("Error: set width and height \n");
		return;
	}

	//float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
	float *rel_width_height_array = calloc(1000, sizeof(float));

	list *options = read_data_cfg(datacfg);
	char *train_images = option_find_str(options, "train", "data/train.list");
	list *plist = get_paths(train_images);
	int number_of_images = plist->size;
	char **paths = (char **)list_to_array(plist);

	int number_of_boxes = 0;
	printf(" read labels from %d images \n", number_of_images);

	int i, j;
	for (i = 0; i < number_of_images; ++i) {
		char *path = paths[i];
		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".png", ".txt", labelpath);
		find_replace(labelpath, ".bmp", ".txt", labelpath);
		find_replace(labelpath, ".JPG", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);
		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		//printf(" new path: %s \n", labelpath);
		for (j = 0; j < num_labels; ++j)
		{
			number_of_boxes++;
			rel_width_height_array = realloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));
			rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
			rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
			printf("\r loaded \t image: %d \t box: %d", i+1, number_of_boxes);
		}
	}
	printf("\n all loaded. \n");

	CvMat* points = cvCreateMat(number_of_boxes, 2, CV_32FC1);
	CvMat* centers = cvCreateMat(num_of_clusters, 2, CV_32FC1);
	CvMat* labels = cvCreateMat(number_of_boxes, 1, CV_32SC1);

	for (i = 0; i < number_of_boxes; ++i) {
		points->data.fl[i * 2] = rel_width_height_array[i * 2];
		points->data.fl[i * 2 + 1] = rel_width_height_array[i * 2 + 1];
		//cvSet1D(points, i * 2, cvScalar(rel_width_height_array[i * 2], 0, 0, 0));
		//cvSet1D(points, i * 2 + 1, cvScalar(rel_width_height_array[i * 2 + 1], 0, 0, 0));
	}


	const int attemps = 10;
	double compactness;

	enum {
		KMEANS_RANDOM_CENTERS = 0,
		KMEANS_USE_INITIAL_LABELS = 1,
		KMEANS_PP_CENTERS = 2
	};
	
	printf("\n calculating k-means++ ...");
	// Should be used: distance(box, centroid) = 1 - IoU(box, centroid)
	cvKMeans2(points, num_of_clusters, labels, 
		cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10000, 0), attemps, 
		0, KMEANS_PP_CENTERS,
		centers, &compactness);

	// sort anchors
	qsort(centers->data.fl, num_of_clusters, 2*sizeof(float), anchors_comparator);

	//orig 2.0 anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
	//float orig_anch[] = { 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52 };
	// worse than ours (even for 19x19 final size - for input size 608x608)

	//orig anchors = 1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892, 9.47112,4.84053, 11.2364,10.0071
	//float orig_anch[] = { 1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892, 9.47112,4.84053, 11.2364,10.0071 };
	// orig (IoU=59.90%) better than ours (59.75%)

	//gen_anchors.py = 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66
	//float orig_anch[] = { 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66 };

	// ours: anchors = 9.3813,6.0095, 3.3999,5.3505, 10.9476,11.1992, 5.0161,9.8314, 1.5003,2.1595
	//float orig_anch[] = { 9.3813,6.0095, 3.3999,5.3505, 10.9476,11.1992, 5.0161,9.8314, 1.5003,2.1595 };
	//for (i = 0; i < num_of_clusters * 2; ++i) centers->data.fl[i] = orig_anch[i];
	
	//for (i = 0; i < number_of_boxes; ++i)
	//	printf("%2.2f,%2.2f, ", points->data.fl[i * 2], points->data.fl[i * 2 + 1]);

	float avg_iou = 0;
	for (i = 0; i < number_of_boxes; ++i) {
		float box_w = points->data.fl[i * 2];
		float box_h = points->data.fl[i * 2 + 1];
		//int cluster_idx = labels->data.i[i];		
		int cluster_idx = 0;
		float min_dist = FLT_MAX;
		for (j = 0; j < num_of_clusters; ++j) {
			float anchor_w = centers->data.fl[j * 2];
			float anchor_h = centers->data.fl[j * 2 + 1];
			float w_diff = anchor_w - box_w;
			float h_diff = anchor_h - box_h;
			float distance = sqrt(w_diff*w_diff + h_diff*h_diff);
			if (distance < min_dist) min_dist = distance, cluster_idx = j;
		}
		
		float anchor_w = centers->data.fl[cluster_idx * 2];
		float anchor_h = centers->data.fl[cluster_idx * 2 + 1];
		float min_w = (box_w < anchor_w) ? box_w : anchor_w;
		float min_h = (box_h < anchor_h) ? box_h : anchor_h;
		float box_intersect = min_w*min_h;
		float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
		float iou = box_intersect / box_union;
		if (iou > 1 || iou < 0) {
			printf(" i = %d, box_w = %d, box_h = %d, anchor_w = %d, anchor_h = %d, iou = %f \n",
				i, box_w, box_h, anchor_w, anchor_h, iou);
		}
		else avg_iou += iou;
	}
	avg_iou = 100 * avg_iou / number_of_boxes;
	printf("\n avg IoU = %2.2f %% \n", avg_iou);

	char buff[1024];
	FILE* fw = fopen("anchors.txt", "wb");
	printf("\nSaving anchors to the file: anchors.txt \n");
	printf("anchors = ");
	for (i = 0; i < num_of_clusters; ++i) {
		sprintf(buff, "%2.4f,%2.4f", centers->data.fl[i * 2], centers->data.fl[i * 2 + 1]);
		printf("%s", buff);
		fwrite(buff, sizeof(char), strlen(buff), fw);
		if (i + 1 < num_of_clusters) {
			fwrite(", ", sizeof(char), 2, fw);
			printf(", ");
		}
	}
	printf("\n");
	fclose(fw);

	if (show) {
		size_t img_size = 700;
		IplImage* img = cvCreateImage(cvSize(img_size, img_size), 8, 3);
		cvZero(img);
		for (j = 0; j < num_of_clusters; ++j) {
			CvPoint pt1, pt2;
			pt1.x = pt1.y = 0;
			pt2.x = centers->data.fl[j * 2] * img_size / width;
			pt2.y = centers->data.fl[j * 2 + 1] * img_size / height;
			cvRectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
		}

		for (i = 0; i < number_of_boxes; ++i) {
			CvPoint pt;
			pt.x = points->data.fl[i * 2] * img_size / width;
			pt.y = points->data.fl[i * 2 + 1] * img_size / height;
			int cluster_idx = labels->data.i[i];
			int red_id = (cluster_idx * (uint64_t)123 + 55) % 255;
			int green_id = (cluster_idx * (uint64_t)321 + 33) % 255;
			int blue_id = (cluster_idx * (uint64_t)11 + 99) % 255;
			cvCircle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED, 8, 0);
			//if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
		}
		cvShowImage("clusters", img);
		cvWaitKey(0);
		cvReleaseImage(&img);
		cvDestroyAllWindows();
	}

	free(rel_width_height_array);
	cvReleaseMat(&points);
	cvReleaseMat(&centers);
	cvReleaseMat(&labels);
}
#else
void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show) {
	printf(" k-means++ can't be used without OpenCV, because there is used cvKMeans2 implementation \n");
}
#endif // OPENCV

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, int dont_show)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1); // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
	fuse_conv_batchnorm(net);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.45;	// 0.4F
    while(1){
        if(filename){
            strncpy(input, filename, 256);
			if(strlen(input) > 0)
				if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
		int letterbox = 0;
        //image sized = resize_image(im, net.w, net.h);
		image sized = letterbox_image(im, net.w, net.h); letterbox = 1;
        layer l = net.layers[net.n-1];

        //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        //float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

        float *X = sized.data;
        time= what_time_is_it_now();
        network_predict(net, X);
		//network_predict_image(&net, im); letterbox = 1;
        printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));
        //get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
		// if (nms) do_nms_sort_v2(boxes, probs, l.w*l.h*l.n, l.classes, nms);
		//draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
		int nboxes = 0;
		detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
		if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
		draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes);
		free_detections(dets, nboxes);
        save_image(im, "predictions");
		if (!dont_show) {
			show_image(im, "predictions");
		}

        free_image(im);
        free_image(sized);
        //free(boxes);
        //free_ptrs((void **)probs, l.w*l.h*l.n);
#ifdef OPENCV
		if (!dont_show) {
			cvWaitKey(0);
			cvDestroyAllWindows();
		}
#endif
        if (filename) break;
    }
}

void run_detector(int argc, char **argv)
{
	int dont_show = find_arg(argc, argv, "-dont_show");
	int show = find_arg(argc, argv, "-show");
	int http_stream_port = find_int_arg(argc, argv, "-http_port", -1);
	char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
	char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);	// 0.24
	float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
	int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
	int width = find_int_arg(argc, argv, "-width", -1);
	int height = find_int_arg(argc, argv, "-height", -1);
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
		if(strlen(weights) > 0)
			if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear, dont_show);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights);
	else if(0==strcmp(argv[2], "map")) validate_detector_map(datacfg, cfg, weights, thresh);
	else if(0==strcmp(argv[2], "calc_anchors")) calc_anchors(datacfg, num_of_clusters, width, height, show);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
		if(filename)
			if(strlen(filename) > 0)
				if (filename[strlen(filename) - 1] == 0x0d) filename[strlen(filename) - 1] = 0;
        demo(cfg, weights, thresh, hier_thresh, cam_index, filename, names, classes, frame_skip, prefix, out_filename,
			http_stream_port, dont_show);
    }
	else printf(" There isn't such command: %s", argv[2]);
}
