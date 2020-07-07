#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "darknet.h"
#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#ifdef OPENCV

#include "http_stream.h"

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static int nboxes = 0;
static detection *dets = NULL;

static network net;
static image in_s ;
static image det_s;

static cap_cv *cap;
static float fps = 0;
static float demo_thresh = 0;
static int demo_ext_output = 0;
static long long int frame_id = 0;
static int demo_json_port = -1;


static int avg_frames;
static int demo_index = 0;
static mat_cv** cv_images;

mat_cv* in_img;
mat_cv* det_img;
mat_cv* show_img;

static volatile int flag_exit;
static int letter_box = 0;

static const int thread_wait_ms = 1;
static volatile int run_fetch_in_thread = 0;
static volatile int run_detect_in_thread = 0;


void *fetch_in_thread(void *ptr)
{
    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_fetch_in_thread)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }
        int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream
        if (letter_box)
            in_s = get_image_from_stream_letterbox(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
        else
            in_s = get_image_from_stream_resize(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
        if (!in_s.data) {
            printf("Stream closed.\n");
            custom_atomic_store_int(&flag_exit, 1);
            custom_atomic_store_int(&run_fetch_in_thread, 0);
            //exit(EXIT_FAILURE);
            return 0;
        }
        //in_s = resize_image(in, net.w, net.h);

        custom_atomic_store_int(&run_fetch_in_thread, 0);
    }
    return 0;
}

void *fetch_in_thread_sync(void *ptr)
{
    custom_atomic_store_int(&run_fetch_in_thread, 1);
    while (custom_atomic_load_int(&run_fetch_in_thread)) this_thread_sleep_for(thread_wait_ms);
    return 0;
}

void *detect_in_thread(void *ptr)
{
    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_detect_in_thread)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }

        layer l = net.layers[net.n - 1];
        float *X = det_s.data;
        float *prediction = network_predict(net, X);

        cv_images[demo_index] = det_img;
        det_img = cv_images[(demo_index + avg_frames / 2 + 1) % avg_frames];
        demo_index = (demo_index + 1) % avg_frames;

        if (letter_box)
            dets = get_network_boxes(&net, get_width_mat(in_img), get_height_mat(in_img), demo_thresh, demo_thresh, 0, 1, &nboxes, 1); // letter box
        else
            dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized

        custom_atomic_store_int(&run_detect_in_thread, 0);
    }

    return 0;
}

void *detect_in_thread_sync(void *ptr)
{
    custom_atomic_store_int(&run_detect_in_thread, 1);
    while (custom_atomic_load_int(&run_detect_in_thread)) this_thread_sleep_for(thread_wait_ms);
    return 0;
}

double get_wall_time()
{
    struct timeval walltime;
    if (gettimeofday(&walltime, NULL)) {
        return 0;
    }
    return (double)walltime.tv_sec + (double)walltime.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes, int avgframes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int dontdraw_bbox, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
    int benchmark, int benchmark_layers)
{
    if (avgframes < 1) avgframes = 1;
    avg_frames = avgframes;
    letter_box = letter_box_in;
    in_img = det_img = show_img = NULL;
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_ext_output = ext_output;
    demo_json_port = json_port;
    printf("Demo\n");
    net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = get_capture_video_stream(filename);
    }else{
        printf("Webcam index: %d\n", cam_index);
        cap = get_capture_webcam(cam_index);
    }

    if (!cap) {
#ifdef WIN32
        printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
        error("Couldn't connect to webcam.\n");
    }

    layer l = net.layers[net.n-1];
    int j;

    cv_images = (mat_cv**)xcalloc(avg_frames, sizeof(mat_cv));

    int i;
    for (i = 0; i < net.n; ++i) {
        layer lc = net.layers[i];
        if (lc.type == YOLO) {
            lc.mean_alpha = 1.0 / avg_frames;
            l = lc;
        }
    }

    if (l.classes != demo_classes) {
        printf("\n Parameters don't match: in cfg-file classes=%d, in data-file classes=%d \n", l.classes, demo_classes);
        getchar();
        exit(0);
    }

    flag_exit = 0;

    custom_thread_t fetch_thread = NULL;
    custom_thread_t detect_thread = NULL;
    if (custom_create_thread(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
    if (custom_create_thread(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

    fetch_in_thread_sync(0); //fetch_in_thread(0);
    det_img = in_img;
    det_s = in_s;

    fetch_in_thread_sync(0); //fetch_in_thread(0);
    detect_in_thread_sync(0); //fetch_in_thread(0);
    det_img = in_img;
    det_s = in_s;

    for (j = 0; j < avg_frames / 2; ++j) {
        free_detections(dets, nboxes);
        fetch_in_thread_sync(0); //fetch_in_thread(0);
        detect_in_thread_sync(0); //fetch_in_thread(0);
        det_img = in_img;
        det_s = in_s;
    }

    int count = 0;
    if(!prefix && !dont_show){
        int full_screen = 0;
        create_window_cv("Demo", full_screen, 1352, 1013);
    }


    write_cv* output_video_writer = NULL;
    if (out_filename && !flag_exit)
    {
        int src_fps = 25;
        src_fps = get_stream_fps_cpp_cv(cap);
        output_video_writer =
            create_video_writer(out_filename, 'D', 'I', 'V', 'X', src_fps, get_width_mat(det_img), get_height_mat(det_img), 1);

        //'H', '2', '6', '4'
        //'D', 'I', 'V', 'X'
        //'M', 'J', 'P', 'G'
        //'M', 'P', '4', 'V'
        //'M', 'P', '4', '2'
        //'X', 'V', 'I', 'D'
        //'W', 'M', 'V', '2'
    }

    int send_http_post_once = 0;
    const double start_time_lim = get_time_point();
    double before = get_time_point();
    double start_time = get_time_point();
    float avg_fps = 0;
    int frame_counter = 0;
    int global_frame_counter = 0;

    while(1){
        ++count;
        {
            const float nms = .45;    // 0.4F
            int local_nboxes = nboxes;
            detection *local_dets = dets;
            this_thread_yield();

            if (!benchmark) custom_atomic_store_int(&run_fetch_in_thread, 1); // if (custom_create_thread(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            custom_atomic_store_int(&run_detect_in_thread, 1); // if (custom_create_thread(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            //if (nms) do_nms_obj(local_dets, local_nboxes, l.classes, nms);    // bad results
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(local_dets, local_nboxes, l.classes, nms);
                else diounms_sort(local_dets, local_nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

            if (l.embedding_size) set_track_id(local_dets, local_nboxes, demo_thresh, l.sim_thresh, l.track_ciou_norm, l.track_history_size, l.dets_for_track, l.dets_for_show);

            //printf("\033[2J");
            //printf("\033[1;1H");
            //printf("\nFPS:%.1f\n", fps);
            printf("Objects:\n\n");

            ++frame_id;
            if (demo_json_port > 0) {
                int timeout = 400000;
                send_json(local_dets, local_nboxes, l.classes, demo_names, frame_id, demo_json_port, timeout);
            }

            //char *http_post_server = "webhook.site/898bbd9b-0ddd-49cf-b81d-1f56be98d870";
            if (http_post_host && !send_http_post_once) {
                int timeout = 3;            // 3 seconds
                int http_post_port = 80;    // 443 https, 80 http
                if (send_http_post_request(http_post_host, http_post_port, filename,
                    local_dets, nboxes, classes, names, frame_id, ext_output, timeout))
                {
                    if (time_limit_sec > 0) send_http_post_once = 1;
                }
            }

            if (!benchmark && !dontdraw_bbox) draw_detections_cv_v3(show_img, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
            free_detections(local_dets, local_nboxes);

            printf("\nFPS:%.1f \t AVG_FPS:%.1f\n", fps, avg_fps);

            if(!prefix){
                if (!dont_show) {
                    const int each_frame = max_val_cmp(1, avg_fps / 60);
                    if(global_frame_counter % each_frame == 0) show_image_mat(show_img, "Demo");
                    int c = wait_key_cv(1);
                    if (c == 10) {
                        if (frame_skip == 0) frame_skip = 60;
                        else if (frame_skip == 4) frame_skip = 0;
                        else if (frame_skip == 60) frame_skip = 4;
                        else frame_skip = 0;
                    }
                    else if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
                    {
                        flag_exit = 1;
                    }
                }
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d.jpg", prefix, count);
                if(show_img) save_cv_jpg(show_img, buff);
            }

            // if you run it with param -mjpeg_port 8090  then open URL in your web-browser: http://localhost:8090
            if (mjpeg_port > 0 && show_img) {
                int port = mjpeg_port;
                int timeout = 400000;
                int jpeg_quality = 40;    // 1 - 100
                send_mjpeg(show_img, port, timeout, jpeg_quality);
            }

            // save video file
            if (output_video_writer && show_img) {
                write_frame_cv(output_video_writer, show_img);
                printf("\n cvWriteFrame \n");
            }

            while (custom_atomic_load_int(&run_detect_in_thread)) {
                if(avg_fps > 180) this_thread_yield();
                else this_thread_sleep_for(thread_wait_ms);   // custom_join(detect_thread, 0);
            }
            if (!benchmark) {
                while (custom_atomic_load_int(&run_fetch_in_thread)) {
                    if (avg_fps > 180) this_thread_yield();
                    else this_thread_sleep_for(thread_wait_ms);   // custom_join(fetch_thread, 0);
                }
                free_image(det_s);
            }

            if (time_limit_sec > 0 && (get_time_point() - start_time_lim)/1000000 > time_limit_sec) {
                printf(" start_time_lim = %f, get_time_point() = %f, time spent = %f \n", start_time_lim, get_time_point(), get_time_point() - start_time_lim);
                break;
            }

            if (flag_exit == 1) break;

            if(delay == 0){
                if(!benchmark) release_mat(&show_img);
                show_img = det_img;
            }
            det_img = in_img;
            det_s = in_s;
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            //double after = get_wall_time();
            //float curr = 1./(after - before);
            double after = get_time_point();    // more accurate time measurements
            float curr = 1000000. / (after - before);
            fps = fps*0.9 + curr*0.1;
            before = after;

            float spent_time = (get_time_point() - start_time) / 1000000;
            frame_counter++;
            global_frame_counter++;
            if (spent_time >= 3.0f) {
                //printf(" spent_time = %f \n", spent_time);
                avg_fps = frame_counter / spent_time;
                frame_counter = 0;
                start_time = get_time_point();
            }
        }
    }
    printf("input video stream closed. \n");
    if (output_video_writer) {
        release_video_writer(&output_video_writer);
        printf("output_video_writer closed. \n");
    }

    this_thread_sleep_for(thread_wait_ms);

    custom_join(detect_thread, 0);
    custom_join(fetch_thread, 0);

    // free memory
    free_image(in_s);
    free_detections(dets, nboxes);

    demo_index = (avg_frames + demo_index - 1) % avg_frames;
    for (j = 0; j < avg_frames; ++j) {
            release_mat(&cv_images[j]);
    }
    free(cv_images);

    free_ptrs((void **)names, net.layers[net.n - 1].classes);

    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);
    free_network(net);
    //cudaProfilerStop();
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes, int avgframes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int dontdraw_bbox, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
    int benchmark, int benchmark_layers)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
