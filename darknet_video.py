import random
import os
import cv2
import time
import darknet
import argparse
import threading
import queue


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action="store_true",
                        help="window inference display. For headless systems")
    parser.add_argument("--ext_output", action="store_true",
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns strings although webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(output_video, size, fps):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(output_video, fourcc, fps, size)


def convert2relative(bbox, preproc_h, preproc_w):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    return x / preproc_w, y / preproc_h, w / preproc_w, h / preproc_h


def convert2original(image, bbox, preproc_h, preproc_w):
    x, y, w, h = convert2relative(bbox, preproc_h, preproc_w)

    image_h, image_w, __ = image.shape

    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


# @TODO - cfati: Unused
def convert4cropping(image, bbox, preproc_h, preproc_w):
    x, y, w, h = convert2relative(bbox, preproc_h, preproc_w)

    image_h, image_w, __ = image.shape

    orig_left = int((x - w / 2.) * image_w)
    orig_right = int((x + w / 2.) * image_w)
    orig_top = int((y - h / 2.) * image_h)
    orig_bottom = int((y + h / 2.) * image_h)

    if orig_left < 0:
        orig_left = 0
    if orig_right > image_w - 1:
        orig_right = image_w - 1
    if orig_top < 0:
        orig_top = 0
    if orig_bottom > image_h - 1:
        orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(stop_flag, input_path, raw_frame_queue, preprocessed_frame_queue, preproc_h, preproc_w):
    cap = cv2.VideoCapture(input_path)
    while cap.isOpened() and not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (preproc_w, preproc_h),
                                   interpolation=cv2.INTER_LINEAR)
        raw_frame_queue.put(frame)
        img_for_detect = darknet.make_image(preproc_w, preproc_h, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        preprocessed_frame_queue.put(img_for_detect)
    stop_flag.set()
    cap.release()


def inference(stop_flag, preprocessed_frame_queue, detections_queue, fps_queue,
              network, class_names, threshold):
    while not stop_flag.is_set():
        darknet_image = preprocessed_frame_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=threshold)
        fps = 1 / (time.time() - prev_time)
        detections_queue.put(detections)
        fps_queue.put(int(fps))
        print("FPS: {:.2f}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)


def drawing(stop_flag, input_video_fps, queues, preproc_h, preproc_w, vid_h, vid_w):
    random.seed(3)  # deterministic bbox colors
    raw_frame_queue, preprocessed_frame_queue, detections_queue, fps_queue = queues
    video = set_saved_video(args.out_filename, (vid_w, vid_h), input_video_fps)
    fps = 1
    while not stop_flag.is_set():
        frame = raw_frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox, preproc_h, preproc_w)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            if not args.dont_show:
                cv2.imshow("Inference", image)
            if args.out_filename is not None:
                video.write(image)
            if cv2.waitKey(fps) == 27:
                break
    stop_flag.set()
    video.release()
    cv2.destroyAllWindows()
    timeout = 1 / (fps if fps > 0 else 0.5)
    for q in (preprocessed_frame_queue, detections_queue, fps_queue):
        try:
            q.get(block=True, timeout=timeout)
        except queue.Empty:
            pass


if __name__ == "__main__":
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1)
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)  # Open video twice :(
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    del cap

    ExecUnit = threading.Thread
    Queue = queue.Queue
    stop_flag = threading.Event()

    raw_frame_queue = Queue()
    preprocessed_frame_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    exec_units = (
        ExecUnit(target=video_capture, args=(stop_flag, input_path, raw_frame_queue, preprocessed_frame_queue,
                                             darknet_height, darknet_width)),
        ExecUnit(target=inference, args=(stop_flag, preprocessed_frame_queue, detections_queue, fps_queue,
                                         network, class_names, args.thresh)),
        ExecUnit(target=drawing, args=(stop_flag, video_fps,
                                       (raw_frame_queue, preprocessed_frame_queue, detections_queue, fps_queue),
                                       darknet_height, darknet_width, video_height, video_width)),
    )
    for exec_unit in exec_units:
        exec_unit.start()
    for exec_unit in exec_units:
        exec_unit.join()

    print("\nDone.")

