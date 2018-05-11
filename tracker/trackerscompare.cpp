
#include "kcf/kcftracker.hpp"
//#include "goturn/network/regressor.h"
//#include "goturn/tracker/tracker.h"

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

#ifdef MAESTRO
#include "maestro.h"
static int ud = 6000;
static int lr = 6000;
#endif

using namespace cv;
using namespace std;

// Convert to string
#define SSTR(x) static_cast<std::ostringstream &>(           \
                    (std::ostringstream() << std::dec << x)) \
                    .str()

void write_txt(int detect_flag, int lost_flag, int imagewriting_flag)
{
    FILE *f = fopen("tracker.txt", "w");
    if (f == NULL)
    {
        printf("ERROR opening tracker.txt to save the box that needed to follow.\n");
    }
    fprintf(f, "%d,%d,%d\n", detect_flag, lost_flag, imagewriting_flag);
    fclose(f);
}

extern "C" bool trackerscompare();
//int main(int argc, char **argv)
bool trackerscompare()
{
    int detect_flag = 0;
    int lost_flag = 0;
    int lost_count = 0;
    int imagewriting_flag = 0;

#ifdef DEBUG
    string path = "/media/elab/sdd/mycodes/darknet"; //"/home/nvidia/amy/Sam";//
#else
    string path = "/home/nvidia/amy/darknet";
#endif
    // Read the groundtruth bbox
    ifstream groundtruth(path + "/yolo.txt");
    std::string f;
    int x, y, w, h;
    std::string s;
    // Read the image
    cv::Mat frame;
    ostringstream osfile;
    osfile << path << "/yolo.png";
    // opencv bbox
    Rect2d opencvbbox(x, y, w, h);

#ifdef MAESTRO
    int fd = maestroIni();
#endif

    // Open camera
    VideoCapture cap(0); // open the default camera
    //cap.set(CV_CAP_PROP_BUFFERSIZE, 0);
    if (!cap.isOpened()) // check if we succeeded
    {
        cout << "Could not open camera \n"
             << std::endl;
        return -1;
    }

    // Display frame.
    cvNamedWindow("Tracking", CV_WINDOW_NORMAL);
    //cvResizeWindow("Tracking", 1352, 1013);
#ifndef DEBUG
    cvSetWindowProperty("Tracking", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
#endif

    /*
// Create KCFTracker: 
    bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool LAB = false; //LAB color space features
	KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
*/
    // Create Opencv tracker:
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN"};
    string trackerType = trackerTypes[2];
    Ptr<cv::Tracker> opencvtracker;

    if (trackerType == "BOOSTING")
        opencvtracker = cv::TrackerBoosting::create();
    if (trackerType == "MIL")
        opencvtracker = cv::TrackerMIL::create();
    if (trackerType == "KCF")
        opencvtracker = cv::TrackerKCF::create();
    if (trackerType == "TLD")
        opencvtracker = cv::TrackerTLD::create();
    if (trackerType == "MEDIANFLOW")
        opencvtracker = cv::TrackerMedianFlow::create();
    if (trackerType == "GOTURN")
        opencvtracker = cv::TrackerGOTURN::create();
    /*
// Create GOTURN tracker:
    const string model_file = "goturn/nets/deploy.prototxt";
    const string pretrain_file = "goturn/nets/goturun_tracker.caffemodel";
    int gpu_id = 0;

    Regressor regressor(model_file,pretrain_file,gpu_id, false);
    goturn::Tracker goturntracker(false);
*/
    /*
// Read from the images ====================================================
    string path = "/media/elab/sdd/data/TLP/Sam";
	// Read the groundtruth bbox
	ifstream groundtruth(path + "/groundtruth_rect.txt");
	int f,x,y,w,h,isLost;
	std::string s;
	getline(groundtruth, s, ',');	
	f = atoi(s.c_str());
	getline(groundtruth, s, ',');
	x = atoi(s.c_str());
	getline(groundtruth, s, ',');	
	y = atoi(s.c_str());
	getline(groundtruth, s, ',');
	w = atoi(s.c_str());
	getline(groundtruth, s, ',');	
	h = atoi(s.c_str());
	getline(groundtruth, s);
	isLost = atoi(s.c_str());
	cout << f <<" " << x <<" " << y <<" " << w <<" " << h <<" " << isLost << endl;
    Rect2d bboxGroundtruth(x,y,w,h);
	
	// Read images in a folder
	ostringstream osfile;
	osfile << path << "/img/" << setw(5) << setfill('0') << f <<".jpg";
	cout << osfile.str() << endl;
    cv::Mat frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if(! frame.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
*/

    // Read the groundtruth bbox
    getline(groundtruth, s, ',');
    f = s;
    getline(groundtruth, s, ',');
    x = atoi(s.c_str());
    getline(groundtruth, s, ',');
    y = atoi(s.c_str());
    getline(groundtruth, s, ',');
    w = atoi(s.c_str());
    getline(groundtruth, s, ',');
    h = atoi(s.c_str());
    getline(groundtruth, s);
    //detect_flag = atoi(s.c_str());
    cout << "Target to track:" << f << " " << x << " " << y << " "
         << w << " " << h << " " << detect_flag << endl;

    // Read the image
    frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if (!frame.data)
    {
        cout << "Could not capture the image \n"
             << std::endl;
        return -1;
    }
    // Init the trackers==================================================
    /*
    Rect2d kcfbbox(x,y,w,h);
    kcftracker.init(frame, kcfbbox);
*/
    opencvbbox.height = h;
    opencvbbox.width = w;
    opencvbbox.x = x;
    opencvbbox.y = y;
    //opencvtracker->init(frame, opencvbbox);
    if (!opencvtracker->init(frame, opencvbbox))
    {
        cout << "Could not init the tracker \n"
             << std::endl;
        return -1;
    }
    /*
    cv::Rect goturnbbox(x,y,w,h);
    BoundingBox bbox_gt;
    BoundingBox bbox_estimate_uncentered;
    bbox_gt.getRect(goturnbbox);
    goturntracker.Init(frame,bbox_gt,&regressor);
*/
    detect_flag = 0;
    lost_flag = 0;
    imagewriting_flag = 0;
    write_txt(detect_flag, lost_flag, imagewriting_flag);

    while (frame.data)
    {
        // Read the groundtruth bbox
        groundtruth.clear();
        groundtruth.seekg(0, std::ios::beg);
        getline(groundtruth, s, ',');
        f = s;
        getline(groundtruth, s, ',');
        x = atoi(s.c_str());
        getline(groundtruth, s, ',');
        y = atoi(s.c_str());
        getline(groundtruth, s, ',');
        w = atoi(s.c_str());
        getline(groundtruth, s, ',');
        h = atoi(s.c_str());
        getline(groundtruth, s);
        detect_flag = atoi(s.c_str());
        //cout << "Target to track:" << f << " " << x << " " << y << " "
        //     << w << " " << h << " " << detect_flag << endl;
        if (lost_flag == 1 && detect_flag == 1)
        {
            // Read the image
            frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
            if (!frame.data)
            {
                cout << "Could not capture the image \n"
                     << std::endl;
                return -1;
            }
            // Init the trackers==================================================
            /*
    Rect2d kcfbbox(x,y,w,h);
    kcftracker.init(frame, kcfbbox);
*/
            opencvbbox.height = h;
            opencvbbox.width = w;
            opencvbbox.x = x;
            opencvbbox.y = y;
            opencvtracker = cv::TrackerKCF::create();
            if (!opencvtracker->init(frame, opencvbbox))
            {
                cout << "Could not re-init the tracker \n"
                     << std::endl;
                return -1;
            }
            /*
    cv::Rect goturnbbox(x,y,w,h);
    BoundingBox bbox_gt;
    BoundingBox bbox_estimate_uncentered;
    bbox_gt.getRect(goturnbbox);
    goturntracker.Init(frame,bbox_gt,&regressor);
*/
            detect_flag = 0;
            lost_flag = 0;
            imagewriting_flag = 0;
            write_txt(detect_flag, lost_flag, imagewriting_flag);
        }
        // Draw ground truth box
        //rectangle(frame, bboxGroundtruth, Scalar( 0, 0, 0 ), 2, 1 );

        // Start timer
        double timer = (double)getTickCount();
        /*         
        // Update the KCF tracking result-----------------------------
        bool okkcf = kcftracker.update(frame, kcfbbox);
        // draw kcf bbox
        if (okkcf) {
            rectangle(frame, kcfbbox, Scalar( 225, 0, 0 ), 2, 1); //blue
        } else {
            putText(frame, "Kcf tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255,0,0),2);
        }
*/
        // Update the Opencv tracking result----------------------------
        bool okopencv = opencvtracker->update(frame, opencvbbox);
        if (!okopencv)
        {
            lost_count++;
        }
        if (lost_count >= 50)
        {
            lost_count = 0;
            detect_flag = 0;
            lost_flag = 1;
            imagewriting_flag = 1;
            write_txt(detect_flag, lost_flag, imagewriting_flag);
            imwrite("tracker.png", frame);
            imagewriting_flag = 0;
            write_txt(detect_flag, lost_flag, imagewriting_flag);
        }

        // draw opencv bbox
        if (okopencv)
        {
            rectangle(frame, opencvbbox, Scalar(0, 225, 0), 2, 1); //green
        }
        else
        {
            putText(frame, "Opencv tracking failure detected", Point(100, 110), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 225, 0), 2);
        }
        /*
        // Update the GOTURN tracking result--------------------------
        goturntracker.Track(frame, &regressor, &bbox_estimate_uncentered);
        bbox_estimate_uncentered.putRect(goturnbbox);
        // draw goturn bbox
        rectangle(frame, goturnbbox, Scalar(0, 0, 255), 2, 1); //red
*/
        // Calculate Frames per second (FPS)-------------------------------
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
        // Display FPS on frame
        putText(frame, "FPS in total: " + SSTR(long(fps)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 2);
        // Display tracker type on frame
        putText(frame, "Blue is KCF; Red is GOTURN; Green is opencv " + trackerType + ";", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 2);

        // Display frame.
        //cvNamedWindow("Tracking", CV_WINDOW_NORMAL);
        imshow("Tracking", frame);

        int c = cvWaitKey(1);
        if (c != -1)
            c = c % 256;
        if (c == 27)
        {
            cvDestroyWindow("Tracking");
            return 0;
        }
        waitKey(1);

        cap.grab();
        cap.retrieve(frame);
/*        
		// Read next image
		f++;
		osfile.str("");
		osfile << path << "/img/" << setw(5) << setfill('0') << f <<".jpg";
		cout << osfile.str() << endl;
    	frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
		// Read next bbox
		getline(groundtruth, s, ',');	
		f = atoi(s.c_str());
		getline(groundtruth, s, ',');
		x = atoi(s.c_str());
		getline(groundtruth, s, ',');	
		y = atoi(s.c_str());
		getline(groundtruth, s, ',');
		w = atoi(s.c_str());
		getline(groundtruth, s, ',');	
		h = atoi(s.c_str());
		getline(groundtruth, s);
		isLost = atoi(s.c_str());
		cout << f <<" " << x <<" " << y <<" " << w <<" " << h <<" " << isLost << endl;
		bboxGroundtruth.x = x;
		bboxGroundtruth.y = y;
		bboxGroundtruth.width = w;
		bboxGroundtruth.height = h;
*/
#ifdef MAESTRO
        //test_maestro();
        //cout << frame.rows << ", " << frame.cols << ",x- "
        //<< bbox.x << ",w- " << bbox.width << ",y- " << bbox.y
        //<< ",h- " << bbox.height << endl;
        //        cout << frame.cols/2 - (opencvbbox.x + opencvbbox.width/2) << ", "
        //        << frame.rows/2 - (opencvbbox.y + opencvbbox.height/2) << endl;

        int templr = (frame.cols / 2 - (opencvbbox.x + opencvbbox.width / 2));
        int tempud = (frame.rows / 2 - (opencvbbox.y + opencvbbox.height / 2));

        lr += templr;
        ud += tempud;
        //        cout << lr << ", " << ud << endl;

        maestroSetTarget(fd, 1, lr); //control left right
        maestroSetTarget(fd, 0, ud); //control up down
        lr = 6000;
        ud = 6000;
#endif
    }
    cvDestroyWindow("Tracking");
    cap.release();
    return 0;
}
