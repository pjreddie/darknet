
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
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

extern "C" int trackerscompare();
//int main(int argc, char **argv)
int trackerscompare()
{
#ifdef MAESTRO
  int fd = maestroIni();
#endif

// Create KCFTracker: 
    bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool LAB = false; //LAB color space features
	KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

// Create Opencv tracker:
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
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
#ifdef DEBUG
    string path = "/media/elab/sdd/mycodes/darknet";//"/home/nvidia/amy/Sam";//
#else
    string path = "/home/nvidia/amy/darknet";
#endif
    //cout << path << endl;
	// Read the groundtruth bbox
	ifstream groundtruth(path + "/tracking.txt");
    std::string f;
	int x,y,w,h;
	std::string s;
	getline(groundtruth, s, ',');	
	f = s;
	getline(groundtruth, s, ',');
	x = atoi(s.c_str());
	getline(groundtruth, s, ',');	
	y = atoi(s.c_str());
	getline(groundtruth, s, ',');
	w = atoi(s.c_str());
	getline(groundtruth, s);	
	h = atoi(s.c_str());
	cout << "Target to track:" << f <<" " << x <<" " << y <<" " << w <<" " << h <<" " << endl;
	Rect2d bbox(x,y,w,h);

	// Open camera
    VideoCapture cap(0); // open the default camera
    //cap.set(CV_CAP_PROP_BUFFERSIZE, 0);
   if(!cap.isOpened())// check if we succeeded
    {
        cout <<  "Could not open camera \n" << std::endl ;
        return -1;        
    }
    cv::Mat frame;
	ostringstream osfile;
	osfile << path << "/tracking.png";
	//cout << osfile.str() << endl;
    frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);

    if(! frame.data )
    {
        cout <<  "Could not capture the image \n" << std::endl ;
        return -1;
    }
    // Display frame.
    cvNamedWindow("Tracking", CV_WINDOW_NORMAL); 
    //cvResizeWindow("Tracking", 1352, 1013);
#ifndef DEBUG
    cvSetWindowProperty("Tracking", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
#endif

	// Init the trackers==================================================
    Rect2d kcfbbox(x,y,w,h);
    kcftracker.init(frame, kcfbbox);

    Rect2d opencvbbox(x,y,w,h);
    opencvtracker->init(frame, opencvbbox);
/*
    cv::Rect goturnbbox(x,y,w,h);
    BoundingBox bbox_gt;
    BoundingBox bbox_estimate_uncentered;
    bbox_gt.getRect(goturnbbox);
    goturntracker.Init(frame,bbox_gt,&regressor);
*/

    while(frame.data) {
        // Draw ground truth box
		//rectangle(frame, bboxGroundtruth, Scalar( 0, 0, 0 ), 2, 1 );

        // Start timer
        double timer = (double)getTickCount();
         
        // Update the KCF tracking result-----------------------------
        bool okkcf = kcftracker.update(frame, kcfbbox);
        // draw kcf bbox
        if (okkcf) {
            rectangle(frame, kcfbbox, Scalar( 225, 0, 0 ), 2, 1); //blue
        } else {
            putText(frame, "Kcf tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255,0,0),2);
        }

        // Update the Opencv tracking result----------------------------
        bool okopencv = opencvtracker->update(frame, opencvbbox);
        // draw opencv bbox
        if (okopencv) {
            rectangle(frame, opencvbbox, Scalar( 0, 225, 0 ), 2, 1); //green
        } else {
            putText(frame, "Opencv tracking failure detected", Point(100,110), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,225,0),2);
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
        putText(frame, "FPS in total: " + SSTR(long(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0), 2);
        // Display tracker type on frame
        putText(frame, "Blue is KCF; Red is GOTURN; Green is opencv " + trackerType + ";", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),2);

        // Display frame.        
        //cvNamedWindow("Tracking", CV_WINDOW_NORMAL); 
        imshow("Tracking", frame);

        int c = cvWaitKey(1);
        if (c != -1) c = c%256;
        if (c == 27) {
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
        cout << frame.cols/2 - (opencvbbox.x + opencvbbox.width/2) << ", "
        << frame.rows/2 - (opencvbbox.y + opencvbbox.height/2) << endl;

        int templr = (frame.cols/2 - (opencvbbox.x + opencvbbox.width/2));
        int tempud = (frame.rows/2 - (opencvbbox.y + opencvbbox.height/2));

        lr += templr;
        ud += tempud;
        cout << lr << ", " << ud << endl;

        maestroSetTarget(fd, 1, lr); //control left right
        maestroSetTarget(fd, 0, ud); //control up down
        lr = 6000;
        ud = 6000;
#endif

    }
    cvDestroyWindow("Tracking");

    cap.release();
    return 0;

/*
// Read from the video ====================================================
    // Read video
    VideoCapture video("videos/chaplin.mp4");
     
    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;      
    }
     
    // Read first frame
    Mat frame;
    bool ok = video.read(frame);
     
    // Define initial boundibg box
    Rect2d bbox(287, 23, 86, 320);
     
    // Uncomment the line below to select a different bounding box
    bbox = selectROI(frame, false);
 
    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Tracking", frame);
   
    while(video.read(frame))
    {     
        // Start timer
        double timer = (double)getTickCount();
         
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
         
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
         
        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
 
        // Display frame.
        imshow("Tracking", frame);
         
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
 
    }

*/
}
