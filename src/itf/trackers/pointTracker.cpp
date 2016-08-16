#include "itf/trackers/trackers.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>

#include <cuda_runtime.h>
#include "itf/trackers/gpucommon.hpp"
#include "itf/trackers/utils.h"
#include "opencv2/gpu/device/common.hpp"
using namespace cv;
using namespace cv::gpu;
#define ZERO 1e-9

#define HSV_features

CrowdTracker::CrowdTracker()
{
    frame_width=0, frame_height=0;
    frameidx=0;
    /**cuda **/
    persDone=false;
}
CrowdTracker::~CrowdTracker()
{
}

int CrowdTracker::init(int w, int h,unsigned char* framedata,int nPoints)
{
    /** Checking Device Properties **/
    int nDevices;
    int maxthread=0;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
        std::cout << "maxgridDim" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
		std::cout << "maxThreadsPerBlock:" << prop.maxThreadsPerBlock << std::endl;
        

        //cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,MyKernel, 0, arrayCount);
        if(maxthread==0)maxthread=prop.maxThreadsPerBlock;
        //debuggingFile << prop.major << "," << prop.minor << std::endl;
    }
    //cudaSetDevice(1);
    //std::cout <<"device Status:"<< cudaSetDevice(1) << std::endl;
    /** Basic **/
    frame_width = w,frame_height = h;
	frameSize = frame_width*frame_height;
	frameSizeRGB = frame_width*frame_height*3;
    tailidx = 0, buffLen = 10, mididx = 0, preidx = 0, nextidx = 0;
    frameidx=0;
    mot_tracker = new Sort(0.1,0.5,8,0.2,20,frame_width,frame_height);
    centers = std::vector<data>();
    feat = std::vector< vector<float> >();
    step_occlude = 0;
    steps=5;//2053 261 5;
    idx=0;
    ii=1;
    frameNum = 1;
    frame_to_frame = std::vector<cv::Mat>(2);
    Mat frame(frame_height,frame_width,CV_8UC3,framedata);
    frame_to_frame[0]=frame;
    frame_to_frame[1]=frame;

    colours =cv::Mat_<double>(64,3); //random for the color,mark for tracking
    for(int row = 0; row < colours.rows; row++)
    {
      for(int col = 0; col < colours.cols; col++)
      {
        colours(row,col) = std::rand()%255;
      }
    }
    return 1;
}

void CrowdTracker::initDet(std::string fname)
{
    std::ifstream infile(fname.c_str(),std::ios_base::in);
    if(!infile.is_open())
    {
        std::cout<<"fail to open features file"<<std::endl;
        return;
    }

    while(infile)
    {
        float features;
        infile >> features;
        inFeatures.push_back(features);
    }
    std::cout<<"all features size: "<<inFeatures.size()<<std::endl;
}
int CrowdTracker::updateAframe(unsigned char* framedata, int fidx)
{
    std::clock_t start=std::clock();
    curStatus=FINE;
    frameidx=fidx;
    Mat frame(frame_height,frame_width,CV_8UC3,framedata);
   frameNum=fidx;
   frame_to_frame[0]=frame_to_frame[1];
   frame_to_frame[1]=frame;

// Detection Feature Extraction
    centers.clear();
    feat.clear();
    getDetection(frame);

// Tracking
   mot_tracker->Update(centers,frame,frameNum);

    float duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
    return 1;
}
void CrowdTracker::getDetection(cv::Mat& frame)
{
    int id_obj = 0;
    cv::Point2f pt1;
    cv::Point2f pt2;

    while(inFeatures.at(idx)==frameNum)
    {
        pt1.x = (int)inFeatures.at(idx+boxLeft);
        pt1.y = (int)inFeatures.at(idx+boxTop);
        pt2.x = (int)inFeatures.at(idx+boxRight);
        pt2.y = (int)inFeatures.at(idx+boxBottom);
        UperLowerBound(pt1.x,0,frame_width);
        UperLowerBound(pt1.y,0,frame_height);
        UperLowerBound(pt2.x,0,frame_width);
        UperLowerBound(pt2.y,0,frame_height);

        //Point2d center;
        data center;
        center.bbox.push_back(pt1.x);
        center.bbox.push_back(pt1.y);
        center.bbox.push_back(pt2.x);
        center.bbox.push_back(pt2.y);
        center.score = 1;
        center.index = id_obj;

//#ifdef HSV_features
        cv::Rect Drect;
        Drect.x=min(center.bbox[0],float(frame_width));
        Drect.y=min(center.bbox[1],float(frame_height));
        if (Drect.x<0) Drect.x=0;
        if (Drect.y<0) Drect.y=0;
        Drect.width=min(float(frame_width-Drect.x),center.bbox[2]-center.bbox[0]);
        Drect.height=min(float(frame_height-Drect.y),center.bbox[3]-center.bbox[1]);
        cv::Mat Droi=cv::Mat(frame,Drect).clone();
        // RGB to HSV
        cv::cvtColor(Droi,Droi,cv::COLOR_BGR2HSV);
        // hue 26bin saturation 18bin
        int h_bins=26; int s_bins=18;
        int histSize[]={h_bins, s_bins};

        //hue 0_256 saturation 0_180
        float h_ranges[]={0,256};
        float s_ranges[]={0,180};

        const float* ranges[]={h_ranges, s_ranges};

        //use 0 and 1 channels
        int channels[]={0,1};
        cv::Mat hist1;
        cv::calcHist(&Droi,1,channels,cv::Mat(),hist1,2,histSize,ranges,true,false);
        cv::normalize(hist1,hist1,0,1,cv::NORM_MINMAX,-1,cv::Mat());
        center.hist_feature=hist1;
//#endif

       if(step_occlude == 0)
           centers.push_back(center);
       else if(frameNum % step_occlude == 0)
           centers.push_back(center);
       else
           centers.clear();
       id_obj++;
       idx += steps;

   }
}
void GenerateROI(std::vector<cv::Point>& contour, cv::Mat& roi)
{
    roi = cv::Scalar(0);
    cv::Rect boundbox = cv::boundingRect(contour);
    int row_start = std::max(0,boundbox.y);
    int row_end = std::min(roi.rows,boundbox.y+boundbox.height);
    int col_start = std::max(0,boundbox.x);
    int col_end = std::min(roi.cols, boundbox.x+boundbox.width);
    for(int i= row_start;i<row_end;i++)
       for(int j=col_start;j<col_end;j++)
        {
           if(cv::pointPolygonTest(contour,cv::Point(j,i),false)==1)
              roi.at<uchar>(i,j) = 255;
        }

}

/**
cv::Mat vis_tracker(cv::Mat_<double> colours, cv::Mat& img, Sort& mot_tracker)
{
    for(int i=0;i<mot_tracker.tracks.size();i++)
    {
        if(mot_tracker.tracks[i]->trace.size()>1)
        {

            //display the trace
            for(int j=0;j<mot_tracker.tracks[i]->trace.size()-1;j++)
            {
                cv::line(img,mot_tracker.tracks[i]->trace[j].point,mot_tracker.tracks[i]->trace[j+1].point,colours,2,CV_AA);
            }
            //// only dispaly last rectangle,
            cv::Point2d pt1(mot_tracker.tracks[i]->trace[j].point.x-mot_tracker.tracks[i]->trace[j].w*0.5,mot_tracker.tracks[i]->trace[j].point.y-mot_tracker.tracks[i]->trace[j].h*0.5);
            cv::Point2d pt2(mot_tracker.tracks[i]->trace[j].point.x+mot_tracker.tracks[i]->trace[j].w*0.5,mot_tracker.tracks[i]->trace[j].point.y+mot_tracker.tracks[i]->trace[j].h*0.5);
            cv::rectangle(img, pt1, pt2, colours, 2);
        }
    }

    cv::Mat image = img.clone();
    return image;

#if 0
      for (unsigned int i = 0; i < trackers.size(); i++)
    {
        // std::cout<<"The trackers' index: "<<trackers[i].index<<std::endl;

        int selRows = trackers[i].index % colours.rows;
        cv::Scalar color = cv::Scalar(colours(selRows, 0), colours(selRows, 1), colours(selRows, 2));

        std::vector<float> bbox = trackers[i].bbox;
        cv::rectangle(img, cv::Point(bbox[0], bbox[1]), cv::Point(bbox[2], bbox[3]), color, 2); //draw the bounding box
        // cv::rectangle(img, cv::Rect(bbox[0], bbox[1] - 15, 80,15), cv::Scalar(255,255,255), -1);

        //ADD

        for(unsigned int j=0;j<trackers[i].trace.size();j++)
        {
          cv::circle(img,trackers[i].trace[j],3,color,-1,8);
        }

        float per_score = trackers[i].score;
        char str_score[15];
        std::sprintf(str_score, "%f", per_score);
        cv::putText(img, str_score, cv::Point(bbox[0], bbox[1] - 5), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, 5);
    }
    cv::Mat image = img.clone();
    return image;
#endif
}
**/

/*
cv::Point2d ConvertData2Point(data& D)
{
    cv::Point2d tmp;
    tmp.x =
    return bbox;
}
*/
