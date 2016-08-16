//
//  klt_gpu.h
//  ITF_Inegrated
//
//  Created by Chenyang Xia on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef TRACKER_H
#define TRACKER_H
#include "itf/trackers/buffgpu.h"
#include "itf/trackers/buffers.h"
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "SortTracking.hpp"
//#include "testHungarian.hpp"
#include "itf/trackers/utils.h"

#include <stdint.h>
#include <inttypes.h>
#include <ctime>
#include <fstream>
#include <algorithm>
//#define DEBUG

enum TrackerStatus {FINE=0,TRACKINGERROR};


/*Visualize the tracking result*/
//cv::Mat vis_tracker(cv::Mat_<double> colours, cv::Mat& img, Sort& trackers);
void GenerateROI(std::vector<cv::Point>& contour, cv::Mat& roi);


enum{ frameNumber=0, boxLeft, boxTop, boxRight, boxBottom};

class CrowdTracker
{
public:
    CrowdTracker();

    ~CrowdTracker();
    /***** CPU *****/
    int init(int w,int h,unsigned char* framedata,int nPoints);
    void initDet(std::string fname);
    int updateAframe(unsigned char* framedata,int fidx);
    void getDetection(cv::Mat& frame);
    TrackerStatus curStatus;
    int frame_width,frame_height,frameidx,frameSize,frameSizeRGB;
    int tailidx, buffLen, mididx, preidx, nextidx;
    bool persDone;

    Sort* mot_tracker;
    std::vector<float> inFeatures;
    std::vector<data> centers;
    std::vector< vector<float> > feat;
    cv::Mat_<double> colours;
    int step_occlude = 0;
    int steps=5;//2053 261 5;
    int idx=0;
    int ii=1;
    unsigned long frameNum = 1;
    std::vector< cv::Mat > frame_to_frame;
};

#endif
