//
//  utils.h
//  ITF_Inegrated
//
//  Created by ChenYang Xia on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef UTILS
#define UTILS

#include <algorithm>
#include <vector>
#include "itf/trackers/buffers.h"

#include <opencv2/core/core.hpp>

struct data
{
    std::vector<float> bbox;
    float score;
    int index;
    std::vector<float> Feature;
    cv::Mat hist_feature;
    long time_frame;
    bool predict;
};

int getLineIdx(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB);
int getLineProp(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB,double linedist);
void HSVtoRGB(unsigned char *r, unsigned char *g, unsigned char *b, float h, float s, float v );
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void convex_hull(std::vector<cvxPnt>& P, FeatBuff &H);



#endif // UTILS

