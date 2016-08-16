#include "itf/trackers/SortTracking.hpp"
#include <cmath>
#define AVOID
//#define debug
//#define ID_Features
#define HSV_Features
//#define patch_match

std::size_t Track::NextTrackID=0;

/*track constructor,use initial value*/
Track::Track(data pt, float dt, float Accel_noise_mag,bool reID, int ID)
{
	if(!reID)
	{
		track_id=NextTrackID;
		NextTrackID++;
	}
	if(reID)
	{
		track_id=ID;
	}
	// every track have its own Kalman filter
	KF = new KalmanBoxTracker(pt,dt,Accel_noise_mag);

	prediction=pt; //the box final location
	prediction.predict=false;
	skipped_frames=0;
}
Track::~Track()
{
	delete KF;
}


Sort::Sort(float _dt, float _Accel_noise_mag,int _maximum_allowed_skipped_frames,double _max_time,int _max_trace_length,unsigned int w,unsigned int h)
{
	dt=_dt;
	Accel_noise_mag=_Accel_noise_mag;
	maximum_allowed_skipped_frames=_maximum_allowed_skipped_frames;
	max_time=_max_time;
	max_trace_length=_max_trace_length;
        imageW = w;
        imageH = h;
}

void Sort::Update(vector<data>& detections,cv::Mat frame,unsigned long frameNumber) //,cv::Mat frame)
{
	
	if(tracks.size()==0)
	{
		//  no tracks ,then create it
		for(int i=0;i<detections.size();i++)
		{
			bool reid=false;
			int id=-1;
			Track* tr=new Track(detections[i],dt,Accel_noise_mag,reid,id);
			tracks.push_back(tr);
		}	
	}


	unsigned int N=tracks.size();		 
	unsigned int M=detections.size();
	//
	//cv::Mat boxframe(frame.clone());
#ifdef debug
	{
		std::vector<int> assignments;
		std::vector< vector<double> > Costs(4,vector<double>(4));
		Costs[0][0]=0.1;
		Costs[0][1]=10000;
		Costs[0][2]=10000;
		Costs[0][3]=10000;
		Costs[1][0]=10000;
		Costs[1][1]=0.2;
		Costs[1][2]=10000;
		Costs[1][3]=10000;
		Costs[2][0]=10000;
		Costs[2][1]=10000;
		Costs[2][2]=0.4;
		Costs[2][3]=10000;
		Costs[3][0]=10000;
		Costs[3][1]=10000;
		Costs[3][2]=10000;
		Costs[3][3]=10000;
		HungarianAlign HA;
		HA.Solve(Costs,assignments,HungarianAlign::without_forbidden_assignments);
		int j;
		j=1;
	}
#endif
	


	std::vector< vector<double> > Cost(N,vector<double>(M));
	std::vector< vector<double> > IOU_matrix(N,vector<double>(M));
	std::vector< vector<double> > ID_matrix(N,vector<double>(M));
	std::vector< vector<double> > Hist_matrix(N,vector<double>(M));
	std::vector< vector<double> > match_matrix(N,vector<double>(M));
	std::vector<int> assignment;

	for(int i=0;i<tracks.size();i++)
	{
		for (int j = 0; j < detections.size(); j++)
		{

			std::vector<int> bb_test;
			std::vector<int> bb_gt;
			std::vector<int> box_img;
			std::vector<int> box_tmp;

			//computer IOU
			bb_gt.push_back(detections[j].bbox[0]); //x1
			bb_gt.push_back(detections[j].bbox[1]); //y1
			bb_gt.push_back(detections[j].bbox[2]); //x2
			bb_gt.push_back(detections[j].bbox[3]); //y2


			bb_test.push_back(tracks[i]->prediction.bbox[0]); //x1
			bb_test.push_back(tracks[i]->prediction.bbox[1]); //y1
			bb_test.push_back(tracks[i]->prediction.bbox[2]); //x2
			bb_test.push_back(tracks[i]->prediction.bbox[3]); //y2

			double xx1 = std::max(bb_test[0], bb_gt[0]);
			double yy1 = std::max(bb_test[1], bb_gt[1]);
			double xx2 = std::min(bb_test[2], bb_gt[2]);
			double yy2 = std::min(bb_test[3], bb_gt[3]);

			double iou_w = std::max(0.0, xx2 - xx1 + 1);
			double iou_h = std::max(0.0, yy2 - yy1 + 1);

			double region_area = iou_w * iou_h;


			double IOU = 1 - (region_area / ((bb_test[2] - bb_test[0] + 1) * (bb_test[3] - bb_test[1] + 1) +
											 (bb_gt[2] - bb_gt[0] + 1) * (bb_gt[3] - bb_gt[1] + 1) - region_area));
			IOU_matrix[i][j]=1-IOU;
			if (IOU < 0.7)
			{

#ifdef ID_Features

				std::vector<float> dfeatures;
				std::vector<float> tfeatures;
				dfeatures = detections[j].Feature;
				tfeatures = tracks[i]->prediction.Feature;

				//cv::normalize(dfeatures,dfeatures,0,1,cv::NORM_MINMAX,-1,cv::Mat());
				//cv::normalize(tfeatures,tfeatures,0,1,cv::NORM_MINMAX,-1,cv::Mat());

				// cos value
				float sum = 0;
				float sum1 = 0;
				float sum2 = 0;
				for (int z = 0; z < dfeatures.size(); z++)
				{

					float value = dfeatures.at(z) * tfeatures.at(z);
					sum += value;
					float value1 = dfeatures.at(z) * dfeatures.at(z);
					float value2 = tfeatures.at(z) * tfeatures.at(z);
					sum1 += value1;
					sum2 += value2;

				}

				float ID_similarity = sum / (std::sqrt(sum1) * std::sqrt(sum2));
				ID_matrix[i][j]=ID_similarity;
#endif
#ifdef patch_match

				// computer appearance similarity  (patch match)

				//enlarge search area (detection area) by 25%
				int m=0.25*(bb_gt[2]-bb_gt[0]);
				box_img.push_back(std::max(bb_gt[0]-m,1));
				box_img.push_back(std::max(bb_gt[1]-m,1));
				box_img.push_back(std::min(bb_gt[2]+m,frame_to_frame[1].cols));
				box_img.push_back(std::min(bb_gt[3]+m,frame_to_frame[1].rows));

				//clip template
				box_tmp.push_back(std::max(bb_test[0],1));
				box_tmp.push_back(std::max(bb_test[1],1));
				box_tmp.push_back(std::min(bb_test[2],frame_to_frame[0].cols));
				box_tmp.push_back(std::min(bb_test[3],frame_to_frame[0].rows));

				//extract template and search area
				cv::Rect T;
				T.x=std::min(box_tmp[0],frame_to_frame[0].cols);
				T.y=std::min(box_tmp[1],frame_to_frame[0].rows);
				T.width=std::min(box_tmp[2]-box_tmp[0],(frame_to_frame[0].cols-T.x));
				T.height=std::min(box_tmp[3]-box_tmp[1],(frame_to_frame[0].rows-T.y));
				cv::Mat Temp=cv::Mat(frame_to_frame[0],T);
				cv::cvtColor(Temp,Temp,cv::COLOR_BGR2GRAY);

				cv::Rect S;
				S.x=std::min(box_img[0],frame_to_frame[1].cols);
				S.y=std::min(box_img[1],frame_to_frame[1].rows);
				S.width=std::min(box_img[2]-box_img[0],(frame_to_frame[1].cols-S.x));
				S.height=std::min(box_img[3]-box_img[1],(frame_to_frame[1].rows-S.y));
				cv::Mat S_area=cv::Mat(frame_to_frame[1],S);
				cv::cvtColor(S_area,S_area,cv::COLOR_BGR2GRAY);

				//crop template upper and lower, left and right part by 15%
				int c1=0.15*Temp.cols;
				int r=0.15*Temp.rows;
				Temp=Temp.rowRange(r+1,Temp.rows-r);
				Temp=Temp.colRange(c1+1,Temp.cols-c1);

				//compute correlation score
				double c;
				double d;
				if(Temp.cols>S_area.cols|| Temp.rows>S_area.rows)
				{
					 Cost[i][j]=10000;

				}
				else
				{
					cv::Mat result,D;
				//creat result matrix
					int result_cols=S_area.cols-Temp.cols+1;
					int result_rows=S_area.rows-Temp.rows+1;
					result.create(result_cols,result_rows,CV_32FC1);
				// match and normalize
					cv::matchTemplate(S_area,Temp,result,CV_TM_CCOEFF_NORMED);
					cv::normalize(result,result,0,1,cv::NORM_MINMAX,-1,cv::Mat());
					double maxVal,minVal;
					cv::Point minLoc,maxLoc,matchLoc;
					cv::minMaxLoc(result,&minVal,&maxVal,&minLoc,&maxLoc,cv::Mat());
					matchLoc=maxLoc;
				//extract match box from S_area
					cv::Rect R;
					R.x=matchLoc.x;
					R.y=matchLoc.y;
					R.width=Temp.cols;
					R.height=Temp.rows;
					cv::Mat match=cv::Mat(S_area,R);
				// computer SAD
					cv::absdiff(Temp,match,D);
					cv::Mat mean,stddev;
					cv::meanStdDev(D,mean,stddev);
				//normalize
					mean=mean/256;
					d=mean.at<uchar>(0,0);
					c=maxVal;
					match_matrix[i][j]=c;
				if(d<0.2 && c>0.6 )
				{

						Cost[i][j]=IOU*(1-hist_similarity);//*(1-c);//+(1-hist_similarity);  //IOU*(1-c)

				}

				else

				{
					Cost[i][j]=10000;
				}

				}
#endif

#ifdef HSV_Features

				//computer HSV similarity
				cv::Mat dhists;
				cv::Mat thists;
				dhists=detections[j].hist_feature;
				thists=tracks[i]->prediction.hist_feature;
				double hist_similarity=1-cv::compareHist(dhists,thists,CV_COMP_BHATTACHARYYA);
				Hist_matrix[i][j]=hist_similarity;

                if(hist_similarity<0.4)
				{
					Cost[i][j]=10000;
				}
				else
				{
					Cost[i][j]=IOU*(1-hist_similarity);
				}

			}

			else
			{
				Cost[i][j] = 10000; //std::numeric_limits<float>::infinity();
			}
#endif

		}
	}


	// ---------------------------------------------------
	// Solving assignment problem using Hungarian algorithm
	// ---------------------------------------------------
	
	HungarianAlign HA;
	if(N!=0 && M!=0)
	HA.Solve(Cost,assignment,HungarianAlign::optimal);

   if(M==0)
   {
    for(int i=0;i<N;i++)
	{
	 assignment.push_back(-1);
	}
   }

	// not assigned tracks
	std::vector<int> not_assigned_tracks;

	for(int i=0;i<assignment.size();i++)
	{
		if(assignment[i]!=-1)
		{
			if(Cost[i][assignment[i]]>1000)  //remove
			{ 
			    // mark unassigned tracks
				assignment[i]=-1;

				not_assigned_tracks.push_back(i);
			}
		}
		if(assignment[i]==-1)  //else
		{			
			// if track have no assigned detect, then increment skipped frames counter
			tracks[i]->skipped_frames++;
		}

	}


	// if track didnot get detects long time, remove it
	for(int i=0;i<tracks.size();i++)
	{
		if(tracks[i]->skipped_frames>maximum_allowed_skipped_frames)
		{
			data delTrack;
			// only save those occlude trackers
			if(tracks[i]->prediction.bbox[0]<20 || tracks[i]->prediction.bbox[2]>frame.cols-20 || tracks[i]->prediction.bbox[1]<20 || tracks[i]->prediction.bbox[4]>frame.rows-20)
			{
				delete tracks[i];
				tracks.erase(tracks.begin()+i);
				assignment.erase(assignment.begin()+i);
			}
			else
			{
				delTrack=tracks[i]->prediction;
				delTrack.score=1;
				delTrack.time_frame=1;
				delTrack.index=tracks[i]->track_id;
				delTracks.push_back(delTrack);
				delete tracks[i];
				tracks.erase(tracks.begin()+i);
				assignment.erase(assignment.begin()+i);
			}

			i--;
		}
	}
	// find unassigned detects
	std::vector<int> not_assigned_detections;
	std::vector<int>::iterator it;
	for(int i=0;i<detections.size();i++)
	{
		it=std::find(assignment.begin(), assignment.end(), i);
		if(it==assignment.end())
		{
		   not_assigned_detections.push_back(i);
		}
	}

	
	
	//  start new tracks for unmacthced detection
	// remove the long tracks
	if(delTracks.size()!=0)
	{
		for(int k=0;k<delTracks.size();k++)
		{
			if(delTracks[k].time_frame>max_time*25*60)
			{
				delTracks.erase(delTracks.begin()+k);
				k--;
			}
		}
	}

	if(not_assigned_detections.size()!=0)
	{
		bool reID = false;
		int Id = -1;
		unsigned int K=delTracks.size();

		if (K!=0)
		{
			for (int i = 0; i < not_assigned_detections.size(); i++)
			{
				std::vector<double> ID;
				std::vector<double> Dist;
				std::vector<double>Hist;
				if(delTracks.size()!=0)
				{
					for (int j = 0; j < delTracks.size(); j++)
					{

#ifdef ID_Features
						//computer dis_ID
						std::vector<float> dfeatures;
						std::vector<float> tfeatures;
						dfeatures = detections[not_assigned_detections[i]].Feature;
						tfeatures = delTracks[j].Feature;

						// computer its cos
						float sum=0;
						float sum1=0;
						float sum2=0;
						for(int z=0;z<256;z++)
						{
							float value=dfeatures.at(z)*tfeatures.at(z);
							sum+=value;
							float value1=dfeatures.at(z)*dfeatures.at(z);
							float value2=tfeatures.at(z)*tfeatures.at(z);
							sum1+=value1;
							sum2+=value2;

						}
						// cos value
						float cos_value=sum/(std::sqrt(sum1)*std::sqrt(sum2));
						ID.push_back(cos_value);
#endif

#ifdef HSV_Features

						//computer the HSV histgram features
						cv::Mat dhists;
						cv::Mat thists;
						dhists=detections[not_assigned_detections[i]].hist_feature;
						thists=delTracks[j].hist_feature;
						double hist_similarity=1-cv::compareHist(dhists,thists,CV_COMP_BHATTACHARYYA);
						Hist.push_back(hist_similarity);
#endif

						//computer dist
						double diff_x,diff_y;
						diff_x = 0.5*(detections[not_assigned_detections[i]].bbox[0] + detections[not_assigned_detections[i]].bbox[2]) - 0.5*(delTracks[j].bbox[0] + delTracks[j].bbox[2]);
						diff_y = 0.5*(detections[not_assigned_detections[i]].bbox[1] + detections[not_assigned_detections[i]].bbox[3]) - 0.5*(delTracks[j].bbox[1] + delTracks[j].bbox[3]);
						double dist=sqrtf(diff_x*diff_x+diff_y*diff_y);
						if (std::isnan(dist) || std::isinf(dist))
							dist =imageW;
						Dist.push_back(dist);

					}

					double maxid=0;
					maxid=*std::max_element(Hist.begin(),Hist.end());
					vector<double>::iterator idb=Hist.begin();
					vector<double>::iterator ide=std::find(Hist.begin(),Hist.end(),maxid);
					int id_idx=ide-idb;
					if (maxid>0.6 && Dist.at(id_idx)<imageH/3)   //0.6
					{
						reID = true;
						Id = delTracks[id_idx].index;
						delTracks.erase(delTracks.begin() + id_idx);
						Track *tr = new Track(detections[not_assigned_detections[i]], dt, Accel_noise_mag, reID, Id);
						tracks.push_back(tr);
						Hist.clear();
						Dist.clear();

					}
					else
					{
						reID=false;
						Id=-1;
						Track *tr = new Track(detections[not_assigned_detections[i]], dt, Accel_noise_mag, reID, Id);
						tracks.push_back(tr);
						Hist.clear();
						Dist.clear();
					}
				}
				else
				{
					reID=false;
					Id=-1;
					Track *tr = new Track(detections[not_assigned_detections[i]], dt, Accel_noise_mag, reID, Id);
					tracks.push_back(tr);
					Hist.clear();
					Dist.clear();
				}


			}
		}
		else
		{
			for (int i = 0; i < not_assigned_detections.size(); i++)
			{
				Track *tr = new Track(detections[not_assigned_detections[i]], dt, Accel_noise_mag, reID, Id);
				tracks.push_back(tr);
			}
		}

	}

	// Update Kalman Filters state
	
	for(int i=0;i<assignment.size();i++)
	{
		tracks[i]->KF->GetPrediction();

		if(assignment[i]!=-1) // if have assigned detect, then update using detection
		{
			tracks[i]->skipped_frames=0;
			tracks[i]->prediction=tracks[i]->KF->Update(detections[assignment[i]],1);
		}
                else				  // else using predictions
		{
		  data tmp;
	          tmp.bbox.push_back(0.0);
		  tmp.bbox.push_back(0.0);
                  tmp.bbox.push_back(0.0);
		  tmp.bbox.push_back(0.0);
	          tmp.score = 0;
	          tmp.index = 0;
			tmp.Feature=tracks[i]->prediction.Feature;
			tmp.hist_feature=tracks[i]->prediction.hist_feature;
			tracks[i]->prediction=tracks[i]->KF->Update(tmp,0);
		}

		
		if(tracks[i]->trace.size()>max_trace_length)
		{
			tracks[i]->trace.erase(tracks[i]->trace.begin(),tracks[i]->trace.end()-max_trace_length);
		}

		tracks[i]->trace.push_back(tracks[i]->prediction);
		tracks[i]->KF->LastResult=tracks[i]->prediction; // not necessery
	}
	// record delet trackers stay frames
	if(delTracks.size()!=0)
	{
		for(int i=0;i<delTracks.size();i++)
		{
			delTracks[i].time_frame++;
		}

	}

}

Sort::~Sort(void)
{
	for(int i=0;i<tracks.size();i++)
	{
	delete tracks[i];
	}
	tracks.clear();
}
