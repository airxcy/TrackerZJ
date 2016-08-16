#include "Qts/streamthread.h"

#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "itf/trackers/utils.h"
#include <iostream>
#include <fstream>
#include <ostream>
//#include <stdlib.h>

//#include <direct.h>
#include "Qts/mainwindow.h"
#include "opencv2/opencv.hpp""
#include <ctime>

#include <QMessageBox>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QStringList>

using namespace cv;

VideoCapture cap;
Mat frame;
float fps=0;
char strbuff[100];
QDir qdirtmp;
Mat gray,segMat;

VideoWriter vwriter;
QImage rgbimg,img;
QPainter* painter;
Mat renderframe;

StreamThread::StreamThread(QObject *parent) : QThread(parent)
{
    restart = false;
    abort = false;
    pause = false;
    paused=false;
    bufflen=0;
    trkscene=NULL;
    framebuff=NULL;
    inited=false;
	firsttime = true;
    persDone=false;
    outVid =false;
    tracker = new CrowdTracker();
}
StreamThread::~StreamThread()
{
    abort = true;
	vwriter.release();
    cv0.wakeAll();
    wait(3000);
    terminate();
}

bool StreamThread::init()
{
    restart=false,pause=false;
    bufflen=0;

    if(!cap.isOpened())
    {
        cap.open(vidfname);
        std::cout<<"reopened"<<std::endl;
    }
    cap.set(CV_CAP_PROP_POS_FRAMES,0);
    frameidx=0;
    cap>>frame;
    fps=0;
    delay=25;
    bufflen=delay+10;
    cvtColor(frame,frame,CV_BGR2RGB);
    framewidth=frame.size[1],frameheight=frame.size[0];
    //cvtColor(frame,gray,CV_BGR2GRAY);
    if(framebuff==NULL)
    {
        framebuff = new FrameBuff();
        framebuff->init(frame.elemSize(),framewidth,frameheight,bufflen);
    }
    else
        framebuff->clear();
	
    frameByteSize=frame.size[0]*frame.size[1]*frame.elemSize();
    framebuff->updateAFrame(frame.data);
    frameptr=framebuff->cur_frame_ptr;
	parsefname();



	std::cout << "persDone:" << persDone << std::endl;

	if (firsttime){
        tracker->init( framewidth, frameheight,frame.data,1024);
        tracker->initDet(detPath.toStdString());
    }
    if(!persDone)
    {
        pause=true;
    }
    if(outVid&&!vwriter.isOpened())
    {
        vwriter.open(vidid.toStdString() + "out.avi", CV_FOURCC('D','I','V','X'), 25, Size(framewidth, frameheight));
        rgbimg = QImage(framewidth, frameheight, QImage::Format_RGB888);
        img = QImage(framewidth, frameheight, QImage::Format_RGB888);
        renderframe = Mat(frameheight, framewidth,CV_8UC3);
        painter = new QPainter(&img);
    }
    inited=true;
	firsttime = false;
    return cap.isOpened();
}
void StreamThread::writeVid()
{
    painter->begin(&img);
    trkscene->render(painter);
    painter->end();
    rgbimg = img.convertToFormat(QImage::Format_RGB888);
    //debuggingFile<<rgbimg.byteCount()<<std::endl;
    memcpy(renderframe.data, rgbimg.bits(), rgbimg.byteCount());
    cvtColor(renderframe, renderframe, CV_BGR2RGB);
    vwriter << renderframe;
}
void StreamThread::parsefname()
{
	QFileInfo qvidfileinfo(vidfname.data());
	baseDirname = qvidfileinfo.path();
	vidid = qvidfileinfo.baseName();
    detPath = baseDirname + "/" + vidid + ".txt";
	vidid = vidid + "_" + qvidfileinfo.completeSuffix();

	gtdir = baseDirname + "/" + vidid + "/";
	qdirstr = baseDirname + "/" + vidid + "/";
}
void StreamThread::setUpPersMap()
{
    if(!persDone)
    {
        QString savefname = baseDirname +"/"+vidid +"_persMap.csv";
    }
}
void StreamThread::setUpROI()
{
    QString savefname = baseDirname +"/"+vidid +"_ROI.csv";
	std::string binraryName = savefname.left(savefname.size() - 4).toStdString() + ".dat";
}

void StreamThread::streaming()
{
    forever
    {
        if (abort)
            break;
        if(init())
        {
            emit initSig();
            frameidx=0;
            int fcounter=0;
            std::clock_t start = std::clock();
            double duration;
            int bbfcounter=0;
            while(!frame.empty())
            {
                if (abort)
                    break;
                if (pause)
                {
                    mutex.lock();
                    paused=true;
                    cv0.wait(&mutex);
                    paused=false;
                    mutex.unlock();
                }
                cap >> frame;

                if(frame.empty())
                    break;
                //cvtColor(frame,gray,CV_BGR2GRAY);
                cvtColor(frame,frame,CV_BGR2RGB);
                tracker->updateAframe(frame.data, frameidx);
                //tracker->Render(frame.data);
                framebuff->updateAFrame(frame.data);

                frameptr=framebuff->cur_frame_ptr;
                frameidx++;
                //video write
                trkscene->update();
                if(outVid)emit aframedone();

                fcounter++;
                duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
                if(duration>=1)
                {
                    fps=fcounter/duration;
                    start=std::clock() ;
                    fcounter=0;
                }
               //pause=true;
            }
            vwriter.release();
            break;
        }
        else
        {
            //emit debug( "init Failed");
        }
        trkscene->clear();
        inited=false;
    }
}
void StreamThread::run()
{
    streaming();
}

void StreamThread::streamStart(std::string & filename)
{
    QMutexLocker locker(&mutex);
    //QMessageBox::question(NULL, "Test", "msg",QMessageBox::Ok);
    if (!isRunning()) {
        vidfname=filename;
        start(InheritPriority);
    }
    else
    {
        restart=true;
    }
}
