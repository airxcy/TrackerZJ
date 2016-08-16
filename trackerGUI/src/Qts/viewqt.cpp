#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "Qts/streamthread.h"

#include <iostream>
#include <stdio.h>

#include <QPainter>
#include <QBrush>
#include <QPixmap>
#include <cmath>
#include <QGraphicsSceneEvent>
#include <QMimeData>
#include <QByteArray>
#include <QFont>
char viewstrbuff[200];
QPointF points[100];

void DefaultScene::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    emit clicked(event);
}
void DefaultScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    QPen pen;
    QFont txtfont("Roman",40);
    txtfont.setBold(true);
    pen.setColor(QColor(255,255,255));
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(10);
    painter->setPen(QColor(243,134,48,150));
    painter->setFont(txtfont);
    painter->drawText(rect, Qt::AlignCenter,"打开文件\nOpen File");
}
TrkScene::TrkScene(const QRectF & sceneRect, QObject * parent):QGraphicsScene(sceneRect, parent)
{
    streamThd=NULL;
}
TrkScene::TrkScene(qreal x, qreal y, qreal width, qreal height, QObject * parent):QGraphicsScene( x, y, width, height, parent)
{
    streamThd=NULL;
}
void TrkScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    //debuggingFile<<streamThd->inited<<std::endl;
     if(streamThd!=NULL&&streamThd->inited)
     {
         updateFptr(streamThd->frameptr, streamThd->frameidx);
     }
     painter->setBrush(bgBrush);
     painter->drawRect(rect);
 //    painter->setBrush(QColor(0,0,0,100));
 //    painter->drawRect(rect);
     painter->setBrush(Qt::NoBrush);
     if(streamThd!=NULL&&streamThd->inited)
     {
         linepen.setColor(QColor(255,200,200));
         linepen.setWidth(3);
         painter->setPen(linepen);
         painter->setFont(QFont("System",20,2));
         QString infoString="fps:"+QString::number(streamThd->fps)+"\n";
         painter->drawText(rect, Qt::AlignLeft|Qt::AlignTop,infoString);
         painter->setFont(QFont("System",20,2));
         Sort* tracker = streamThd->tracker->mot_tracker;
         vector<Track*>& tracks=tracker->tracks;
         cv::Mat_<double>& colours=streamThd->tracker->colours;
         for(int i=0;i<tracks.size();i++)
         {
             Track* trk = tracks[i];
             int selRows = trk->track_id % colours.rows;
             cv::Scalar color = cv::Scalar(colours(selRows, 0), colours(selRows, 1), colours(selRows, 2));
             linepen.setWidth(2);
             linepen.setColor(QColor(colours(selRows, 0), colours(selRows, 1), colours(selRows, 2)));
             painter->setPen(linepen);
             if(tracks[i]->trace.size()>1)
             {
                                 //start point
                cv::Point2d startPt1(trk->trace[0].bbox[0],trk->trace[0].bbox[1]);
                cv::Point2d startPt2(trk->trace[0].bbox[2],trk->trace[0].bbox[3]);
                cv::Point startPtCen(0.5*(startPt1.x+startPt2.x),0.5*(startPt1.y+startPt2.y));
                 for(int j=0;j<trk->trace.size()-1;j++)
                 {
                        cv::Point2d pt1(0.5*(trk->trace[j].bbox[0]+trk->trace[j].bbox[2]),0.5*(trk->trace[j].bbox[1]+trk->trace[j].bbox[3]));
                        cv::Point2d pt2(0.5*(trk->trace[j+1].bbox[0]+trk->trace[j+1].bbox[2]),0.5*(trk->trace[j+1].bbox[1]+trk->trace[j+1].bbox[3]));
                        painter->drawLine(pt1.x,pt1.y,pt2.x,pt2.y);
                 }
                    int lastidix = trk->trace.size()-1;
                     int bbx=trk->trace[lastidix].bbox[0];
                     int bby=trk->trace[lastidix].bbox[1];
                     int bbw=trk->trace[lastidix].bbox[2]-trk->trace[lastidix].bbox[0];
                     int bbh=trk->trace[lastidix].bbox[3]-trk->trace[lastidix].bbox[1];
                     painter->drawRect( bbx, bby,bbw,bbh);
                     painter->drawText(bbx,bby,QString::number(trk->track_id));
            }
         }
     }
}
void TrkScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	/*
    if(event->button()==Qt::RightButton)
    {
        int x = event->scenePos().x(),y=event->scenePos().y();
        DragBBox* newbb = new DragBBox(x-10,y-10,x+10,y+10);
        int pid = dragbbvec.size();
        newbb->bbid=pid;
        newbb->setClr(255,255,255);
        sprintf(newbb->txt,"%c\0",pid+'A');
        dragbbvec.push_back(newbb);
        addItem(newbb);
    }
    QGraphicsScene::mousePressEvent(event);
	*/
}
void TrkScene::updateFptr(unsigned char * fptr,int fidx)
{
    bgBrush.setTextureImage(QImage(fptr,streamThd->framewidth,streamThd->frameheight,QImage::Format_RGB888));
    frameidx=fidx;
    //debuggingFile<<frameidx<<std::endl;
}
void TrkScene::clear()
{
    bgBrush.setStyle(Qt::NoBrush);
}
