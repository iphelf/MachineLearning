#ifndef ORL_H
#define ORL_H
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "src/stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/eigen.hpp>
#include "src/regression/LinearRegression.h"
using namespace cv;
using namespace Eigen;
using namespace std;

void generateImages(bool useLarge=true) {
    int n=400;
    int w,h;
    w=h=(useLarge?64:32);
    Mat m(h,w,CV_8U,Scalar(0));
    char filename[20];
    sprintf(filename,"fea%d.txt",w);
    freopen(filename,"r",stdin);
    int t;
    for(int k=0; k<n; k++) {
        for(int j=0; j<w; j++) for(int i=0; i<h; i++) {
                scanf("%d",&t);
                m.at<uchar>(i,j)=uchar(t);
            }
        sprintf(filename,"ORL%d/%03d.jpg",w,k);
        imwrite(filename,m);
    }
}

void runOnORL(bool useLarge=true) {
    int n=400;
    int sz=(useLarge?64:32);
    char filename[20];
    vector<MatrixXd> X(40,MatrixXd(sz*sz,10));
    VectorXd x(sz*sz);
    Mat m;
    for(int i=0; i<n; i++) {
        sprintf(filename,"ORL%d/%03d.jpg",sz,i);
        m=imread(filename,0);
        m=m.reshape(0,sz*sz);
        cv2eigen(m,x);
        X[i/10].col(i%10)=x/255;
    }
    Evaluation e=crossValidate(X,2);
    printf("Average correct rate=%3.2f%%\n",e.pctT);
}

void runOnNumbers() {
    MatrixXd X(2,3);
    X<<1,1,1,1,3,5;
    VectorXd Y(3);
    Y<<4.8,11.3,17.2;
    VectorXd w=linear(X,Y);
    cout<<"w="<<endl<<w<<endl;
}

#endif // ORL_H
