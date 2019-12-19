#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "src/stdafx.h"
#include <vector>
using namespace Eigen;
using namespace std;

VectorXd linear(MatrixXd X,VectorXd Y) {
    //O(max(X.rows*X.rows*X.cols,X.cols*X.cols*X.cols))
    return (X.transpose()*X).inverse()*X.transpose()*Y;
}

struct Evaluation {
    int nT,nF,n; //number
    int _padding; //padding for byte align
    double pctT,pctF; //percent
    double rtT,rtF; //rate
    Evaluation() {
        reset();
    }
    Evaluation operator +(Evaluation &r) {
        Evaluation ret;
        ret.n=n+r.n;
        ret.nT=nT+r.nT;
        ret.nF=nF+r.nF;
        ret.update();
        return ret;
    }
    void reset() {
        memset(this,0,sizeof(Evaluation));
    }
    void update() {
        if(n==0) return;
        rtT=nT*1.0/n;
        rtF=nF*1.0/n;
        pctT=rtT*100;
        pctF=rtF*100;
    }
};

Evaluation crossValidate(vector<MatrixXd> X,int fold=4) {
    int nType=X.size();
    double avgAvgRateT=0;
    Evaluation avgAvgE;
    //Cross validation for each type
    for(int iType=0; iType<nType; iType++) {
        MatrixXd &Xi=X[iType];
        int n=Xi.cols();
        double avgRateT=0;
        Evaluation avgE;
        //Validation for each fold as targets
        for(int i=0; i<fold; i++) {
            int l=n/fold*i;
            int r=n/fold*(i+1);
            Evaluation e;
            e.n=n/fold;
            //For each target in fold
            for(int j=l; j<r; j++) {
                VectorXd Y=Xi.col(j);
                //With other folds of each type as training set
                //To select out the type that best fit
                int bestType=-1;
                double lowestSE=0;
                for(int jType=0; jType<nType; jType++) {
                    MatrixXd &Xj=X[jType];
                    MatrixXd tX(Xi.rows(),n-n/fold+1);
                    tX<<MatrixXd::Constant(Xj.rows(),1,1),
                    Xj.block(0,0,Xj.rows(),l-0),
                    Xj.block(0,r,Xj.rows(),n-r);
                    //Linear Regression!
                    VectorXd tY=tX*linear(tX,Y);
                    double se=(tY-Y).norm();
                    if(bestType==-1||se<lowestSE) {
                        bestType=jType;
                        lowestSE=se;
                    }
                }
                //OK, the best type is on the dish
                if(bestType==iType) e.nT++;
                else e.nF++;
            }
            e.update();
            avgE=avgE+e;
        }
        avgAvgE=avgAvgE+avgE;
        printf("\tCross validation for type %d:\n",iType);
        printf("\t\tCorrect rate=%3.2f%%\n",avgE.pctT);
    }
    return avgAvgE;
}

#endif // LINEARREGRESSION_H
