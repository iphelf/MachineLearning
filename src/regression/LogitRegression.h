#ifndef LOGITREGRESSION_H
#define LOGITREGRESSION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include <chrono>
#include "src/stdafx.h"
using namespace std;
using namespace Eigen;

VectorXd logit(const vector<VectorXd> &X,
               const vector<int> &Y,
               int repeat=100) {
    int n=X.size();
    int k=X[0].rows();
    VectorXd w=VectorXd::Constant(k,0);
    VectorXd bestW=VectorXd::Constant(k,0);
    double L;
    double minL=-1;
    VectorXd Ld(k);
    MatrixXd Ldd(k,k);
    while(repeat--) {
        L=0;
        Ld=VectorXd::Zero(k);
        Ldd=MatrixXd::Zero(k,k);
        for(int i=0; i<n; i++) {
            double wx=w.transpose()*X[i];
            double p1=1/(1+exp(-wx));
            L+=(wx>100?
                -Y[i]*wx+wx:
                -Y[i]*wx+log(1+exp(wx))/log(exp(1)));
            Ld+=-X[i]*(double(Y[i])-p1);
            Ldd+=X[i]*X[i].transpose()*p1*(1-p1);
        }
        if(minL==-1||L<minL) {
            minL=L;
            bestW=w;
        }
        w-=Ldd.inverse()*Ld;
    }
    return bestW;
}

struct Evaluation {
    int nT,nF,n; //number
    int nTP,nTN,nFP,nFN; //true/false positive/negative
    int _padding; //padding for byte align
    double pctT,pctF; //percent
    double rtT,rtF; //rate
    double mae; //mean absolute error
    double mse; //mean squared error
    double pcs; //precision
    double rcl; //recall
    Evaluation() {
        reset();
    }
    Evaluation operator +(Evaluation &r) {
        Evaluation ret;
        ret.n=n+r.n;
        ret.nTP=nTP+r.nTP;
        ret.nTN=nTN+r.nTN;
        ret.nFP=nFP+r.nFP;
        ret.nFN=nFN+r.nFN;
        ret.update();
        if(ret.n==0) return ret;
        ret.mae=(mae*n+r.mae*r.n)/ret.n;
        ret.mse=(mse*n+r.mse*r.n)/ret.n;
        return ret;
    }
    void reset() {
        memset(this,0,sizeof(Evaluation));
    }
    void update() {
        nT=nTP+nTN;
        nF=nFP+nFN;
        if(n==0) return;
        rtT=nT*1.0/n;
        rtF=nF*1.0/n;
        pctT=rtT*100;
        pctF=rtF*100;
        if(nTP+nFP==0) return;
        pcs=nTP*1.0/(nTP+nFP);
        if(nTP+nFN==0) return;
        rcl=nTP*1.0/(nTP+nFN);
    }
};

Evaluation evaluate(const vector<VectorXd> &X,
                    const vector<int> &Y,
                    const VectorXd &w) {
    Evaluation e;
    e.n=X.size();
    int y;
    double ty;
    double se=0;
    double ae=0;
    for(int i=0; i<int(X.size()); i++) {
        const VectorXd &x=X[i];
        y=Y[i];
        ty=1/(1+exp(-w.transpose()*x));
        se+=(y-ty)*(y-ty);
        ae+=(y-ty<0?ty-y:y-ty);
        if(y==1 && ty>0.5) 		e.nTP++, e.nT++;
        else if(y==1 && ty<0.5) e.nFN++, e.nF++;
        else if(y==0 && ty<0.5) e.nTN++, e.nT++;
        else if(y==0 && ty>0.5) e.nFP++, e.nF++;
        else;
    }
    e.mse=se/e.n;
    e.mae=ae/e.n;
    e.update();
    return e;
}

Evaluation crossValidate(vector<VectorXd> X,vector<int> Y,int fold=4) {
    int n=X.size();
    Evaluation e;
    vector<Evaluation> ve(fold,Evaluation());
    for(int i=0; i<fold; i++) {
        int l=n/fold*i;
        int r=n/fold*(i+1);
        vector<VectorXd> sX(X.begin()+r,X.end());
        sX.insert(sX.begin(),X.begin(),X.begin()+l);
        vector<int> sY(Y.begin()+r,Y.end());
        sY.insert(sY.begin(),Y.begin(),Y.begin()+l);
        vector<VectorXd> tX(X.begin()+l,X.begin()+r);
        vector<int> tY(Y.begin()+l,Y.begin()+r);
        VectorXd w=logit(sX,sY);
        ve[i]=evaluate(tX,tY,w);
        e=e+ve[i];
    }
    return e;
}

#endif // LOGITREGRESSION_H
