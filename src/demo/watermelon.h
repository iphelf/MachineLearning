#ifndef WATERMELON_H
#define WATERMELON_H
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include "src/regression/logitregression.h"
using namespace std;

void generateWatermelonData() {
    freopen("watermelon-linear-perfect.csv","w",stdout);
    double w[3]= {-7,10,4};
    int n=1000;
    printf("4 %d 1 2 3\n",n);
    printf("西瓜数据集3.0perfect\n");
    printf("编号	密度	含糖率	好瓜\n");
    srand(0);
    int mod=1007;
    for(int i=0; i<n; i++) {
        double rho=(rand()%mod)*1.0/mod;
        double swt=(rand()%mod)*1.0/mod;
        printf("%d\t%.3f\t%.3f\t%s\n", i+1, rho, swt,
               (w[0]+w[1]*rho+w[2]*swt>0?"是":"否"));
    }
}

void runOnWatermelon(string filename,bool deNoise=false) {
    printf("runOnWatermelon(%s,%s)-------------------------------\n",
           filename.c_str(),deNoise?"true":"false");
    ifstream in;
    in.open(filename,ios_base::in);
    int k;
    int n; //|X|
    int lb,rb,rs;
    in>>k>>n>>lb>>rb>>rs;
    string title;
    in>>title;
    vector<string> attribute(k);
    for(int i=0; i<k; i++) in>>attribute[i];
    vector< vector<string> > data(n,vector<string>(k));
    for(int i=0; i<n; i++) for(int j=0; j<k; j++) in>>data[i][j];
    in.close();

    if(deNoise) {
        vector<int> toDelete;
        for(int i=0; i<n; i++) for(int j=i+1; j<n; j++) {
                vector<string> l=data[i],r=data[j];
                if(l[3]==r[3]) continue;
                if(l[1]<r[1]) swap(l,r);
                if(l[2]>r[2] && l[3]=="否" && r[3]=="是") {
                    cout<<"Noise pair: "<<l[0]<<" & "<<r[0]<<endl;
                    toDelete.push_back(i);
                    toDelete.push_back(j);
                }
            }
        sort(toDelete.begin(),toDelete.end());
        toDelete.erase(unique(toDelete.begin(),toDelete.end()),toDelete.end());
        for(int i=toDelete.size()-1; i>=0; i--) {
            data.erase(data.begin()+toDelete[i]);
            n--;
        }
        cout<<toDelete.size()<<" noise points were deleted!"<<endl<<endl;
    }

    int d=rb-lb+1+1; //|x|
    vector<VectorXd> X(n,VectorXd(d));
    vector<int> Y(n);
    for(int i=0; i<n; i++) {
        X[i](0)=1;
        for(int j=lb; j<=rb; j++) X[i](j-lb+1)=stod(data[i][j]);
        Y[i]=(data[i][rs]=="是"?1:0);
    }

    vector<int> permutation(n);
    for(int i=0; i<n; i++) permutation[i]=i;
    int nRound=10;
    vector<VectorXd> tX(n,VectorXd(d));
    vector<int> tY(n);
    Evaluation avgE;
    for(int I=0; I<nRound; I++) {
        shuffle(permutation.begin(),
                permutation.end(),
                default_random_engine(0));
        for(int i=0; i<n; i++) {
            tX[i]=X[permutation[i]];
            tY[i]=Y[permutation[i]];
        }
        Evaluation e=crossValidate(tX,tY,2); //*-fold cross validation
        avgE=avgE+e;
        printf("#%02d: Correct rate=%3.2f%%\n",I,e.pctT);
    }
    printf("Overall average: Cr=%3.2f%%\n"
           "                MSE=%3.2f\n"
           "\n",avgE.pctT,avgE.mse);

    printf("Logit regression over all data\n");
    VectorXd w=logit(X,Y);
    cout<<"w="<<endl<<w<<endl;
    printf("Evaluation over all data\n");
    avgE=evaluate(X,Y,w);
    printf("Overall average: Cr=%3.2f%%\n"
           "                MSE=%3.2f\n"
           "\n",avgE.pctT,avgE.mse);
}

#endif // WATERMELON_H
