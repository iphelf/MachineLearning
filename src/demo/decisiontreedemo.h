#ifndef DECISIONTREEDEMO_H
#define DECISIONTREEDEMO_H

#include <iostream>
#include <fstream>
#include <vector>
#include "src/decisiontree/decisiontree.h"
using namespace std;

void runWithDecisionTree(string filename) {
    printf("runWithDecisionTree(%s)-------------------------------\n",
           filename.c_str());
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

    Dataset dataset(n);
    for(int i=0; i<n; i++) {
        dataset[i]=new Data{vector<string>(data[i].begin()+lb,
                                           data[i].begin()+rb+1),
                            *(data[i].begin()+rs)};
    }
    int d=rb-lb+1;
    vector<int> keys(d);
    for(int i=0; i<d; i++) keys[i]=i;
    vector<vector<string>> valueSpace(n,vector<string>(d));
    for(int i=lb; i<=rb; i++) {
        set<string> space;
        for(int j=0; j<n; j++) space.insert(data[j][i]);
        valueSpace[i-lb].assign(space.begin(),space.end());
    }
    DecisionNode *dtree=generateDicisionTree(dataset,keys,valueSpace);
    printDecisionTree(dtree,vector<string>(attribute.begin()+lb,
                                           attribute.begin()+rb+1),"");
}

#endif // DECISIONTREEDEMO_H
