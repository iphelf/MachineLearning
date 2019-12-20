#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <iostream>
#include <map>
#include <vector>
#include <set>
using namespace std;

typedef pair<vector<string>,string> Data;
typedef vector<Data*> Dataset;

class DecisionNode {
  public:
    DecisionNode() {
        mKey=-1;
    }
    bool isLeaf() {
        return mKey<0;
    }
    string label() {
        return mLabel;
    }
    int key() {
        return mKey;
    }
    DecisionNode *decide(string v) {
        map<string,DecisionNode*>::iterator ret=child.find(v);
        if(ret==child.end()) return nullptr;
        else return ret->second;
    }
    friend DecisionNode *generateDicisionTree(
        const Dataset &data,
        const vector<int> &keys,
        const vector<vector<string>> &keyVSpace);
    friend void printDecisionTree(DecisionNode *rt,
                                  const vector<string> &keyName,
                                  const string &prefix);
  private:
    map<string,DecisionNode*> child; //[key=value]->node
    int mKey;
    string mLabel;
};

double infent(const vector<string> &v) {
    //Information Entropy
    map<string,int> cnt;
    for(const string &s:v) cnt[s]++;
    double ent=0;
    int n=v.size();
    for(auto it=cnt.begin(); it!=cnt.end(); it++) {
        double p=1.0*it->second/n;
        ent-=p*log2(p);
    }
    return ent;
}

int getDivider(const Dataset &data,const vector<int> &keys,
               const vector<vector<string>> valueSpace) {
    int n=data.size();
    int bestKey=-1;
    double bestEnt=-1; //the smaller, the better
    for(int key:keys) {
        vector<vector<string>> div(valueSpace[key].size());
        map<string,int> rank;
        for(int i=0; i<valueSpace[key].size(); i++)
            rank[valueSpace[key][i]]=i;
        for(int i=0; i<n; i++)
            div[rank[data[i]->first[key]]].push_back(data[i]->second);
        double ent=0;
        for(int i=0; i<div.size(); i++)
            ent+=1.0*div[i].size()/n*infent(div[i]);
        if(bestKey==-1||ent<bestEnt) {
            bestEnt=ent;
            bestKey=key;
        }
    }
    return bestKey;
}

vector<Dataset*> divide(const Dataset &data,int key,
                        const vector<string> valueSpace) {
    vector<Dataset*> *div=new vector<Dataset*>(valueSpace.size(),nullptr);
    for(int i=0; i<div->size(); i++) div->at(i)=new Dataset;
    map<string,int> rank;
    for(int i=0; i<valueSpace.size(); i++) rank[valueSpace[i]]=i;
    for(int i=0; i<data.size(); i++)
        div->at(rank[data[i]->first[key]])->push_back(data[i]);
    return *div;
}

DecisionNode *generateDicisionTree(const Dataset &data,
                                   const vector<int> &keys,
                                   const vector<vector<string>> &valueSpace) {
    if(data.empty()) return nullptr;
    DecisionNode *root=new DecisionNode;

    vector<string> labels(data.size());
    bool equalValues=true;
    labels[0]=data[0]->second;
    for(int i=1; i<data.size(); i++) {
        labels[i]=data[i]->second;
        if(data[i]->second!=data[0]->second)
            equalValues=false;
    }
    if(equalValues) {
        root->mLabel=data[0]->second;
        return root;
    }

    sort(labels.begin(),labels.end());
    set<string> labelSpace(labels.begin(),labels.end());
    string mostLabel;
    int cntLabel=0;
    for(const string &label:labelSpace) {
        int cnt=upper_bound(labels.begin(),labels.end(),label)-
                lower_bound(labels.begin(),labels.end(),label);
        if(cnt>cntLabel) {
            mostLabel=label;
            cntLabel=cnt;
        }
    }

    if(keys.empty()||labelSpace.size()==1) {
        root->mLabel=mostLabel;
        return root;
    }

    int key=getDivider(data,keys,valueSpace);
    root->mKey=key;
    vector<Dataset*> div=divide(data,key,valueSpace[key]);

    vector<int> newKeys=keys;
    newKeys.erase(find(newKeys.begin(),newKeys.end(),key));

    for(int i=0; i<div.size(); i++) {
        if(div[i]->empty()) {
            DecisionNode *leaf=new DecisionNode;
            leaf->mLabel=mostLabel;
            root->child[valueSpace[key][i]]=leaf;
        } else {
            root->child[valueSpace[key][i]]=
                generateDicisionTree(*div[i],newKeys,valueSpace);
        }
    }

    return root;
}

void printDecisionTree(DecisionNode *rt,
                       const vector<string> &keyName,
                       const string &prefix) {
    if(rt->isLeaf()) {
        printf(">>%s\n",rt->mLabel.data());
        return;
    }
    int width=0;
    string key=keyName[rt->mKey];
    for(auto it=rt->child.begin(); it!=rt->child.end(); it++)
        width=max(width,int(it->first.size()));
    printf("<%s>-",key.data());
    if(rt->child.size()>1) printf("©Ð");
    else printf("©¤");
    string currPrefix=prefix+string(3+key.size(),' ');
    string newPrefix=currPrefix+"©¦"+string(4+width,' ');
    string s=rt->child.begin()->first.data();
    printf("-[%*s%s]-",
           width-(width-s.size()+1)/2,s.data(),
           string((width-s.size()+1)/2,' ').data());
    printDecisionTree(rt->child.begin()->second,keyName,newPrefix);
    auto it=rt->child.begin();
    it++;
    auto tail=rt->child.end();
    tail--;
    for(; it!=tail && it!=rt->child.end(); it++) {
        s=it->first;
        printf("%s©À-[%*s%s]-",
               currPrefix.data(),
               width-(width-s.size()+1)/2,s.data(),
               string((width-s.size()+1)/2,' ').data());
        printDecisionTree(it->second,keyName,newPrefix);
    }
    if(it!=rt->child.end() && it==tail) {
        s=it->first;
        printf("%s©¸-[%*s%s]-",
               currPrefix.data(),
               width-(width-s.size()+1)/2,s.data(),
               string((width-s.size()+1)/2,' ').data());
        printDecisionTree(it->second,keyName,currPrefix+" "+string(4+width,' '));
    }
}

#endif // DECISIONTREE_H
