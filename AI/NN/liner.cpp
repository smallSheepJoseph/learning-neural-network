#include<bits/stdc++.h>
using namespace std;
using dl = double;
double random(){
    return (double)rand() / 32767;
}

const dl training_rate = 0.05;

int main(){
    vector<pair<dl,dl>> data;
    for(int i = 0;i<1000;i++){
        dl x = random();
        dl y = x*1.5+2+0.2*random()-0.1;
        data.push_back({x,y});
    }
    dl ta = 0,tb = 0;
    for(int i = 0;i<10;i++){
        for(auto [x,y] : data){
            dl error = y - (ta*x+tb);
            tb += training_rate*error;
            ta += training_rate*x*error;
        }
        printf("y = %lfx + %lf\n",ta,tb);
    }
}