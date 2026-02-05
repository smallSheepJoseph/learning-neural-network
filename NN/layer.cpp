#include<bits/stdc++.h>
using namespace std;
using dl = double;
#define SZ 60
// double random(){
//     return (double)rand() / 32767 + (double)rand() / (32767*32767.0);
// }

double random() {
    static std::mt19937 gen(1337);
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}

double init_value(){
    return (random() - 0.5) * 2.0 * sqrt(1.0/SZ);
}



#define adj(x) if(x>100.0){x=100.0;}if(x<-100.0){x=-100.0;}

dl training_rate = 0.001;



struct small_NN{
    dl L1[SZ]={},L2[SZ][SZ]={},L3[SZ]={},C1[SZ]={},C2[SZ]={};dl C3 = 0;

    small_NN(){
        C3 = init_value();
        for(int i = 0;i<SZ;i++){
            L1[i] = init_value();
            L3[i] = init_value();
            C1[i] = init_value();
            C2[i] = init_value();
            for(int j = 0;j<SZ;j++){
                L2[i][j] = init_value();
            }
        }

    }

    inline dl fc(dl x){
        if(x>0){
            return x;
        }else{
            return x*0.05;
        }
    }

    inline dl dfc(dl x){
        if(x>0){
            return 1;
        }else{
            return 0.05;
        }
    }

    dl operator()(dl x){
        dl v[SZ]={},v2[SZ]={},v3=0;
        for(int i = 0;i<SZ;i++){
            v[i] = fc(x*L1[i] + C1[i]);
        }
        for(int i = 0;i<SZ;i++){
            for(int j = 0;j<SZ;j++){
                v2[i] += v[j]*L2[j][i];
            }
            v2[i] += C2[i];
            v2[i] = fc(v2[i]);
        }
        for(int i = 0;i<SZ;i++){
            v3 += v2[i]*L3[i];
        }
        v3 += C3;
        v3 = fc(v3);
        return v3;
    }

   

    void train(dl x,dl y){


        //cul
        dl v[SZ]={},v2[SZ]={},v3=0;
        for(int i = 0;i<SZ;i++){
            v[i] = x*L1[i] + C1[i];
        }
        for(int i = 0;i<SZ;i++){
            for(int j = 0;j<SZ;j++){
                v2[i] += fc(v[j])*L2[j][i];
            }
            v2[i] += C2[i];
        }
        for(int i = 0;i<SZ;i++){
            v3 += fc(v2[i])*L3[i];
        }
        v3 += C3;
        dl prec_value = fc(v3);

        //Gradient
        dl dy = prec_value - y;
        dl dL1[SZ]={},dL2[SZ][SZ]={},dL3[SZ]={},dC1[SZ]={},dC2[SZ]={};dl dC3 = 0;
        dC3 = dy*dfc(v3);
        for(int i = 0;i<SZ;i++){
            dC2[i] += dfc(v2[i]) * dC3 * L3[i];
            dL3[i] += dfc(v2[i]) * dC3;
        }
        for(int i = 0;i<SZ;i++){
            for(int j = 0;j<SZ;j++){
                dC1[i] += dfc(v[i])*dC2[j]*L2[i][j];
                dL2[i][j] += dfc(v[i])*dC2[j];
            }
        }
        for(int i = 0;i<SZ;i++){
            dL1[i] += dC1[i]*x;
        }

        //Descent
        C3 -= training_rate*dC3;
        adj(C3);
        for(int i = 0;i<SZ;i++){
            L1[i] -= training_rate*dL1[i];
            adj(L1[i]);
            L3[i] -= training_rate*dL3[i];
            adj(L3[i]);
            C1[i] -= training_rate*dC1[i];
            adj(C1[i]);
            C2[i] -= training_rate*dC2[i];
            adj(C2[i]);
            for(int j = 0;j<SZ;j++){
                L2[i][j] -= training_rate*dL2[i][j];
                adj(L2[i][j]);
            }
        }

    }
    void save_bin(string filename) {
        ofstream f(filename, ios::binary);
        f.write((char*)this, sizeof(small_NN)); 
        f.close();
        printf("Binary weights saved!\n");
    }

    void load_bin(string filename) {
        ifstream f(filename, ios::binary);
        if(f) {
            f.read((char*)this, sizeof(small_NN));
            f.close();
            printf("Model loaded successfully.\n");
        }
    }
};



int main(){
    vector<pair<dl,dl>> data;
    for(int i = 0;i<500000;i++){
        dl x = random();
        dl y = sin(x*2*3.14159265358979323846);
        data.push_back({x,y});
    }
    printf("training sin func\n");
    small_NN s = small_NN();
    s.load_bin("layer");
    for(int i = 0;i<50;i++){
        // if(i<=10){
        //     training_rate = 0.01;
        // }else if(i<=50){
        //     training_rate = 0.001;
        // }else if(i<=100){
        //     training_rate = 0.0001;
        // }else{
        //     training_rate = 0.00001;
        // }
        training_rate = 0.001 / (1 + i * 0.01);
        for(auto [x,y] : data){
            dl err = s(x) - y;
            s.train(x,y);
        }
        
        // if(i%10 != 0)continue;

        dl loss = 0;
        for(auto [x,y] : data){
            dl val = s(x);
            loss += (val-y)*(val-y);
        }
        loss /= data.size();
        printf("st: %d,loss : %lf\n",i,loss);
    }
    dl loss = 0;
    for(auto [x,y] : data){
        dl val = s(x);
        loss += (val-y)*(val-y);
    }
    s.save_bin("layer");
    loss /= data.size();
    printf("loss : %lf\n",loss);

    printf("sin 180 = %lf, prec = %lf\n",sin(1.0*3.14159265358979323846),s(0.5));
    printf("sin 30 = %lf, prec = %lf\n",sin(1.0/6.0*3.14159265358979323846),s(1.0/12.0));
    int t = 0;
    scanf("%d",&t);
    while(t != -1){
        printf("sin %d = %lf, prec = %lf\n",t,sin(t/180.0*3.14159265358979323846),s(t/360.0));
        scanf("%d",&t);
    }
}