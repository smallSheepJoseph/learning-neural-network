#include<bits/stdc++.h>
using namespace std;
using dl = double;
#define ff first
#define ss second
// double random(){
//     return (double)rand() / 32767 + (double)rand() / (32767*32767.0);
// }

#define SZ 20
#define DEPTH 5
#define sample_sz 100000
#define real_sample_sz (sample_sz + 1)

// double init_value(int size){
//     return (random() - 0.5) * 2.0 * sqrt(1.0/size);
// }

// double init_he(int n) {
//     static std::default_random_engine generator;
//     std::normal_distribution<double> distribution(0.0, sqrt(2.0/n));
//     return distribution(generator);
// }

#define adj(x) if(abs(x)>10.0){x*=0.99;}if(x>30.0){x=30.0;}if(x<-30.0){x=-30.0;}

dl training_rate = 0.001;

template<int InSize, int OutSize>
struct Layer {
    dl W[InSize][OutSize];
    dl B[OutSize];

    dl dlW[InSize][OutSize]={};
    dl dlB[OutSize]={};

    dl learned_data=0;

    Layer() {
        static std::mt19937 gen(1337);
        std::normal_distribution<dl> dis(0.0, std::sqrt(1.0 / InSize));
        
        for (int i = 0; i < InSize; ++i){
            for (int j = 0; j < OutSize; ++j){
                W[i][j] = dis(gen);
                dlW[i][j] = 0;
            }
        }
        for (int j = 0; j < OutSize; ++j){
            B[j] = dis(gen);
            dlB[j] = 0;
        }
            
    }

    static constexpr int param_count() {
        return InSize * OutSize + OutSize;
    }

    inline void run(dl* input,dl* output){
        for(int i = 0;i<OutSize;i++){
            output[i] = 0;
        }
        for(int i = 0;i<InSize;i++){
            for(int j = 0;j<OutSize;j++){
                output[j] += input[i]*W[i][j];
            }
        }
        for(int j = 0;j<OutSize;j++){
            output[j] += B[j];
        }
    }

    inline void rev(dl* input, dl* output) {
        for(int i = 0;i<InSize;i++){
            output[i] = 0;
        }
        for(int i = 0;i<InSize;i++){
            for(int j = 0;j<OutSize;j++){
                output[i] += input[j]*W[i][j];
            }
        }
    }

    inline void train(dl* y, dl* input,dl* output){
        rev(input,output);
        
        for(int i = 0;i<InSize;i++){
            for(int j = 0;j<OutSize;j++){
                dlW[i][j]+=y[i]*input[j];
            }
        }
        for(int j = 0;j<OutSize;j++){
            dlB[j]+=input[j];
        }
        learned_data++;
    }

    inline void act_train(){
        for(int i = 0;i<InSize;i++){
            for(int j = 0;j<OutSize;j++){
                W[i][j]-=dlW[i][j]*training_rate*(1.0/(learned_data+1));
                adj(W[i][j]);
                dlW[i][j] *= 0.95;
            }
        }
        for(int j = 0;j<OutSize;j++){
            B[j]-=dlB[j]*training_rate*(1.0/(learned_data+1));
            adj(B[j]);
            dlB[j] *= 0.95;
        }
        learned_data*=0.95;
    }


};

//activation function
template<int Size>
struct acf{

    inline dl fc(dl x){
        return tanh(x);
    }

    // inline dl dfc(dl x){
    //     return 1.0 - tanh(x)*tanh(x);
    // }

    inline dl dfc(dl x){
        return 1.0 - x*x;
    }

    inline void run(dl* input,dl* output){
        for(int i = 0;i<Size;i++){
            output[i] = fc(input[i]);
        }
    }

    inline void rev(dl* y, dl* input, dl* output) {
        for(int i = 0; i < Size; i++) {
            output[i] = input[i] * dfc(y[i]);
        }
    }

};


struct NN{
    Layer<1,SZ> input_layer;
    Layer<SZ,SZ> hidden_layer[DEPTH];
    Layer<SZ,2> output_layer;
    acf<SZ> active_fc;
    dl V[DEPTH+1][SZ];
    void run(dl* input,dl* output){
        input_layer.run(input,V[0]);
        active_fc.run(V[0],V[0]); //activation func can handle this

        for(int i = 0;i<DEPTH;i++){
            hidden_layer[i].run(V[i],V[i+1]);
            active_fc.run(V[i+1],V[i+1]); //activation func can handle this
        }
        output_layer.run(V[DEPTH],output);
    }
    void train(dl* input,dl* real_value){
        dl output[SZ];
        run(input,output);
        dl dl_output[2] = {2.0*(output[0]-real_value[0]),2.0*(output[1]-real_value[1])};
        // active_fc.rev(output,dl_output,dl_output);
        dl delta[SZ],delta2[SZ];
        output_layer.train(V[DEPTH],dl_output,delta);
        dl* now_delta = delta2,*last_delta = delta;
        for(int i = DEPTH-1;i>=0;i--){
            active_fc.rev(V[i+1],last_delta,last_delta);
            hidden_layer[i].train(V[i],last_delta,now_delta);
            swap(last_delta,now_delta);
        }
        active_fc.rev(V[0],last_delta,last_delta);
        input_layer.train(input,last_delta,now_delta);
    }
    void act_train(){
        input_layer.act_train();
        for(int i = 0;i<DEPTH;i++){
            hidden_layer[i].act_train();
        }
        output_layer.act_train();
    }
};

int main(){
    vector<pair<dl,pair<dl,dl>>> data;
    for(int i = 0;i<=sample_sz;i++){
        dl x = (dl)i/(dl)sample_sz;
        dl y1 = sin(x*2*3.14159265358979323846);
        dl y2 = sin(x*4*3.14159265358979323846);
        data.push_back({x,{y1,y2}});
    }
    std::random_device rd;
    std::mt19937 g(rd()); 
    shuffle(data.begin(), data.end(), g);

    NN nn;
    for(int i = 0;i<300;i++){
        training_rate = 0.3 / (1 + i * 0.01);
        int t = 0;
        for(auto dd : data){
            dl x = dd.ff,y[2] = {dd.ss.ff,dd.ss.ss};
            nn.train(&x,y);
            t++;
            if(t%100 == 0)nn.act_train();
        }
        nn.act_train();

        dl loss1 = 0,loss2 = 0;
        for(auto dd : data){
            dl x = dd.ff,y[2] = {dd.ss.ff,dd.ss.ss};
            dl val[2];
            nn.run(&x,val);
            loss1 += (val[0]-y[0])*(val[0]-y[0]);
            loss2 += (val[1]-y[1])*(val[1]-y[1]);
        }
        loss1 /= data.size();
        loss2 /= data.size();
        printf("st: %d,loss1 : %.8lf ,loss2 : %.8lf\n",i,loss1,loss2);
    }
    int t = 0;
    scanf("%d",&t);
    while(t != -1){
        dl x = t/360.0;
        dl prec[2];
        nn.run(&x,prec);
        printf("sin %d = %.8lf, prec = %.8lf\n",t,sin(t/180.0*3.14159265358979323846),prec[0]);
        printf("sin %d = %.8lf, prec = %.8lf\n",t*2,sin(t/90.0*3.14159265358979323846),prec[1]);
        scanf("%d",&t);
    }
}