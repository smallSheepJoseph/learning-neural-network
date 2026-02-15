#include<bits/stdc++.h>
using namespace std;
using dl = double;
using ll = long long;
#define ff first
#define ss second
// double random(){
//     return (double)rand() / 32767 + (double)rand() / (32767*32767.0);
// }
double random_z() {
    static std::mt19937 gen(1337);
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}
#define SZ 40
#define DEPTH 6
#define sample_sz 100000
#define real_sample_sz (sample_sz + 1)

dl fixed_sin(dl x){
    return sin(x*2*3.14159265358979323846);
}

#define adj(x) if(abs(x)>1e6){x*=0.99;}if(x>1e8){x=1e8;}if(x<-1e8){x=-1e8;}

dl training_rate = 0.0003;

template<int InSize, int OutSize>
struct Layer {
    dl W[InSize][OutSize];
    dl B[OutSize];

    dl dlW[InSize][OutSize]={};
    dl dlB[OutSize]={};

    dl mB[OutSize]={};
    dl mW[InSize][OutSize]={};
    dl vB[OutSize]={};
    dl vW[InSize][OutSize]={};

    
    int trained_data = 0;

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
        trained_data++;
    }

    inline void act_train(ll step,dl beta1t,dl beta2t){
        
        for(int i = 0;i<InSize;i++){
            for(int j = 0;j<OutSize;j++){
                dlW[i][j]/=(trained_data+0.1);
                mW[i][j] = 0.9*mW[i][j] + 0.1*dlW[i][j];
                vW[i][j] = 0.999*vW[i][j] + 0.001*dlW[i][j]*dlW[i][j];
                W[i][j] -= training_rate * (mW[i][j]/(1.0-beta1t))/(sqrt(vW[i][j]/(1.0-beta2t)) + 1e-6);
                W[i][j] *= (1-training_rate*1e-5);
                // adj(W[i][j]);
                dlW[i][j] = 0;
            }
        }
        for(int j = 0;j<OutSize;j++){
            dlB[j]/=(trained_data+0.1);
            mB[j] = 0.9*mB[j] + 0.1*dlB[j];
            vB[j] = 0.999*vB[j] + 0.001*dlB[j]*dlB[j];
            B[j] -= training_rate * (mB[j]/(1.0-beta1t))/(sqrt(vB[j]/(1.0-beta2t)) + 1e-6);
            B[j] *= (1-training_rate*1e-5);
            // adj(B[j]);
            dlB[j] = 0;
        }
        trained_data = 0;
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
    Layer<15,SZ> input_layer;
    Layer<SZ,SZ> hidden_layer[DEPTH];
    Layer<SZ,2> output_layer;
    acf<SZ> active_fc;
    dl V[DEPTH+1][SZ];

    dl beta1t = 1;
    dl beta2t = 1;

    ll step = 0;

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
        beta1t*=0.9;
        beta2t*=0.999;
        step++;
        input_layer.act_train(step,beta1t,beta2t);
        for(int i = 0;i<DEPTH;i++){
            hidden_layer[i].act_train(step,beta1t,beta2t);
        }
        output_layer.act_train(step,beta1t,beta2t);
    }
};

struct single_data{
    dl x[15];
    dl y[2];
};


int main(){
    vector<single_data> data(sample_sz);
    for(int i = 0;i<sample_sz;i++){
        dl y1 = random_z();
        dl y2 = random_z();
        data[i] = {
            {
                fixed_sin(y1*y2),
                fixed_sin(y1*y1*y2),
                fixed_sin(y1*y2*y2),
                fixed_sin(0.5*y1*y2),
                fixed_sin(0.5*y1*y1*y2),
                fixed_sin(0.5*y1*y2*y2),
                fixed_sin(0.25*y1*y2),
                fixed_sin(0.25*y1*y1*y2),
                fixed_sin(0.25*y1*y2*y2),
                fixed_sin(y1+y2),
                fixed_sin(0.5*(y1+y2)),
                fixed_sin(0.25*(y1+y2)),
                fixed_sin(y1-y2),
                fixed_sin(0.5*(y1-y2)),
                fixed_sin(0.25*(y1-y2))
            },
            {y1,y2}
        };
    }
    std::random_device rd;
    std::mt19937 g(rd()); 
    shuffle(data.begin(), data.end(), g);

    NN nn;
    for(int i = 0;i<50;i++){
        if(i<10){
            training_rate = 0.0001*(i+1);
        }else{
            training_rate = 0.001 / (1 + i * 0.1);
        }
        int t = 0;
        for(auto dd : data){
            nn.train(dd.x,dd.y);
            t++;
            if((t&127) == 0)nn.act_train();
        }
        nn.act_train();

        dl loss1 = 0,loss2 = 0;
        for(auto dd : data){
            dl val[2];
            nn.run(dd.x,val);
            loss1 += (val[0]-dd.y[0])*(val[0]-dd.y[0]);
            loss2 += (val[1]-dd.y[1])*(val[1]-dd.y[1]);
        }
        loss1 /= data.size();
        loss2 /= data.size();
        printf("st: %d,loss1 : %.8lf ,loss2 : %.8lf\n",i,loss1,loss2);
    }
    for(int i = 0; i <= 10; i++){
        for(int j = 0; j <= 10; j++){
            dl y1 = i/10.0;
            dl y2 = j/10.0;
            dl test_x[15] = {
                fixed_sin(y1*y2),
                fixed_sin(y1*y1*y2),
                fixed_sin(y1*y2*y2),
                fixed_sin(0.5*y1*y2),
                fixed_sin(0.5*y1*y1*y2),
                fixed_sin(0.5*y1*y2*y2),
                fixed_sin(0.25*y1*y2),
                fixed_sin(0.25*y1*y1*y2),
                fixed_sin(0.25*y1*y2*y2),
                fixed_sin(y1+y2),
                fixed_sin(0.5*(y1+y2)),
                fixed_sin(0.25*(y1+y2)),
                fixed_sin(y1-y2),
                fixed_sin(0.5*(y1-y2)),
                fixed_sin(0.25*(y1-y2))
            };
            dl prec[2];
            nn.run(test_x, prec);
            printf("Target: (%.2f, %.2f) | Prec: (%.4f, %.4f) | Error: %.6f\n", 
                    y1, y2, prec[0], prec[1], 
                    abs(y1-prec[0]) + abs(y2-prec[1]));
        }
    }
}