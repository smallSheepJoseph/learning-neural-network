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
#define SZ 45
#define DEPTH 6
#define sample_sz 20000
#define real_sample_sz (sample_sz + 1)

dl fixed_sin(dl x){
    return sin(x*2*3.14159265358979323846);
}

#define adj(x) if(abs(x)>1e10){x*=0.99;}if(x>1e16){x=1e16;}if(x<-1e16){x=-1e16;}

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
                dlW[i][j] = 0;
                adj(W[i][j]);
            }
        }
        for(int j = 0;j<OutSize;j++){
            dlB[j]/=(trained_data+0.1);
            mB[j] = 0.9*mB[j] + 0.1*dlB[j];
            vB[j] = 0.999*vB[j] + 0.001*dlB[j]*dlB[j];
            B[j] -= training_rate * (mB[j]/(1.0-beta1t))/(sqrt(vB[j]/(1.0-beta2t)) + 1e-6);
            B[j] *= (1-training_rate*1e-5);
            dlB[j] = 0;
            adj(B[j]);
        }
        trained_data = 0;
    }

    
};

//activation function
template<int Size>
struct acf{

    inline dl fc(dl x){
        // return tanh(x);
        if(x>0){
            return x;
        }else{
            return 0.01*x;
        }
    }

    // inline dl dfc(dl x){
    //     return 1.0 - tanh(x)*tanh(x);
    // }

    inline dl dfc(dl x){
        // return 1.0 - x*x;
        if(x>0){
            return 1;
        }else{
            return 0.01;
        }
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

#define NN_insz 1
#define NN_outsz 1
struct NN{
    
    Layer<NN_insz,SZ> input_layer;
    Layer<SZ,SZ> hidden_layer[DEPTH];
    Layer<SZ,NN_outsz> output_layer;
    acf<SZ> active_fc;
    acf<1> single;
    dl V[DEPTH+1][SZ];
    dl V2[DEPTH+1][SZ];
    dl temp_ans[NN_outsz];
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
        single.run(output,output);
    }

    void run_mul(dl* input,dl* output){
        input_layer.run(input,V[0]);
        active_fc.run(V[0],V[0]); //activation func can handle this

        for(int i = 0;i<DEPTH;i++){
            hidden_layer[i].run(V[i],V[i+1]);
            active_fc.run(V[i+1],V[i+1]); //activation func can handle this
        }
        output_layer.run(V[DEPTH],temp_ans);

        
        single.run(temp_ans,temp_ans);

        input_layer.run(temp_ans,V2[0]);
        active_fc.run(V2[0],V2[0]); 

        for(int i = 0;i<DEPTH;i++){
            hidden_layer[i].run(V2[i],V2[i+1]);
            active_fc.run(V2[i+1],V2[i+1]); 
        }
        output_layer.run(V2[DEPTH],output);
        single.run(output,output);
    }
    void train(dl* input,dl* real_value){
        dl output[NN_outsz];
        run_mul(input,output);
        dl dl_output[NN_outsz] = {};
        for(int i = 0;i<NN_outsz;i++){
            dl_output[i] = 2.0*(output[i]-real_value[i]);
        }
        // active_fc.rev(output,dl_output,dl_output);
        dl delta[SZ],delta2[SZ];
        dl* now_delta = delta2,*last_delta = delta;
        single.rev(output,dl_output,dl_output);
        output_layer.train(V2[DEPTH],dl_output,now_delta);
        swap(last_delta,now_delta);
        for(int i = DEPTH-1;i>=0;i--){
            active_fc.rev(V2[i+1],last_delta,last_delta);
            hidden_layer[i].train(V2[i],last_delta,now_delta);
            swap(last_delta,now_delta);
        }
        active_fc.rev(V2[0],last_delta,last_delta);
        input_layer.train(temp_ans,last_delta,now_delta);
        swap(last_delta,now_delta);

        for(int i = 0;i<NN_outsz;i++){
            if(temp_ans[i]<0){
                last_delta[i] += 2.0*(temp_ans[i]); // neg
            }
        }

        single.rev(temp_ans,last_delta,last_delta);
        output_layer.train(V[DEPTH],last_delta,now_delta);
        swap(last_delta,now_delta);
        for(int i = DEPTH-1;i>=0;i--){
            active_fc.rev(V[i+1],last_delta,last_delta);
            hidden_layer[i].train(V[i],last_delta,now_delta);
            swap(last_delta,now_delta);
        }
        active_fc.rev(V[0],last_delta,last_delta);
        input_layer.train(input,last_delta,now_delta);
        swap(last_delta,now_delta);
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
    dl x[1];
    dl y[1];
};


int main(){
    vector<single_data> data(sample_sz);
    static std::mt19937 gen(1337);
    static std::normal_distribution<double> dis(0, 40.0);
    static std::normal_distribution<double> dis2(0, 7.0);
    for(int i = 0;i<sample_sz/2;i++){
        
        dl x = abs(dis(gen))+0.01;
        data[i] = {
            {
                log(x)
            },
            {
                log((2*x*x+1)/3)
            }
        };
    }
    for(int i = sample_sz/2;i<sample_sz;i++){
        
        dl x = abs(dis2(gen))+0.01;
        data[i] = {
            {
                log(x)
            },
            {
                log((2*x*x+1)/3)
            }
        };
    }
    // std::random_device rd;
    // std::mt19937 g(rd()); 
    // shuffle(data.begin(), data.end(), g);

    NN nn;
    for(int i = 0;i<50;i++){
        if(i<20){
            training_rate = 0.00004*(i+1);
        }else{
            training_rate = 0.001 / (1 + i * 0.3);
        }
        int t = 0;
        for(auto dd : data){
            nn.train(dd.x,dd.y);
            t++;
            if((t&127) == 0)nn.act_train();
        }
        nn.act_train();

        dl loss = 0;
        for(auto dd : data){
            dl val[2];
            nn.run_mul(dd.x,val);
            loss += (val[0]-dd.y[0])*(val[0]-dd.y[0]);
        }
        loss /= data.size();
        printf("st: %d,loss : %.8lf\n",i,loss);
    }
    
    for(dl i = 0.5;i<=10.001;i+=0.5){
        dl test_x[1] = {
            i
        };
        dl prec[1];
        nn.run(test_x, prec);
        printf("f(%lf) = %lf\n",i,exp(prec[0]));
    }

    dl t = -1;
    scanf("%lf",&t);
    while(1){
        dl test_x[1] = {
            log(t)
        };
        dl prec[1],prec2[1];
        nn.run(test_x, prec);
        nn.run(prec, prec2);
        printf("f(f(%lf)) = %lf, prec: f(%lf) = %lf, f(f(%lf)) = %lf\n",t,(2*t*t+1)/3 , t,exp(prec[0]),t,exp(prec2[0]));
        scanf("%lf",&t);
    }
}