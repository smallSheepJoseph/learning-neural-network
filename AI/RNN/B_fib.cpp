#include<bits/stdc++.h>
using namespace std;
using dl = double;
using ll = long long;
#define ff first
#define ss second
// double random(){
//     return (double)rand() / 32767 + (double)rand() / (32767*32767.0);
// }

static std::mt19937 gen(1337);
double random_z() {
    // static std::mt19937 gen(1337);
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}


// #define sampleSize 100000

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
        std::normal_distribution<dl> dis(0.0, std::sqrt(2.0 / (InSize+OutSize)));
        
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

    //run this layer
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

    //only calculate gradient
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
    // y: the result from layer/step
    // input: the gradient travle from the next layer/step
    // output: the gradient to last layer
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

    //active the weight changes using adam optimizer
    inline void act_train(ll step,dl beta1t,dl beta2t){
        (void)step; // don't need yet
        if(trained_data == 0){
            printf("Warn: tried to active train on a layer with no trained data\n");
            return;
        }

        dl rev_trained_data = 1.0/trained_data; // trained_data is NOT zero
        for(int i = 0;i<InSize;i++){
            for(int j = 0;j<OutSize;j++){
                dlW[i][j]*=rev_trained_data;
                mW[i][j] = 0.9*mW[i][j] + 0.1*dlW[i][j];
                vW[i][j] = 0.999*vW[i][j] + 0.001*dlW[i][j]*dlW[i][j];
                W[i][j] -= training_rate * (mW[i][j]/(1.0-beta1t))/(sqrt(vW[i][j]/(1.0-beta2t)) + 1e-6);
                W[i][j] *= (1-training_rate*1e-5);
                // adj(W[i][j]);
                dlW[i][j] = 0;
            }
        }
        
        for(int j = 0;j<OutSize;j++){
            dlB[j]*=rev_trained_data;
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
        if(x>0){
            return x;
        }else{
            return 0.02*x;
        }
    }

    // inline dl dfc(dl x){
    //     return 1.0 - tanh(x)*tanh(x);
    // }

    //input : tanh(x)
    //because tanh'(x) = 1-(tanh(x))^2
    //reduce the calcution 


    //use Leaky ReLU instead cuz why not
    inline dl dfc(dl x){
        if(x>0){
            return 1;
        }else{
            return 0.02;
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


struct TrainConfig {
    bool enable_fix_gradient = false;
    dl clip_low = 0.1, clip_high = 10;
    ll step = 0;
};

//gradient fixer
struct gradient_fixer{
    TrainConfig& config;
    gradient_fixer(TrainConfig &tconfig) : config(tconfig) {
        //pass
    }
    void fix(dl* gradient,int Size){
        if(!config.enable_fix_gradient)return;
        dl g2 = 0;
        for(int i = 0;i<Size;i++){
            g2 += gradient[i]*gradient[i];
        }
        dl g = sqrt(g2);
        if(g < config.clip_low || g > config.clip_high){
            dl target = min((dl)config.clip_high, max((dl)config.clip_low, g));
            dl factor = target / (g+1e-6);
            for(int i = 0;i<Size;i++){
                gradient[i] *= factor;
            }
        }
    }
};



template<int InSize,int HiddenSize,int OutSize,int depth>
struct NN{
    Layer<InSize,HiddenSize> input_layer;
    Layer<HiddenSize,HiddenSize> hidden_layer[depth];
    Layer<HiddenSize,OutSize> output_layer;
    acf<HiddenSize> active_fc;

    dl V[depth+1][HiddenSize];
    dl Vinput[InSize];

    TrainConfig &config;
    gradient_fixer gfix;
    NN(TrainConfig &tconfig)
        : config(tconfig),gfix(tconfig){
            //pass
    }

    // dl beta1t = 1;
    // dl beta2t = 1;
    // ll step = 0;

    bool enable_fix_gradient = false;

    const int input_len = InSize, hidden_len = HiddenSize, output_len = OutSize, _depth = depth;

    void run(dl* input,dl* output){
        memcpy(Vinput,input,InSize * sizeof(dl));
        input_layer.run(input,V[0]);
        active_fc.run(V[0],V[0]); //activation func can handle this

        for(int i = 0;i<depth;i++){
            hidden_layer[i].run(V[i],V[i+1]);
            active_fc.run(V[i+1],V[i+1]); //activation func can handle this
        }
        output_layer.run(V[depth],output);
    }

    //copy the data that can be use to train later
    void copy_record(dl (*target_V)[HiddenSize],dl* target_Vinput){
        memcpy(target_V, V, sizeof(V));
        memcpy(target_Vinput, Vinput, sizeof(Vinput));
    }

    //only calculate the gradient
    //input the gradiend from next step(or another NN), and output the gradiend of last step
    //this function is not safe, not fixed yet
    void rev(dl* input, dl* output){
        dl gradient[HiddenSize],gradient2[HiddenSize];
        output_layer.rev(input,gradient);
        dl* now_gradient = gradient2,*last_gradient = gradient;
        for(int i = depth-1;i>=0;i--){
            active_fc.rev(V[i+1],last_gradient,last_gradient);
            hidden_layer[i].rev(last_gradient,now_gradient);
            swap(last_gradient,now_gradient);
        }
        active_fc.rev(V[0],last_gradient,last_gradient);
        input_layer.rev(input,last_gradient,output);
        gfix.fix(output,InSize); // output last gradiend to front, so it's InSize
        //memcpy(output, now_gradient ,sizeof(dl)*InSize);
    }
    
    //train this NN directly using copied data and gradient
    void train_directly(dl (*copied_V)[HiddenSize],dl* copied_Vinput, dl* dl_output, dl* dl_last = nullptr){
        dl gradient[HiddenSize],gradient2[HiddenSize];
        output_layer.train(copied_V[depth],dl_output,gradient);
        dl* now_gradient = gradient2,*last_gradient = gradient;
        for(int i = depth-1;i>=0;i--){
            gfix.fix(last_gradient,HiddenSize);
            active_fc.rev(copied_V[i+1],last_gradient,last_gradient);
            hidden_layer[i].train(copied_V[i],last_gradient,now_gradient);
            swap(last_gradient,now_gradient);
        }
        gfix.fix(last_gradient,HiddenSize);
        active_fc.rev(copied_V[0],last_gradient,last_gradient);
        input_layer.train(copied_Vinput,last_gradient,now_gradient);
        //copy to output gradient
        if(dl_last != nullptr){
            memcpy(dl_last, now_gradient, sizeof(dl) * InSize);
        }
    }

    //train this NN using real data
    void train(dl* input, dl* real_value, dl* dl_last = nullptr){
        dl pred[OutSize];               
        run(input, pred);
        dl dl_output[OutSize];
        for (int i = 0; i < OutSize; ++i) dl_output[i] = 2.0 * (pred[i] - real_value[i]);
        train_directly(V, Vinput, dl_output, dl_last);
    }

    //active the weight changes
    void active_train(dl beta1t,dl beta2t,ll step){
        // beta1t*=0.9;
        // beta2t*=0.999;
        // step++;
        input_layer.act_train(step,beta1t,beta2t);
        for(int i = 0;i<depth;i++){
            hidden_layer[i].act_train(step,beta1t,beta2t);
        }
        output_layer.act_train(step,beta1t,beta2t);
    }
};


template<int InSize,int HiddenSize,int OutSize,int MemorySize,int DataLen,int depth>
struct RNN{
    NN<InSize+MemorySize,HiddenSize,OutSize+MemorySize,depth> nn;
    dl V[DataLen][depth+1][HiddenSize];
    dl Vinput[DataLen][InSize];


    dl h0[MemorySize];
    dl dlh0[MemorySize];
    dl mh0[MemorySize];
    dl vh0[MemorySize];
    
    bool enable_fix_gradient = false;

    static constexpr int BUF = (InSize + MemorySize > OutSize + MemorySize)
                    ? (InSize + MemorySize) : (OutSize + MemorySize);


    TrainConfig &config;
    gradient_fixer gfix;
    RNN(TrainConfig &tconfig)
        : nn(tconfig),config(tconfig),gfix(tconfig){
        
        memset(h0,0,sizeof(dl)*MemorySize);
        memset(dlh0,0,sizeof(dl)*MemorySize);
        memset(mh0,0,sizeof(dl)*MemorySize);
        memset(vh0,0,sizeof(dl)*MemorySize);
    }

    //run this RNN and save data
    void run(dl (*input)[InSize],dl (*output)[OutSize],int len){
        assert(len<=DataLen);
        dl tmp[BUF],tmp2[BUF];
        memcpy(tmp,h0,sizeof(dl)*MemorySize);
        dl *last_data = tmp, *now_data = tmp2;
        for(int i = 0;i<len;i++){
            memcpy(last_data+MemorySize,input[i],sizeof(dl)*InSize);
            nn.run(last_data,now_data);
            nn.copy_record(V[i],Vinput[i]);
            memcpy(output[i],now_data+MemorySize,sizeof(dl)*OutSize);
            swap(last_data, now_data);
        }
    }

    void train(dl (*input)[InSize],dl (*real_value)[OutSize],int len,int train_stoppoint){
        dl pred[DataLen][OutSize];
        run(input,pred,len);
        dl gradient[BUF],gradient2[BUF];
        memset(gradient,0,sizeof(dl)*BUF);
        memset(gradient2,0,sizeof(dl)*BUF);
        dl *last_gradient = gradient, *now_gradient = gradient2;
        for(int i = len-1;i>=0;i--){
            if(i>=train_stoppoint){
                for(int j = 0;j<OutSize;j++){
                    last_gradient[MemorySize+j] = 2.0*(pred[i][j] - real_value[i][j]);
                }
                gfix.fix(last_gradient,MemorySize+OutSize);
            }else{
                memset(last_gradient+MemorySize,0,sizeof(dl)*OutSize);
                gfix.fix(last_gradient,MemorySize);
            }
            
            nn.train_directly(V[i],Vinput[i],last_gradient,now_gradient);
            swap(last_gradient,now_gradient);
        }
        for(int i = 0;i<MemorySize;i++){
            dlh0[i]+=last_gradient[i];
        }
        
    }


    dl beta1t = 1;
    dl beta2t = 1;
    ll step = 0;

    void active_train(){
        step++;
        beta1t*=0.9;
        beta2t*=0.999;
        nn.active_train(beta1t,beta2t,step);

        for(int i = 0;i<MemorySize;i++){
            mh0[i] = 0.9*mh0[i] + 0.1*dlh0[i];
            vh0[i] = 0.999*vh0[i] + 0.001*dlh0[i]*dlh0[i];
            h0[i] -= training_rate * (mh0[i]/(1.0-beta1t))/(sqrt(vh0[i]/(1.0-beta2t)) + 1e-6);
        }
        memset(dlh0,0,sizeof(dl)*MemorySize);
    }

};


template<int InSize,int OutSize>
struct NN_data{
    dl x[InSize];
    dl y[OutSize];
};

template<int InSize,int OutSize,int Len>
struct RNN_data{
    dl x[Len][InSize]={};
    dl y[Len][OutSize]={};
};

using Data = RNN_data<5,5,100>;

int main(){
    TrainConfig config;
    auto rnn = make_unique<RNN<5, 20, 5, 7, 100, 4>>(config);
    config.enable_fix_gradient = true;
    auto train_data_ptr = make_unique<array<Data, 25>>();
    auto& train_data = *train_data_ptr;
    int t1 = 0,t2 = 0;
    for(int t = 0;t<25;t++){
        char arr[101];
        arr[0] = t1;arr[1] = t2;
        for(int i = 2;i<101;i++){
            arr[i] = (arr[i-1] + arr[i-2] + 1)%5;
        }
        for(int i = 0 ;i<100;i++){
            train_data[t].x[i][(int)arr[i]] = 1.0;
            train_data[t].y[i][(int)arr[i+1]] = 1.0;
        }
        t1++;
        t2+=t1/5;
        t1%=5;
    }
    for(int t = 0;t<50;t++){
        for(int i = 0;i<1000;i++){
            for(auto td:train_data){
                rnn->train(td.x,td.y,100,1);
            }
            rnn->active_train();
        }
        dl loss = 0;
        int err_cnt = 0;
        for(auto td:train_data){
            dl pred[100][5];
            rnn->run(td.x,pred,100);
            for(int i = 2;i<100;i++){
                int pred_ans = 0;
                for(int j = 0;j<5;j++){
                    if(pred[i][j]>pred[i][pred_ans])pred_ans = j;
                    loss += (pred[i][j] - td.y[i][j])*(pred[i][j] - td.y[i][j]);
                }
                if(td.y[i][pred_ans]<0.5){
                    err_cnt++;
                }
            }
        }
        dl err_rate = (dl)err_cnt/(25.0 * 98.0) * 100.0;
        loss /= 25 * 98 * 5;
        printf("st: %d, loss: %.20lf, err: %lf%%\n",t,loss,err_rate);
    }

    while(1){
        int a = 0;
        int b = 0;
        dl input[20][5]={};
        scanf("%d %d",&a,&b);
        if(a<0 || a>=5 || b<0 || b>=5){
            printf("don't do this\n");
            continue;
        }
        char arr[21];
        arr[0] = a;
        arr[1] = b;
        for(int i = 2;i<21;i++){
            arr[i] = (arr[i-1] + arr[i-2] + 1)%5;
        }
        for(int i = 0;i<20;i++){
            input[i][(int)arr[i]] = 1.0;
        }
        dl pred[20][5];
        rnn->run(input,pred,20);
        printf("pred: ");
        for(int i = 1;i<20;i++){
            int ans = 0;
            for(int j = 0;j<5;j++){
                if(pred[i][j]>pred[i][ans])ans = j;
            }
            printf("%d ",ans);
        }printf("\n");
        printf("ans : ");
        for(int i = 1;i<20;i++){
            printf("%d ",arr[i+1]);
        }printf("\n");
    }

}