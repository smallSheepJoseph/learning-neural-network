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


#define sampleSize 100000

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
        std::normal_distribution<dl> dis(0.0, std::sqrt(2.0 / (InSize/*+OutSize*/)));
        
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
                adj(W[i][j]);
                dlW[i][j] = 0;
            }
        }
        
        for(int j = 0;j<OutSize;j++){
            dlB[j]*=rev_trained_data;
            mB[j] = 0.9*mB[j] + 0.1*dlB[j];
            vB[j] = 0.999*vB[j] + 0.001*dlB[j]*dlB[j];
            B[j] -= training_rate * (mB[j]/(1.0-beta1t))/(sqrt(vB[j]/(1.0-beta2t)) + 1e-6);
            B[j] *= (1-training_rate*1e-5);
            adj(B[j]);
            dlB[j] = 0;
        }
        trained_data = 0;
    }

    
};

//activation function
template<int Size>
struct acf{

    // inline dl fc(dl x){
    //     return tanh(x);
    // }

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
    // inline dl dfc(dl x){
    //     return 1.0 - x*x;
    // }

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

//gradient fixer
template<int Size>
void fix_gradient(dl* gradient){
    dl g2 = 0;
    for(int i = 0;i<Size;i++){
        g2 += gradient[i]*gradient[i];
    }

    if(g2 < 1 || g2 > 100){
        dl target = sqrt(min((dl)100, max((dl)1, g2)));
        dl factor = target / sqrt(g2 + 1e-8);
        for(int i = 0;i<Size;i++){
            gradient[i] *= factor;
        }
    }
}


template<int InSize,int HiddenSize,int OutSize,int depth>
struct NN{
    Layer<InSize,HiddenSize> input_layer;
    Layer<HiddenSize,HiddenSize> hidden_layer[depth];
    Layer<HiddenSize,OutSize> output_layer;
    acf<HiddenSize> active_fc;
    dl V[depth+1][HiddenSize];
    dl Vinput[InSize];

    dl beta1t = 1;
    dl beta2t = 1;

    ll step = 0;

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
    void copy_record(dl* target_V,dl* target_Vinput){
        memcpy(target_V, V, sizeof(V));
        memcpy(target_Vinput, Vinput, sizeof(Vinput));
    }

    //only calculate the gradient
    //input the gradiend from next step(or another NN), and out put the gradiend of last step
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
        input_layer.rev(input,last_gradient,now_gradient);
        if(enable_fix_gradient)fix_gradient<OutSize>(gradient);
        memcpy(output, now_gradient ,sizeof(gradient));
        
    }

    //train this NN directly using copied data and gradient
    void train_directly(dl (*copied_V)[HiddenSize],dl* copied_Vinput, dl* dl_output, dl* dl_last = nullptr){
        dl gradient[HiddenSize],gradient2[HiddenSize];
        output_layer.train(copied_V[depth],dl_output,gradient);
        dl* now_gradient = gradient2,*last_gradient = gradient;
        for(int i = depth-1;i>=0;i--){
            if(enable_fix_gradient)fix_gradient<HiddenSize>(last_gradient);
            active_fc.rev(copied_V[i+1],last_gradient,last_gradient);
            hidden_layer[i].train(copied_V[i],last_gradient,now_gradient);
            swap(last_gradient,now_gradient);
        }
        if(enable_fix_gradient)fix_gradient<HiddenSize>(last_gradient);
        active_fc.rev(copied_V[0],last_gradient,last_gradient);
        input_layer.train(copied_Vinput,last_gradient,now_gradient);
        //copy to output gradient
        if(dl_last != nullptr){
            memcpy(dl_last, now_gradient ,sizeof(gradient));
        }
    }

    //train this NN using real data, return loss
    dl train(dl* input, dl* real_value, dl* dl_last = nullptr){
        dl pred[OutSize];               
        run(input, pred);
        dl dl_output[OutSize];
        dl loss = 0;
        for (int i = 0; i < OutSize; ++i){
            dl_output[i] = 2.0 * (pred[i] - real_value[i]);
            loss += dl_output[i]*dl_output[i];
        }
        train_directly(V, Vinput, dl_output, dl_last);
        return loss/OutSize;
    }

    //active the weight changes
    void active_train(){
        beta1t*=0.9;
        beta2t*=0.999;
        step++;
        input_layer.act_train(step,beta1t,beta2t);
        for(int i = 0;i<depth;i++){
            hidden_layer[i].act_train(step,beta1t,beta2t);
        }
        output_layer.act_train(step,beta1t,beta2t);
    }
};


template<int InSize,int OutSize>
struct single_data{
    dl x[InSize];
    dl y[OutSize];
};


int main(){
    vector<single_data<2,1>> train_data(sampleSize);
    for(int i = 0;i<sampleSize;i++){
        dl x1 = random_z(),x2 = random_z();
        train_data[i]={
            {
                x1,x2
            },
            {
                (x1+x2)*0.5
            }
        };
    }
    NN<2,8,1,200> nn;
    nn.enable_fix_gradient = true;
    vector<queue<int>> rtrain_data(5);
    for(int i = 0;i<sampleSize;i++){
        rtrain_data[0].push(i);
    }
    // auto train_helper = [&](single_data<2,1> td){
    //     nn.train(td.x,td.y);
    // };
    dl loss_gap = 0.5;
    int cnt = 0;
    for(int t = 0;cnt<sampleSize*100;t++){
        if(t/sampleSize<20){
            training_rate = 0.0001*((dl)t/(dl)sampleSize+1);
        }else{
            training_rate = 0.001 / (1 + (dl)t/(dl)sampleSize * 0.5);
        }
        if(cnt/sampleSize>50){
            nn.enable_fix_gradient = false;
        }

        if(t>0 && cnt%100 == 0)nn.active_train();
        if(t>0 && cnt%sampleSize == 0){
            dl loss = 0;
            for(auto [x,y]:train_data){
                dl pred[1];
                nn.run(x,pred);
                loss += (pred[0]-y[0])*(pred[0]-y[0]);
            }
            printf("st: %d cnt: %d loss: %.20lf loss_gap: %.20lf\n",t,cnt,loss/sampleSize,loss_gap);
        }


        int td_idx = -1;
        int dtype = -1;
        
        dl total = rtrain_data[0].size()*16 +
                   rtrain_data[1].size()*8 +
                   rtrain_data[2].size()*4 +
                   rtrain_data[3].size()*2 +
                   rtrain_data[4].size()*1;
        dl rr = random_z()*total;
        if((rr -= rtrain_data[0].size()*16) < 0){
            if(rtrain_data[0].empty()){continue;}
            td_idx = rtrain_data[0].front();rtrain_data[0].pop();
            dtype = 0;
        }else if((rr -= rtrain_data[1].size()*8) < 0){
            if(rtrain_data[1].empty()){continue;}
            td_idx = rtrain_data[1].front();rtrain_data[1].pop();
            dtype = 1;
        }else if((rr -= rtrain_data[2].size()*4) < 0){
            if(rtrain_data[2].empty()){continue;}
            td_idx = rtrain_data[2].front();rtrain_data[2].pop();
            dtype = 2;
        }else if((rr -= rtrain_data[3].size()*2) < 0){
            if(rtrain_data[3].empty()){continue;}
            td_idx = rtrain_data[3].front();rtrain_data[3].pop();
            dtype = 3;
        }else{
            if(rtrain_data[4].empty()){continue;}
            td_idx = rtrain_data[4].front();rtrain_data[4].pop();
            dtype = 4;
        }

        cnt++;
        single_data<2,1> td = train_data[td_idx];
        dl single_loss = nn.train(td.x,td.y);
        if(single_loss>loss_gap){
            dtype = 0;
        }else{
            dtype = min(4,dtype+1);
        }
        rtrain_data[dtype].push(td_idx);
        loss_gap = loss_gap*(((dl)sampleSize-1.0)/(dl)sampleSize) + single_loss*(1.0/(dl)sampleSize);
        // for(auto td:train_data){
        //     train_helper(td);
        //     cnt++;
        //     if(cnt>=100){
        //         nn.active_train();
        //         cnt = 0;
        //     }
        // }

        
        
    }

}