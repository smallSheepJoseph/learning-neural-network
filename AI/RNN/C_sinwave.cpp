#include<bits/stdc++.h>
using namespace std;
using dl = double;
// using dl = float;
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
        // std::normal_distribution<dl> dis(0.0, std::sqrt(2.0 / (1.0004*InSize)));
        
        for (int i = 0; i < InSize; ++i){
            for (int j = 0; j < OutSize; ++j){
                W[i][j] = dis(gen);
                dlW[i][j] = 0;
            }
        }
        for (int j = 0; j < OutSize; ++j){
            // B[j] = dis(gen);
            B[j] = 0;

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
    //     if(x>0){
    //         return x;
    //     }else{
    //         return 0.02*x;
    //     }
    // }

    // inline dl dfc(dl x){
    //     if(x>0){
    //         return 1;
    //     }else{
    //         return 0.02;
    //     }
    // }


    inline dl fc(dl x){
        return tanh(x);
    }

    inline dl dfc(dl x){
        return 1-x*x;
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
        run(input,output,V,Vinput);
    }

    void run(dl* input,dl* output,dl (*target_V)[HiddenSize],dl* target_Vinput){
        memcpy(target_Vinput,input,InSize * sizeof(dl));
        input_layer.run(input,target_V[0]);
        active_fc.run(target_V[0],target_V[0]); //activation func can handle this

        for(int i = 0;i<depth;i++){
            hidden_layer[i].run(target_V[i],target_V[i+1]);
            active_fc.run(target_V[i+1],target_V[i+1]); //activation func can handle this
        }
        output_layer.run(target_V[depth],output);
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
        input_layer.rev(input,output);
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

            // if(i!=len-1){
            //     dl sqg = 0;
            //     for(int j = 0;j<MemorySize;j++){
            //         sqg += last_gradient[j]*last_gradient[j];
            //     }
            //     sqg/=MemorySize+0.01;
            //     dl sqm = 0;
            //     for(int j = 0;j<MemorySize;j++){
            //         sqm += (Vinput[i+1][j] - Vinput[i][j])*(Vinput[i+1][j] - Vinput[i][j]);
            //     }
            //     sqm/=MemorySize+0.01;
            //     for(int j = 0;j<MemorySize;j++){
            //         last_gradient[j] += 0.2*(Vinput[i+1][j] - Vinput[i][j])/sqrt(sqm+0.001)*sqrt(sqg);
            //     }
            // }

template<int InSize,int HiddenSize,int OutSize,int MemorySize,int DataLen,int depth>
struct RNN{
    using Self = RNN<InSize, HiddenSize, OutSize, MemorySize, DataLen, depth>;
    NN<InSize+MemorySize,HiddenSize,OutSize+MemorySize,depth> nn;
    static constexpr int BUF = (InSize + MemorySize > OutSize + MemorySize)
                    ? (InSize + MemorySize) : (OutSize + MemorySize);
    dl V[DataLen][depth+1][HiddenSize];
    dl Vinput[DataLen][BUF];


    dl h0[MemorySize];
    dl dlh0[MemorySize];
    dl mh0[MemorySize];
    dl vh0[MemorySize];
    
    bool enable_fix_gradient = false;

    


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
            nn.run(last_data,now_data,V[i],Vinput[i]);
            // nn.copy_record(V[i],Vinput[i]);
            memcpy(output[i],now_data+MemorySize,sizeof(dl)*OutSize);
            swap(last_data, now_data);
        }
    }

    void run_self_pred(dl (*input)[InSize],dl (*output)[OutSize],int len,int given_data){
        assert(len<=DataLen);
        assert(InSize == OutSize);
        dl tmp[BUF],tmp2[BUF];
        memcpy(tmp,h0,sizeof(dl)*MemorySize);
        dl *last_data = tmp, *now_data = tmp2;
        for(int i = 0;i<len;i++){
            if(i<given_data){
                memcpy(last_data+MemorySize,input[i],sizeof(dl)*InSize);
            }
            nn.run(last_data,now_data,V[i],Vinput[i]);
            // nn.copy_record(V[i],Vinput[i]);
            memcpy(output[i],now_data+MemorySize,sizeof(dl)*OutSize);
            swap(last_data, now_data);
        }
    }

    dl train(dl (*input)[InSize],dl (*real_value)[OutSize],int len,int train_stoppoint,int self_pred_point = -1){
        dl pred[DataLen][OutSize];
        if(self_pred_point!=-1){
            assert(InSize == OutSize);
            run_self_pred(input,pred,len,self_pred_point);
        }else{
            run(input,pred,len);
        }
        dl loss = 0;
        dl loss_cnt = 0;
        dl gradient[BUF],gradient2[BUF];
        memset(gradient,0,sizeof(dl)*BUF);
        memset(gradient2,0,sizeof(dl)*BUF);
        dl *last_gradient = gradient, *now_gradient = gradient2;
        for(int i = len-1;i>=0;i--){
            if(i>=train_stoppoint){
                for(int j = 0;j<OutSize;j++){
                    last_gradient[MemorySize+j] = 2.0*(pred[i][j] - real_value[i][j]);
                    loss += (pred[i][j] - real_value[i][j]) * (pred[i][j] - real_value[i][j]);
                    loss_cnt += 1;
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
        return loss/loss_cnt;
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

template<typename T>
struct RNN_RUNNER;

template<int InSize, int HiddenSize, int OutSize, int MemorySize, int DataLen, int depth>
struct RNN_RUNNER<RNN<InSize, HiddenSize, OutSize, MemorySize, DataLen, depth>> {
    using RNN_Type = RNN<InSize, HiddenSize, OutSize, MemorySize, DataLen, depth>;
    static constexpr int BUF = (InSize + MemorySize > OutSize + MemorySize)
                    ? (InSize + MemorySize) : (OutSize + MemorySize);
    
    dl memory[BUF];
    
    RNN_Type& rnn;
    RNN_RUNNER(RNN_Type &trnn) : rnn(trnn) {
        reset();
    }

    void reset(){
        memset(memory,0,sizeof(dl)*BUF);
        memcpy(memory,rnn.h0,sizeof(dl)*MemorySize);
    }
    
    void step(dl* input, dl* output){
        memcpy(memory+MemorySize,input,sizeof(dl)*InSize);
        rnn.nn.run(memory,memory);
        memcpy(output,memory+MemorySize,sizeof(dl)*OutSize);
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

using Data = RNN_data<1,1,100>;

Data gen_data(dl a,dl b,dl rd=0.0){
    Data d;
    static normal_distribution<dl> distribution(0.0, rd);
    for(int i = 0;i<100;i++){
        d.x[i][0] = fixed_sin(a + b*i);
        if(rd>0)d.x[i][0] += distribution(gen);
        d.y[i][0] = fixed_sin(a + b*(i+1));
    }
    // for(int i = 0;i<100;i++){
    //     d.x[i][0] = i%2;
    //     d.y[i][0] = (i+1)%2;
    // }
    return d;
}

template<int data_size,int batch>
struct hard_sample_mine{
    vector<queue<int>> qu{batch};
    dl avg = 0.5;
    dl late_avg = 0.75;
    hard_sample_mine(){
        for(int i = 0;i<data_size;i++){
            qu[0].push(i);
        }
    }
    void reset(){
        avg = 0.5;
        late_avg = 0.75;
        for(int i = 0;i<batch;i++){
            while(!qu[i].empty()) qu[i].pop();
        }
        for(int i = 0;i<data_size;i++){
            qu[0].push(i);
        }
    }
    pair<int,int> get_index(dl mul){
        assert(mul>=1);
        dl sum = 0.0;
        dl base = 1;
        for(int i = batch-1;i>=0;i--){
            sum += base*(dl)qu[i].size();
            base*=mul;
        }
        dl rr = random_z()*sum;
        int batch_id = 0;
        base = 1;
        for(int i = batch-1;i>=0;i--){
            rr -= base*(dl)qu[i].size();
            if(rr<=0){
                batch_id = i;
                break;
            }
            base*=mul;
        }
        if(qu[batch_id].size() == 0){
            printf("\nWarning: found a empty batch\n");
            return {-1,-1};
        }
        int return_id = qu[batch_id].front();
        qu[batch_id].pop();
        return {return_id,batch_id};
    }

    void push(int id,int batch_id,dl hardness){
        if(hardness>avg){
            batch_id--;
            late_avg = 0.97*late_avg + 0.03*hardness;
        }else{
            batch_id++;
        }
        if(hardness>late_avg){
            batch_id = 0;
        }
        avg = 0.97*avg + 0.03*hardness;
        batch_id = min(max(0,batch_id),batch-1);
        qu[batch_id].push(id);
    }
};

int main(){
    printf("loading...\n");
    TrainConfig config;
    auto rnn_ptr = make_unique<RNN<1, 40, 1, 39, 100, 5>>(config);
    auto& rnn = *rnn_ptr;
    config.enable_fix_gradient = true;
    constexpr int n = 100000;
    auto train_data_ptr = make_unique<array<Data, n>>();
    auto train_data_ptr2 = make_unique<array<Data, n>>();
    auto train_data_ptr3 = make_unique<array<Data, n>>();
    auto train_data_ptr4 = make_unique<array<Data, n>>();
    array<Data, n>* train_ptr = train_data_ptr.get();
    hard_sample_mine<n,5> hsm;hsm.reset();
    printf("gen data...\n");
    normal_distribution<dl> distribution(0.15, 0.02);

    for(int i = 0;i<n;i++){
        dl a = random_z();
        dl b = 0.15;
        (*train_data_ptr)[i] = gen_data(a,b);

    }
    for(int i = 0;i<n;i++){
        dl a = random_z();
        dl b = min(max(distribution(gen),0.1),0.3);
        (*train_data_ptr2)[i] = gen_data(a,b);
    }
    for(int i = 0;i<n;i++){
        dl a = random_z();
        dl b = min(random_z(),random_z())*0.2+0.1;
        (*train_data_ptr3)[i] = gen_data(a,b);
    }
    for(int i = 0;i<n;i++){
        dl a = random_z();
        dl b = min(random_z(),random_z())*0.4+0.05;
        (*train_data_ptr4)[i] = gen_data(a,b);
    }

    printf("start training\n");
    int cnt = 0;
    int self_pred = 0;
    int cool_down = 0;
    int end_cnt = 0;
    dl curr_loss = 100.0;
    config.clip_low = 0.5;
    printf("=====dataset 1 training started=====\n\n");
    for(int t = 0;true;t++){
        if(t<20){
            training_rate = 0.000002*(t+1);
        }else if(t<51){
            training_rate = 0.0002 / sqrt(t);
        }
        
        config.clip_high = 5.0;
        
        if(t>50){
            training_rate = 0.00002 * pow(0.9,self_pred) / sqrt(t);
            if(curr_loss>=0.05){
                cool_down = 3;
            }
            if(curr_loss<0.03 && cool_down <= 0){
                self_pred = min(self_pred + 1, 95);
                cool_down = 5;
                config.clip_low = 0;
            }
            if (curr_loss > 0.1) {
                self_pred = max(self_pred - 1, 0);
                cool_down = 5;
                config.clip_low = 0.1;
            }
            if(self_pred == 95){
                config.clip_low = 0;
                end_cnt++;
                if(end_cnt>50){
                    printf("train finished\n\n");
                    break;
                }
            }else{
                end_cnt--;
            }
            if(cool_down)cool_down--;
        }
        if(t==15){
            printf("=====dataset 2 training started=====\n\n");
            train_ptr = train_data_ptr2.get();
            hsm.reset();
        }
        if(t==150){
            printf("=====dataset 3 training started=====\n\n");
            train_ptr = train_data_ptr3.get();
            self_pred = max(self_pred - 12, 0);
            cool_down = 10;
            hsm.reset();
        }
        if(t==250){
            printf("=====dataset 4 training started=====\n\n");
            train_ptr = train_data_ptr4.get();
            self_pred = max(self_pred - 12, 0);
            cool_down = 10;
            hsm.reset();
        }
        for(int i = 0;i<3000;i++){
            auto [id,batch_id] = hsm.get_index(2.71828);
            if(id < 0) continue;
            auto td = (*train_ptr)[id];
            dl hardness = rnn.train(td.x,td.y,100,5,100-self_pred);
            // hardness = min(hardness,5.0);
            hsm.push(id,batch_id,hardness);
            cnt++;
            if((cnt&7)==0)rnn.active_train();
        }

        dl loss = 0;
        dl sel_loss = 0;
        dl sl_loss1 = 0;dl sl_loss2 = 0;dl sl_loss3 = 0;
        curr_loss = 0;
        for(int i = 0;i<min(200,n);i++){
            auto& td = (*train_ptr)[i];
            dl pred[100][1];
            rnn.run(td.x,pred,100);
            for(int i = 5;i<100;i++){
                loss+=(pred[i][0]-td.y[i][0])*(pred[i][0]-td.y[i][0]);
            }
            rnn.run_self_pred(td.x,pred,100,5);
            for(int i = 5;i<100;i++){
                sel_loss+=(pred[i][0]-td.y[i][0])*(pred[i][0]-td.y[i][0]);
                if(5<=i && i<20){
                    sl_loss1+=(pred[i][0]-td.y[i][0])*(pred[i][0]-td.y[i][0]);
                }else if(20<=i && i<80){
                    sl_loss2+=(pred[i][0]-td.y[i][0])*(pred[i][0]-td.y[i][0]);
                }else{
                    sl_loss3+=(pred[i][0]-td.y[i][0])*(pred[i][0]-td.y[i][0]);
                }
            }
            rnn.run_self_pred(td.x,pred,100,100-self_pred);
            for(int i = 100-self_pred;i<100;i++){
                curr_loss+=(pred[i][0]-td.y[i][0])*(pred[i][0]-td.y[i][0]);
            }
        }
        loss /= 95.0 * (dl)200;
        sel_loss /= 95.0 * (dl)200;
        sl_loss1 /= 15.0 * (dl)200;
        sl_loss2 /= 60.0 * (dl)200;
        sl_loss3 /= 20.0 * (dl)200;
        curr_loss /= self_pred * 200.0;
        if(self_pred == 0){
            curr_loss = loss;
        }
        printf("st: %d, pred: %d, loss: %lf, curr loss: %lf, self pred loss: %lf\n",t,self_pred,loss,curr_loss,sel_loss);
        printf("pred loss: begin: %lf  mid: %lf  end: %lf\n",sl_loss1,sl_loss2,sl_loss3);
        printf("hsm: avg: %lf avg_late: %lf\n\n",hsm.avg,hsm.late_avg);
    }
    //RNN_RUNNER<remove_reference_t<decltype(rnn)>> runner(rnn); //wtf is this
    while(1){
        //runner.reset();
        dl a,b;
        printf("input phase and frequency:\n");
        scanf("%lf %lf",&a,&b);
        printf("\ninput start pred pos \n");
        int dd;
        scanf("%d",&dd);
        printf("\n");
        if(dd<0 || dd>100){
            printf("NO\n");
            continue;
        }
        Data td = gen_data(a,b);
        int i = 0;
        dl pred[100][1];
        rnn.run_self_pred(td.x,pred,100,dd);
        printf("\n\nans: ");
        for(i = 0;i<100;i++){
            printf("%lf ",td.y[i][0]);
        }printf("\n\n");
        printf("pred:");
        for(i = 0;i<100;i++){
            printf("%lf ",pred[i][0]);
        }printf("\n\n");
        printf("err:");
        for(i = 0;i<100;i++){
            printf("%lf ",pred[i][0]-td.y[i][0]);
        }printf("\n");
    }
}