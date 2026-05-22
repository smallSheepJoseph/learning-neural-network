#include<bits/stdc++.h>
using namespace std;

// 使用 double 精度，如果想更快可以改 float
using dl = double;
using ll = long long;

// 固定亂數種子以重現結果
static std::mt19937 gen(1337);

double random_z() {
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}

// 目標函數：Sin 波
dl fixed_sin(dl x){
    return sin(x * 2 * 3.14159265358979323846);
}

// 數值穩定性修剪
#define adj(x) if(std::abs(x)>1e6){x*=0.99;}if(x>1e8){x=1e8;}if(x<-1e8){x=-1e8;}

// 【修改 1】全域學習率：調低至 0.0002，解決震盪問題
dl training_rate = 0.0002; 

// ==========================================
// 核心組件
// ==========================================

template<int InSize, int OutSize>
struct Layer {
    dl W[InSize][OutSize];
    dl B[OutSize];

    // Adam Optimizer 的狀態變數
    dl mB[OutSize]={};
    dl mW[InSize][OutSize]={};
    dl vB[OutSize]={};
    dl vW[InSize][OutSize]={};

    // 暫存梯度 (Accumulated Gradients)
    dl dlW[InSize][OutSize]={};
    dl dlB[OutSize]={};
    
    int trained_data = 0;

    Layer() {
        // 【修改 2】Xavier Initialization (稍微放大一點避免訊號太弱)
        dl scale = std::sqrt(6.0 / (InSize + OutSize)); 
        std::uniform_real_distribution<dl> dis(-scale, scale);
        
        for (int i = 0; i < InSize; ++i){
            for (int j = 0; j < OutSize; ++j){
                W[i][j] = dis(gen);
            }
        }
        for (int j = 0; j < OutSize; ++j){
            B[j] = 0;
        }
    }

    // Forward Pass
    inline void run(dl* input, dl* output){
        // 優化：手動展開或依靠編譯器 SIMD
        for(int j = 0; j < OutSize; j++){
            dl sum = B[j];
            for(int i = 0; i < InSize; i++){
                sum += input[i] * W[i][j];
            }
            output[j] = sum;
        }
    }

    // Backward: 計算對 Input 的梯度 (傳遞給上一層)
    inline void rev(dl* input, dl* output) {
        // input 是從下一層傳回來的梯度 (dL/dY)
        // output 是要傳給上一層的梯度 (dL/dX)
        for(int i = 0; i < InSize; i++){
            dl sum = 0;
            for(int j = 0; j < OutSize; j++){
                sum += input[j] * W[i][j];
            }
            output[i] = sum; // 這裡是 = 覆蓋，因為 Layer 是單向的
        }
    }

    // Backward: 計算對 W, B 的梯度並累積
    inline void train(dl* input_data, dl* gradient_from_next, dl* gradient_to_prev){
        // 1. 先算出要傳給上一層的梯度
        rev(gradient_from_next, gradient_to_prev);
        
        // 2. 累積權重梯度
        for(int j = 0; j < OutSize; j++){
            dl g = gradient_from_next[j];
            dlB[j] += g;
            for(int i = 0; i < InSize; i++){
                dlW[i][j] += input_data[i] * g;
            }
        }
        trained_data++;
    }

    // Update: Adam Optimizer
    inline void act_train(ll step, dl beta1t, dl beta2t){
        (void)step;
        if(trained_data == 0) return;

        dl rev_trained_data = 1.0 / trained_data;
        
        for(int j = 0; j < OutSize; j++){
            // 處理 Bias
            dlB[j] *= rev_trained_data;
            mB[j] = 0.9 * mB[j] + 0.1 * dlB[j];
            vB[j] = 0.999 * vB[j] + 0.001 * dlB[j] * dlB[j];
            B[j] -= training_rate * (mB[j] / (1.0 - beta1t)) / (std::sqrt(vB[j] / (1.0 - beta2t)) + 1e-8);
            adj(B[j]);
            dlB[j] = 0;

            // 處理 Weights
            for(int i = 0; i < InSize; i++){
                dlW[i][j] *= rev_trained_data;
                mW[i][j] = 0.9 * mW[i][j] + 0.1 * dlW[i][j];
                vW[i][j] = 0.999 * vW[i][j] + 0.001 * dlW[i][j] * dlW[i][j];
                
                // Adam Update
                W[i][j] -= training_rate * (mW[i][j] / (1.0 - beta1t)) / (std::sqrt(vW[i][j] / (1.0 - beta2t)) + 1e-8);
                
                // 【修改 3】暫時移除 Weight Decay，避免在訊號微弱時誤殺
                // W[i][j] *= (1.0 - training_rate * 1e-5); 
                
                adj(W[i][j]);
                dlW[i][j] = 0;
            }
        }
        trained_data = 0;
    }
};

// 【修改 4】激活函數：Softsign (比 Tanh 快且梯度更不容易消失)
template<int Size>
struct acf{
    // Softsign: x / (1 + |x|)
    inline dl fc(dl x){
        return x / (1.0 + std::abs(x));
    }

    // Softsign Derivative: 1 / (1 + |x|)^2
    // 為了加速，我們可以用 output (y) 來算： (1 - |y|)^2
    inline dl dfc_from_y(dl y){
        dl t = 1.0 - std::abs(y);
        return t * t;
    }

    inline void run(dl* input, dl* output){
        for(int i = 0; i < Size; i++){
            output[i] = fc(input[i]);
        }
    }

    // Backprop: gradient = next_grad * derivative
    inline void rev(dl* y, dl* input_grad, dl* output_grad) {
        for(int i = 0; i < Size; i++) {
            output_grad[i] = input_grad[i] * dfc_from_y(y[i]);
        }
    }
};

struct TrainConfig {
    bool enable_fix_gradient = true;
    dl clip_low = -1.0; // 【修改 5】禁用 Clip Low
    dl clip_high = 3.0; // 【修改】收緊到 3.0 防止爆炸
    ll step = 0;
};

// Gradient Clipper
struct gradient_fixer{
    TrainConfig& config;
    gradient_fixer(TrainConfig &tconfig) : config(tconfig) {}
    
    void fix(dl* gradient, int Size){
        if(!config.enable_fix_gradient) return;
        dl g2 = 0;
        for(int i = 0; i < Size; i++){
            g2 += gradient[i] * gradient[i];
        }
        dl g = std::sqrt(g2);
        
        // 只處理梯度爆炸
        if(g > config.clip_high){
            dl factor = config.clip_high / (g + 1e-9);
            for(int i = 0; i < Size; i++){
                gradient[i] *= factor;
            }
        }
    }
};

// ==========================================
// RNN 架構
// ==========================================

template<int InSize, int HiddenSize, int OutSize, int depth>
struct NN{
    Layer<InSize, HiddenSize> input_layer;
    Layer<HiddenSize, HiddenSize> hidden_layer[depth];
    Layer<HiddenSize, OutSize> output_layer;
    acf<HiddenSize> active_fc;

    // 儲存 Forward 的中間值 (給 Backward 用)
    dl V[depth+1][HiddenSize];
    dl Vinput[InSize];

    TrainConfig &config;
    gradient_fixer gfix;

    NN(TrainConfig &tconfig) : config(tconfig), gfix(tconfig) {}

    void run(dl* input, dl* output){
        run(input, output, V, Vinput);
    }

    void run(dl* input, dl* output, dl (*target_V)[HiddenSize], dl* target_Vinput){
        memcpy(target_Vinput, input, InSize * sizeof(dl));
        
        // Input -> Hidden
        input_layer.run(input, target_V[0]);
        active_fc.run(target_V[0], target_V[0]); 

        // Hidden -> Hidden
        for(int i = 0; i < depth; i++){
            hidden_layer[i].run(target_V[i], target_V[i+1]);
            active_fc.run(target_V[i+1], target_V[i+1]);
        }
        
        // Hidden -> Output
        output_layer.run(target_V[depth], output);
    }

    // BPTT 核心：直接訓練
    void train_directly(dl (*copied_V)[HiddenSize], dl* copied_Vinput, dl* dl_output, dl* dl_last = nullptr){
        dl gradient[HiddenSize];
        dl gradient2[HiddenSize];
        
        // Output Layer Backward
        output_layer.train(copied_V[depth], dl_output, gradient); // gradient 得到 dL/dHidden_last
        
        dl* now_gradient = gradient2;
        dl* last_gradient = gradient;

        // Hidden Layers Backward
        for(int i = depth - 1; i >= 0; i--){
            gfix.fix(last_gradient, HiddenSize);
            
            // 通過激活函數的導數
            active_fc.rev(copied_V[i+1], last_gradient, last_gradient);
            
            // 通過 Linear Layer
            hidden_layer[i].train(copied_V[i], last_gradient, now_gradient);
            swap(last_gradient, now_gradient);
        }

        // Input Layer Backward
        gfix.fix(last_gradient, HiddenSize);
        active_fc.rev(copied_V[0], last_gradient, last_gradient);
        
        // 最後算出對 Input 的梯度，存入 now_gradient
        input_layer.train(copied_Vinput, last_gradient, now_gradient);
        
        // 如果外部需要這個梯度 (傳給上一個時間步)，就複製出去
        if(dl_last != nullptr){
            memcpy(dl_last, now_gradient, sizeof(dl) * InSize);
        }
    }

    void active_train(dl beta1t, dl beta2t, ll step){
        input_layer.act_train(step, beta1t, beta2t);
        for(int i = 0; i < depth; i++){
            hidden_layer[i].act_train(step, beta1t, beta2t);
        }
        output_layer.act_train(step, beta1t, beta2t);
    }
};

template<int InSize, int HiddenSize, int OutSize, int MemorySize, int DataLen, int depth>
struct RNN{
    NN<InSize + MemorySize, HiddenSize, OutSize + MemorySize, depth> nn;
    
    // 緩衝區大小
    static constexpr int BUF = (InSize + MemorySize > OutSize + MemorySize) 
                             ? (InSize + MemorySize) : (OutSize + MemorySize);
    
    // 儲存每個時間步的狀態
    dl V[DataLen][depth+1][HiddenSize];
    dl Vinput[DataLen][BUF];

    // 初始記憶 (Hidden State at t=0)
    dl h0[MemorySize];
    dl dlh0[MemorySize]; // h0 的梯度
    dl mh0[MemorySize]={};
    dl vh0[MemorySize]={};
    
    TrainConfig &config;
    gradient_fixer gfix;

    RNN(TrainConfig &tconfig) : nn(tconfig), config(tconfig), gfix(tconfig){
        memset(h0, 0, sizeof(dl)*MemorySize);
        memset(dlh0, 0, sizeof(dl)*MemorySize);
    }

    // Forward Propagation Through Time
    void run(dl (*input)[InSize], dl (*output)[OutSize], int len){
        dl tmp[BUF], tmp2[BUF];
        // 載入初始記憶
        memcpy(tmp, h0, sizeof(dl)*MemorySize);
        
        dl *last_data = tmp; 
        dl *now_data = tmp2;
        
        for(int i = 0; i < len; i++){
            // 組合 Input: [Memory, Data]
            memcpy(last_data + MemorySize, input[i], sizeof(dl)*InSize);
            
            // Run NN
            nn.run(last_data, now_data, V[i], Vinput[i]);
            
            // 輸出結果: [New_Memory, Output] -> 取 Output
            memcpy(output[i], now_data + MemorySize, sizeof(dl)*OutSize);
            
            // 下一步的 Memory 就是這一大塊的前半部
            // 這裡直接交換指標，now_data 變成了下一步的 last_data
            swap(last_data, now_data);
        }
    }

    // 支援 Self-Prediction 的 Forward
    void run_self_pred(dl (*input)[InSize], dl (*output)[OutSize], int len, int given_data){
        dl tmp[BUF], tmp2[BUF];
        memcpy(tmp, h0, sizeof(dl)*MemorySize);
        dl *last_data = tmp, *now_data = tmp2;

        for(int i = 0; i < len; i++){
            if(i < given_data){
                // Teacher Forcing: 使用真實數據
                memcpy(last_data + MemorySize, input[i], sizeof(dl)*InSize);
            } else {
                // Self Prediction: 使用上一步的輸出 (注意：必須 InSize == OutSize)
                // 上一步的 Output 存在 last_data 的 [MemorySize...MemorySize+OutSize] 位置嗎?
                // 不，last_data 是 "上一步的輸入"。
                // 我們需要的 "上一步的輸出" 是在上一次迴圈結束時寫入 memory 的。
                // 在 swap 之前，now_data 存著當前輸出。
                // swap 之後，last_data 存著上一步的輸出 (也就是當前的 Memory + Output)。
                // 所以我們不需要 memcpy，因為 output 已經在 buffer 裡了。
                // *注意*：這假設 Output 的位置和 Input 的位置是對齊的，且 InSize == OutSize
            }
            
            nn.run(last_data, now_data, V[i], Vinput[i]);
            memcpy(output[i], now_data + MemorySize, sizeof(dl)*OutSize);
            swap(last_data, now_data);
        }
    }

    // Backpropagation Through Time (BPTT)
    dl train(dl (*input)[InSize], dl (*real_value)[OutSize], int len, int train_stoppoint, int self_pred_point = -1){
        dl pred[DataLen][OutSize];
        
        // 1. Forward Pass
        if(self_pred_point != -1){
            run_self_pred(input, pred, len, self_pred_point);
        } else {
            run(input, pred, len);
        }

        dl loss = 0;
        dl loss_cnt = 0;
        
        // 梯度緩衝區
        dl gradient[BUF], gradient2[BUF];
        memset(gradient, 0, sizeof(dl)*BUF);
        memset(gradient2, 0, sizeof(dl)*BUF);
        
        dl *last_gradient = gradient; // 這是累積 "來自未來" 的梯度
        dl *now_gradient = gradient2; // 這是算出 "傳給過去" 的梯度

        // 2. Backward Pass (從最後一步往回走)
        for(int i = len - 1; i >= 0; i--){
            
            // 計算當前時間步的 Loss Gradient
            if(i >= train_stoppoint){
                for(int j = 0; j < OutSize; j++){
                    dl diff = pred[i][j] - real_value[i][j];
                    dl grad = 2.0 * diff;

                    // 【修改 6】時間加權 Loss: 越後面權重越重，強迫學會長期依賴
                    dl time_weight = 1.0 + (3.0 * (dl)i / len); 
                    grad *= time_weight;

                    // 【修改 7】正確的梯度累積邏輯
                    // last_gradient[MemorySize + j] 存放的是 "下一時刻對 Input(也就是這一時刻的Output) 的梯度"
                    // 我們要加上 "這一時刻 Loss 對 Output 的梯度"
                    
                    // 只有在 Self-Prediction 模式下，這一步的輸出才會被下一步當作 Input
                    // 邊界檢查：self_pred_point 之後或者是交界處
                    if (i >= self_pred_point - 1) {
                         last_gradient[MemorySize + j] += grad;
                    } else {
                         // Teacher Forcing 模式：下一步用的是 Ground Truth，跟我的輸出無關
                         // 所以這一步的梯度純粹來自當下的 Loss
                         last_gradient[MemorySize + j] = grad; 
                    }

                    loss += diff * diff;
                    loss_cnt++;
                }
                gfix.fix(last_gradient, MemorySize + OutSize);
            } else {
                // 如果這一步不訓練，就把 Data 部分的梯度清零 (Hidden State 梯度保留)
                memset(last_gradient + MemorySize, 0, sizeof(dl) * OutSize);
                gfix.fix(last_gradient, MemorySize);
            }

            // 傳遞梯度到神經網路，並算出傳給上一步的梯度 (now_gradient)
            nn.train_directly(V[i], Vinput[i], last_gradient, now_gradient);
            swap(last_gradient, now_gradient);
        }

        // 累積對初始狀態 h0 的梯度
        for(int i = 0; i < MemorySize; i++){
            dlh0[i] += last_gradient[i];
        }
        
        return loss / (loss_cnt + 1e-9);
    }

    dl beta1t = 1;
    dl beta2t = 1;
    ll step = 0;

    void active_train(){
        step++;
        beta1t *= 0.9;
        beta2t *= 0.999;
        
        nn.active_train(beta1t, beta2t, step);

        // 更新 h0 (Learnable Initial State)
        for(int i = 0; i < MemorySize; i++){
            mh0[i] = 0.9 * mh0[i] + 0.1 * dlh0[i];
            vh0[i] = 0.999 * vh0[i] + 0.001 * dlh0[i] * dlh0[i];
            h0[i] -= training_rate * (mh0[i] / (1.0 - beta1t)) / (std::sqrt(vh0[i] / (1.0 - beta2t)) + 1e-8);
        }
        memset(dlh0, 0, sizeof(dl) * MemorySize);
    }
};

// 資料結構
template<int InSize, int OutSize, int Len>
struct RNN_data{
    dl x[Len][InSize]={};
    dl y[Len][OutSize]={};
};

using Data = RNN_data<1, 1, 50>; // 【修改 8】Seq Len 設為 50

Data gen_data(dl a, dl b, dl rd=0.0){
    Data d;
    static std::normal_distribution<dl> distribution(0.0, rd);
    for(int i = 0; i < 50; i++){
        d.x[i][0] = fixed_sin(a + b * i);
        if(rd > 0) d.x[i][0] += distribution(gen);
        d.y[i][0] = fixed_sin(a + b * (i + 1));
    }
    return d;
}

int main(){
    printf("Initializing Optimized RNN...\n");
    
    TrainConfig config;
    // 【配置重點】禁用 Low Clip，保留 High Clip
    config.clip_high = 3.0; // 收緊
    config.clip_low = -1.0; 

    // RNN<Input, Hidden, Output, Memory, DataLen, Layers>
    // Hidden 設為 64 (夠寬), Layers 設為 1 (夠淺)
    auto rnn_ptr = make_unique<RNN<1, 64, 1, 63, 50, 1>>(config);
    auto& rnn = *rnn_ptr;
    
    constexpr int n = 50000;
    auto train_data_ptr = make_unique<array<Data, n>>();
    array<Data, n>* train_ptr = train_data_ptr.get();
    
    printf("Generating data...\n");
    for(int i = 0; i < n; i++){
        dl a = random_z();
        dl b = min(random_z(), random_z()) * 0.4 + 0.05;
        (*train_data_ptr)[i] = gen_data(a, b);
    }

    printf("Start training (LR = %f, SeqLen = 50)...\n", training_rate);
    
    int cnt = 0;
    int self_pred = 0;
    int cool_down = 0;
    dl curr_loss = 100.0;
    
    for(int t = 0; true; t++){
        // 【修改 9】極簡化的 Curriculum Learning
        // 只有當 Loss 夠低時，才增加 self_pred 長度
        
        if(t > 20 && cool_down == 0) {
            if(curr_loss < 0.005) { // 門檻放寬一點點
                self_pred = min(self_pred + 1, 45); // 最多 45 步自回歸 (留 5 步給老師)
                cool_down = 10; // 冷卻久一點，讓模型適應新難度
                printf(">>> UPGRADE! Level %d <<<\n", self_pred);
            } else if(curr_loss > 0.05) {
                self_pred = max(self_pred - 1, 0);
                cool_down = 5;
                printf("<<< DOWNGRADE. Level %d <<<\n", self_pred);
            }
        }
        if(cool_down > 0) cool_down--;

        // 每個 epoch 隨機抽樣訓練
        dl total_epoch_loss = 0;
        int steps_per_epoch = 1000;
        
        for(int i = 0; i < steps_per_epoch; i++){
            int id = (int)(random_z() * (n - 1));
            auto& td = (*train_ptr)[id];
            
            // 訓練長度: 50, 前 5 步強制 Teacher Forcing, 後面看 self_pred
            dl batch_loss = rnn.train(td.x, td.y, 50, 5, 50 - self_pred);
            total_epoch_loss += batch_loss;
            
            cnt++;
            if((cnt & 15) == 0) rnn.active_train(); // 每 16 筆資料更新一次權重 (Mini-batch 概念)
        }

        // 評估階段
        dl eval_loss = 0;
        dl eval_curr_loss = 0;
        dl eval_self_pred_loss = 0;
        int eval_count = 100;

        for(int i = 0; i < eval_count; i++){
            int id = (int)(random_z() * (n - 1));
            auto& td = (*train_ptr)[id];
            dl pred[50][1];

            // 1. 全自回歸測試 (最嚴格) - 計算單筆樣本的 Loss
            dl sample_full_loss = 0;
            rnn.run_self_pred(td.x, pred, 50, 5); 
            
            for(int k = 5; k < 50; k++){
                dl err = pred[k][0] - td.y[k][0];
                sample_full_loss += err * err;
            }
            eval_self_pred_loss += sample_full_loss;

            // 2. 當前難度測試 - 計算單筆樣本的 Loss
            dl sample_curr_loss = 0;
            rnn.run_self_pred(td.x, pred, 50, 50 - self_pred);
            int count_k = 0;
            for(int k = 50 - self_pred; k < 50; k++){
                dl err = pred[k][0] - td.y[k][0];
                sample_curr_loss += err * err;
                count_k++;
            }
            
            if(count_k == 0) {
                // 如果是 Level 0，用全自回歸的 Loss 當作參考，避免 0 導致錯誤升級
                // 這裡要加的是單筆的 loss，不是累計的
                eval_curr_loss += sample_full_loss; 
            } else {
                eval_curr_loss += sample_curr_loss;
            }
        }
        
        // 計算 MSE
        eval_self_pred_loss /= (45.0 * eval_count);
        
        if (self_pred == 0) {
            // 如果是 Level 0，我們上面加的是 sample_full_loss (45步)
            // 所以分母要是 45 * count
            eval_curr_loss = total_epoch_loss/steps_per_epoch;
        } else {
            eval_curr_loss /= (max(1, self_pred) * eval_count);
        }
        
        curr_loss = eval_curr_loss; // 更新用於判斷升級的 loss

        printf("Ep: %d, Lvl: %d, AvgLoss: %.5f, CurrLvlLoss: %.5f, FullAutoLoss: %.5f\n", 
               t, self_pred, total_epoch_loss/steps_per_epoch, curr_loss, eval_self_pred_loss);

        // 如果 Full Auto Loss 已經夠低，就提早結束
        if(self_pred >= 40 && eval_self_pred_loss < 0.001) {
            printf("\nCONVERGED! Training Complete.\n");
            break;
        }
    }

    // 測試模式
    while(1){
        dl a, b;
        printf("\nInput phase(0-1) and freq(0.05-0.45): ");
        if(scanf("%lf %lf", &a, &b) == EOF) break;
        
        printf("Start pred pos (e.g., 5): ");
        int dd;
        scanf("%d", &dd);
        
        Data td = gen_data(a, b);
        dl pred[50][1];
        rnn.run_self_pred(td.x, pred, 50, dd);
        
        printf("\n[Real] vs [Pred] (Diff)\n");
        for(int i = 0; i < 50; i++){
            if(i == dd) printf("--- Self Prediction Start ---\n");
            printf("[%2d] %6.3f  %6.3f  (%6.3f)\n", 
                   i, td.y[i][0], pred[i][0], pred[i][0] - td.y[i][0]);
        }
    }
}