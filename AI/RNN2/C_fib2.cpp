#include<bits/stdc++.h>
using namespace std;
using dl = double;
// using dl = float;
using ll = long long;
#define ff first
#define ss second


static std::mt19937 gen(1337);
double random_z() {
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}

dl fixed_sin(dl x){
    return sin(x*2*3.14159265358979323846);
}

#define adj(x) if(abs(x)>1e6){x*=0.99;}if(x>1e8){x=1e8;}if(x<-1e8){x=-1e8;}



struct ParamBlock {
    dl* w;
    dl* g;
    int size;
    int counter;
};

struct memory_allocator{
    dl *w_block, *g_block;
    int begin,end;
    int ptr;
    ParamBlock *PB_arr;
    int PB_top = 0,PB_size;
    memory_allocator(int size,int _PB_size){
        w_block = (dl*)_aligned_malloc(size * sizeof(dl),32);
        g_block = (dl*)_aligned_malloc(size * sizeof(dl),32);
        memset(w_block, 0, size * sizeof(dl));
        memset(g_block, 0, size * sizeof(dl));
        PB_arr = (ParamBlock*)malloc(_PB_size * sizeof(ParamBlock));
        PB_size = _PB_size;
        PB_top = 0;
        begin = 0;
        ptr = 0;
        end = size;
    }
    ParamBlock& alloc(int size){
        size = (size + 3) & ~3;
        PB_arr[PB_top++]=
        (ParamBlock){
            w_block + ptr,
            g_block + ptr,
            size,
            0
        };
        ptr+=size;
        assert(PB_top<=PB_size);
        assert(ptr<=end);
        return PB_arr[PB_top-1];
    }
    int remaining(){
        return end - ptr;
    }
    int used(){
        return ptr - begin;
    }
    void norm(){
        for(int i = 0;i<PB_top;i++){
            if (PB_arr[i].counter <= 1){
                PB_arr[i].counter = 0;
                continue;
            }
            dl rev = 1.0/PB_arr[i].counter;
            #pragma omp simd
            for(int j = 0;j<PB_arr[i].size;j++){
                PB_arr[i].g[j]*=rev;
            }
            PB_arr[i].counter = 0;
        }
    }
    ~memory_allocator(){
        _aligned_free(w_block);
        _aligned_free(g_block);
        free(PB_arr);
    }
};

namespace fast_hash {
    //this part written by ai
constexpr uint64_t mixer(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

constexpr uint64_t combine(uint64_t seed, uint64_t v) {
    seed ^= mixer(v) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

template <typename... Args>
constexpr uint64_t compute(Args... args) {
    uint64_t seed = 0;
    ((seed = combine(seed, static_cast<uint64_t>(args))), ...);
    return seed;
}

}//namespace fast_hash

namespace active_function {
// also written by ai cuz i don't know what is CRTP
// 使用 CRTP 技巧：把子類別的型別當作模板參數傳給 Base

//TODO: add norm(average = 0, std = 1) and softmax

template <typename Derived>
struct Base {
    template <int Size>
    inline void run(dl* input) { //inplace
        Derived* derived = static_cast<Derived*>(this);
        #pragma omp simd
        for(int i = 0; i < Size; i++){
            input[i] = derived->fc(input[i]);
        }
    }
    template <int Size>
    inline void rev(dl* y, dl* grad_in) { //inplace
        Derived* derived = static_cast<Derived*>(this);
        #pragma omp simd
        for(int i = 0; i < Size; i++) {
            grad_in[i] = grad_in[i] * derived->dfc(y[i]);
        }
    }
    template <int Size>
    inline void run(dl* input, dl* output) {
        Derived* derived = static_cast<Derived*>(this);
        #pragma omp simd
        for(int i = 0; i < Size; i++){
            output[i] = derived->fc(input[i]);
        }
    }
    template <int Size>
    inline void rev(dl* y, dl* grad_in, dl* grad_out) {
        Derived* derived = static_cast<Derived*>(this);
        #pragma omp simd
        for(int i = 0; i < Size; i++) {
            grad_out[i] = grad_in[i] * derived->dfc(y[i]);
        }
    }
};

struct None : public Base<None> {
    template <int Size>
    inline void run(dl* input) {
        (void)input;
        return;// do nothing
    }
    template <int Size>
    inline void rev(dl* y, dl* grad_in) {
        (void)y;(void)grad_in;
        return;//do nothing
    }
    template <int Size>
    inline void run(dl* input, dl* output) {
        if(input != output)memcpy(output, input, Size * sizeof(dl));
    }
    template <int Size>
    inline void rev(dl* y, dl* grad_in, dl* grad_out) {
        if(grad_in != grad_out)memcpy(grad_out, grad_in, Size * sizeof(dl));
    }
};

struct Relu : public Base<Relu> {
    inline dl fc(dl x) { return x > 0 ? x : 0; }
    inline dl dfc(dl y) { return y > 0 ? 1.0 : 0.0; }
};

struct LRelu : public Base<LRelu> {
    dl alpha = 0.02;
    LRelu() {}
    LRelu(dl alpha) : alpha(alpha) {}
    inline dl fc(dl x) { return x > 0 ? x : alpha * x; }
    inline dl dfc(dl y) { return y > 0 ? 1.0 : alpha; }
};

struct Tanh : public Base<Tanh> {
    inline dl fc(dl x) { return std::tanh(x); }
    inline dl dfc(dl y) { return 1.0 - y * y; }
};

struct Softsign : public Base<Softsign> {
    inline dl fc(dl x) { return x / (std::abs(x) + 1.0); }
    inline dl dfc(dl y) { 
        dl tmp = 1.0 - std::abs(y);
        return tmp * tmp; 
    }
};

struct Softmax{
    template <int Size>
    inline void run(dl* input,dl* output){
        if(Size == 0)return;
        dl m=-1e100;
        for(int i = 0;i<Size;i++){
            m = max(m,input[i]);
        }
        dl di=1e-10;
        for(int i = 0;i<Size;i++){
            di+=exp(input[i]-m);
        }
        di = 1/di;
        for(int i = 0;i<Size;i++){
            output[i]=exp(input[i]-m)*di;
        }
    }
    template <int Size>
    inline void rev_loss(dl* y, dl* real_value, dl* grad_out){
        for(int i = 0;i<Size;i++){
            grad_out[i] = y[i] - real_value[i];
        }
    }
    template <int Size>
    inline void rev(dl* y, dl* grad_in, dl* grad_out) {
        dl dot = 0.0;
        for(int i = 0; i < Size; i++) {
            dot += grad_in[i] * y[i];
        }
        for(int i = 0; i < Size; i++) {
            grad_out[i] = y[i] * (grad_in[i] - dot);
        }
    }
};

} // namespace active_function

namespace init {
    struct Xavier {
        static inline dl get_std(int in, int out) { return std::sqrt(2.0 / (in + out)); }
    };
    struct He {
        static inline dl get_std(int in, int out) {(void)out; return std::sqrt(2.0 / (in*1.0004)); } //warning for unused out
    };
}

//dynamic config
//this small thing is important
//please check which struct use it befor edit it
struct TrainConfig {
    bool enable_fix_gradient = true;
    bool weight_decay = false;
    dl clip_low = -1.0, clip_high = 10.0;
    dl lr = 0.002;
};

//gradient fixer
template<int Size>
struct GradientFixer{
    dl* gradient;
    TrainConfig& config;
    GradientFixer(dl* grad,TrainConfig &tconfig) : gradient(grad),config(tconfig){
        //pass
    }
    void fix(){
        dl* __restrict__ _grad = (dl*)__builtin_assume_aligned(gradient, 32);
        if(!config.enable_fix_gradient)return;
        dl g2 = 0;
        if constexpr(Size > 10000){
            #pragma omp parallel for simd reduction(+:g2)
            for(int i = 0;i<Size;i++){
                g2 += _grad[i]*_grad[i];
            }
        }else{
            #pragma omp simd reduction(+:g2)
            for(int i = 0;i<Size;i++){
                g2 += _grad[i]*_grad[i];
            }
        }
        
        dl g = sqrt(g2);
        if(g < config.clip_low || g > config.clip_high){
            dl target = min((dl)config.clip_high, max((dl)config.clip_low, g));
            dl factor = target / (g+1e-6);
            #pragma omp parallel for simd if(Size > 10000)
            for(int i = 0;i<Size;i++){
                _grad[i] *= factor;
            }
        }
    }
};

// globle trainer
template<int Size>
struct Trainer{
    dl *m,*v;
    dl *gradient, *value;
    struct TrainData{
        ll step_counter = 0;
        const dl beta1,beta2;
        dl beta1t,beta2t;
        TrainConfig &config;
        TrainData(TrainConfig &tconfig, dl b1 = 0.9, dl b2 = 0.999)
            : beta1(b1),beta2(b2),config(tconfig){
            beta1t = b1;
            beta2t = b2;
        }

        void step(){
            step_counter++;
            beta1t *= beta1;
            beta2t *= beta2;
        }
    }td;
    Trainer(dl *g, dl *val,TrainConfig& tfg):gradient(g),value(val),td(tfg){
        m = (dl*)_aligned_malloc(Size * sizeof(dl),32);
        v = (dl*)_aligned_malloc(Size * sizeof(dl),32);
        memset(m, 0, Size * sizeof(dl));
        memset(v, 0, Size * sizeof(dl));
    }

    ~Trainer(){
        _aligned_free(m);_aligned_free(v);
    }

    void train(){
        dl* __restrict__ _val = (dl*)__builtin_assume_aligned(value, 32);
        dl* __restrict__ _m   = (dl*)__builtin_assume_aligned(m, 32);
        dl* __restrict__ _v   = (dl*)__builtin_assume_aligned(v, 32);
        dl* __restrict__ _grad = (dl*)__builtin_assume_aligned(gradient, 32);
        dl rev_beta1t = 1.0/(1.0-td.beta1t);
        dl rev_beta2t = 1.0/(1.0-td.beta2t);
        dl beta1 = td.beta1;
        dl beta2 = td.beta2;
        dl lr = td.config.lr;
        if(td.config.weight_decay){
            #pragma omp parallel for simd if(Size > 5000)
            for(int i = 0;i<Size;i++){
                _m[i] = beta1*_m[i] + (1.0-beta1)*_grad[i];
                _v[i] = beta2*_v[i] + (1.0-beta2)*_grad[i]*_grad[i];
                _val[i] -= lr * (_m[i]*rev_beta1t)/(sqrt(_v[i]*rev_beta2t) + 1e-8);
                _val[i] *= (1-lr*1e-6);
                _grad[i] = 0;
            }
        }else{
            #pragma omp parallel for simd if(Size > 5000)
            for(int i = 0;i<Size;i++){
                _m[i] = beta1*_m[i] + (1.0-beta1)*_grad[i];
                _v[i] = beta2*_v[i] + (1.0-beta2)*_grad[i]*_grad[i];
                _val[i] -= lr * (_m[i]*rev_beta1t)/(sqrt(_v[i]*rev_beta2t) + 1e-8);
                _grad[i] = 0;
            }
        }
        
        td.step();
    }
};

template<int InSize, int OutSize, typename Act = active_function::LRelu, typename Init = init::Xavier>
struct Layer {
    using dl_arr = dl[OutSize];
    
    dl_arr* W; //[InSize][OutSize]
    dl* B;
    dl_arr* dlW;
    dl* dlB;
    static constexpr int W_size = (InSize*OutSize + 3) & ~3;
    static constexpr int B_size = (OutSize + 3) & ~3;
    static constexpr int memory_required = W_size + B_size;

    Act act;

    ParamBlock& pbW,pbB;
    Layer(memory_allocator& mal): 
        pbW(mal.alloc(W_size)),
        pbB(mal.alloc(B_size)){
        W = (dl_arr*)pbW.w;
        dlW = (dl_arr*)pbW.g;
        B = pbB.w;
        dlB = pbB.g;
        std::normal_distribution<dl> dis(0.0, Init::get_std(InSize,OutSize));
        for (int i = 0; i < InSize; ++i){
            for (int j = 0; j < OutSize; ++j){
                W[i][j] = dis(gen);
                dlW[i][j] = 0;
            }
        }
        for (int j = 0; j < OutSize; ++j){
            B[j] = 0;
            dlB[j] = 0;
        }
    }

    static constexpr int param_count() {return InSize * OutSize + OutSize;}

    //run this layer
    inline void run(dl*__restrict__ input,dl*__restrict__ output){
        memset(output,0,sizeof(dl) * OutSize);
        // #pragma omp parallel for if(InSize*OutSize > 10000)
        // output[j] not based i
        for(int i = 0;i<InSize;i++){
            #pragma omp simd
            for(int j = 0;j<OutSize;j++){
                output[j] += input[i]*W[i][j];
            }
        }
        for(int j = 0;j<OutSize;j++){
            output[j] += B[j];
        }
        act.template run<OutSize>(output);
    }

    //only calculate gradient
    inline void rev(dl*__restrict__ y,dl*__restrict__ grad_in, dl*__restrict__ grad_out){
        act.template rev<OutSize>(y,grad_in);
        memset(grad_out,0,sizeof(dl)*InSize); // gradiend from input, so using InSize instead of OutSize
        #pragma omp parallel for if(InSize*OutSize > 10000)
        for(int i = 0;i<InSize;i++){
            for(int j = 0;j<OutSize;j++){
                grad_out[i] += grad_in[j]*W[i][j];
            }
        }
    }
    // x: last layer output
    // y: this layer output (for activate)
    // grad_in: the gradient travle from the next layer/step
    // grad_out: the gradient to last layer
    inline void train(dl* __restrict__ x, dl* __restrict__ y, dl* __restrict__ grad_in, dl* __restrict__ grad_out){
        act.template rev<OutSize>(y,grad_in);
        memset(grad_out,0,sizeof(dl)*InSize); // same as above
        #pragma omp parallel for if(InSize*OutSize > 10000 && InSize > 64)
        for(int i = 0;i<InSize;i++){
            #pragma omp simd
            for(int j = 0;j<OutSize;j++){
                grad_out[i]+=grad_in[j]*W[i][j];
                dlW[i][j]+=x[i]*grad_in[j];
            }
        }
        #pragma omp simd
        for(int j = 0;j<OutSize;j++){
            dlB[j]+=grad_in[j];
        }
        pbW.counter++;
        pbB.counter++;
    }


    void save(ofstream& f){
        char header_buf[256]={};
        memset(header_buf,' ',sizeof(header_buf));
        snprintf(header_buf, sizeof(header_buf), 
            "\nlayer: InSize: %d, OutSize: %d\n",InSize,OutSize);
        f.write(header_buf, sizeof(header_buf));
        f.write((char*)W, InSize * OutSize * sizeof(dl));
        f.write((char*)B, OutSize * sizeof(dl));
    }

    void load(ifstream& f){
        char header_buf[256]={};
        printf("loading a layer\n");
        f.read(header_buf, sizeof(header_buf));
        int load_in,load_out;
        if (sscanf(header_buf, "\nlayer: InSize: %d, OutSize: %d\n", &load_in, &load_out) != 2) {
            printf("Error: Layer structure in header is corrupted or format mismatch.\n");
            assert(false);
        }
        if(load_in != InSize || load_out != OutSize){
            printf("size doesn't match\n");
            printf("expected: %d=>%d  loaded: %d=>%d\n",InSize,OutSize,load_in,load_out);
            assert(false);
        }
        f.read((char*)W, InSize * OutSize * sizeof(dl));
        f.read((char*)B, OutSize * sizeof(dl));
    }
    
};

template<int InSize,int HiddenSize,int OutSize,int depth>
struct NN_record{
    dl *Vinput; //[InSize]
    dl (*V)[HiddenSize]; //[depth+1][HiddenSize]
    dl *Voutput; //[OutSize]
    NN_record(){
        Vinput = (dl*)_aligned_malloc(InSize * sizeof(dl),32);
        V = (dl(*)[HiddenSize])_aligned_malloc( (depth+1) * HiddenSize * sizeof(dl), 32 );
        Voutput = (dl*)_aligned_malloc(OutSize * sizeof(dl),32);
        memset(Vinput, 0, InSize * sizeof(dl));
        memset(V, 0, (depth+1) * HiddenSize * sizeof(dl));
        memset(Voutput, 0, OutSize * sizeof(dl));
    }
    ~NN_record(){
        _aligned_free(Vinput);
        _aligned_free(V);
        _aligned_free(Voutput);
    }
    NN_record(NN_record&& other) noexcept {
        Vinput = other.Vinput;  other.Vinput = nullptr;
        V = other.V;            other.V = nullptr;
        Voutput = other.Voutput; other.Voutput = nullptr;
    }
    NN_record(const NN_record&) = delete;
    NN_record& operator=(const NN_record&) = delete;
};

//depth is hidder layer depth
template<int InSize,int HiddenSize,int OutSize,int depth, typename Act = active_function::LRelu, typename Init = init::Xavier>
struct NN{
    static constexpr int input_len = InSize, hidden_len = HiddenSize, output_size = OutSize, _depth = depth;
    using InLayer  = Layer<InSize, HiddenSize, Act, Init>;
    using HidLayer = Layer<HiddenSize, HiddenSize, Act, Init>;
    using OutLayer = Layer<HiddenSize, OutSize, active_function::None, Init>;
    using Record = NN_record<InSize, HiddenSize,OutSize,depth>;

    InLayer input_layer;
    HidLayer* hidden_layer; //[depth]
    OutLayer output_layer;
    Record rd;

    static constexpr int memory_required = 
        InLayer::memory_required + 
        OutLayer::memory_required + 
        (depth > 0 ? depth * HidLayer::memory_required : 0);
    

    NN(memory_allocator& mal):
    input_layer(mal),output_layer(mal){
        if constexpr (depth > 0){
            hidden_layer = (HidLayer*)_aligned_malloc(depth * sizeof(HidLayer),64);
        }else{
            hidden_layer = nullptr;
        }

        for (int i = 0; i < depth; i++)
            new (&hidden_layer[i]) HidLayer(mal);
    }
    ~NN(){
        for (int i = 0; i < depth; i++){
            hidden_layer[i].~HidLayer();
        }
        if constexpr (depth>0)_aligned_free(hidden_layer);
    }


    void run(dl* input,dl* output){
        run(rd,input,output);
    }

    void run(Record& target_rd,dl*__restrict__ input,dl*__restrict__ output){
        assert(input!=output);
        dl* __restrict__ target_Vinput = target_rd.Vinput;
        dl (* __restrict__ target_V)[HiddenSize] = target_rd.V;
        dl* __restrict__ target_Voutput = target_rd.Voutput;
        memcpy(target_Vinput,input,InSize * sizeof(dl));
        input_layer.run(input,target_V[0]);
        for(int i = 0;i<depth;i++){
            hidden_layer[i].run(target_V[i],target_V[i+1]);
        }
        output_layer.run(target_V[depth],target_Voutput);
        memcpy(output,target_Voutput,OutSize * sizeof(dl));
    }

    //move data that can be use to train later
    //only call this when data is about to overwrite
    Record copy_record(){
        return std::exchange(rd, Record{});
    }

    //only calculate the gradient
    //grad_in the gradiend from next step(or another NN), and output the gradiend of last step
    //this function has error mutiple times, take care of it
    void rev(Record &copied_rd, dl*__restrict__ grad_in, dl*__restrict__ grad_out){
        assert(grad_in!=grad_out);
        dl* __restrict__ copied_Vinput = (dl*)__builtin_assume_aligned(copied_rd.Vinput, 32);
        dl (* __restrict__ copied_V)[HiddenSize] = (dl(*)[HiddenSize])__builtin_assume_aligned(copied_rd.V, 32);
        dl* __restrict__ copied_Voutput = (dl*)__builtin_assume_aligned(copied_rd.Voutput, 32);
        dl gradient[HiddenSize],gradient2[HiddenSize];
        output_layer.rev(copied_Voutput,grad_in,gradient);
        dl* now_gradient = gradient2,*last_gradient = gradient;
        for(int i = depth-1;i>=0;i--){
            hidden_layer[i].rev(copied_V[i+1],last_gradient,now_gradient);
            swap(last_gradient,now_gradient);
        }
        input_layer.rev(copied_V[0],last_gradient,grad_out);
        //memcpy(output, now_gradient ,sizeof(dl)*InSize);
    }


    
    //train this NN directly using copied data and gradient
    void train_directly(Record& copied_rd, dl*__restrict__ grad_in, dl*__restrict__ grad_out = nullptr){

        dl* __restrict__ copied_Vinput = (dl*)__builtin_assume_aligned(copied_rd.Vinput, 32);
        dl (* __restrict__ copied_V)[HiddenSize] = copied_rd.V;
        dl* __restrict__ copied_Voutput = (dl*)__builtin_assume_aligned(copied_rd.Voutput, 32);

        dl gradient[HiddenSize],gradient2[HiddenSize];
        output_layer.train(copied_V[depth],copied_Voutput,grad_in,gradient);
        dl*__restrict__ now_gradient = gradient2;dl*__restrict__ last_gradient = gradient;
        for(int i = depth-1;i>=0;i--){
            hidden_layer[i].train(copied_V[i],copied_V[i+1],last_gradient,now_gradient);
            swap(last_gradient,now_gradient);
        }
        input_layer.train(copied_Vinput,copied_V[0],last_gradient,now_gradient);
        //copy to output gradient
        if(grad_out != nullptr){
            memcpy(grad_out, now_gradient, sizeof(dl) * InSize);
        }
    }

    //train this NN using real data
    dl train(dl* input, dl* real_value, dl* grad_out = nullptr){
        dl pred[OutSize];               
        run(input, pred);
        dl grad_in[OutSize];
        dl loss = 0.0;
        for (int i = 0; i < OutSize; ++i){
            grad_in[i] = 2.0 * (pred[i] - real_value[i]);
            loss += (pred[i] - real_value[i])*(pred[i] - real_value[i]);
        }
        train_directly(rd, grad_in, grad_out);
        return loss;
    }

    void save(ofstream& f){
        char header_buf[256]={};
        memset(header_buf,' ',sizeof(header_buf));
        snprintf(header_buf, sizeof(header_buf), 
            "\nNN: InSize: %d, HiddenSize: %d, OutSize: %d, depth:%d \n",InSize,HiddenSize,OutSize,depth);
        f.write(header_buf, sizeof(header_buf));
        input_layer.save(f);
        for(int i = 0;i<depth;i++){
            hidden_layer[i].save(f);
        }
        output_layer.save(f);
    }

    void load(ifstream& f){
        char header_buf[256]={};
        printf("loading a layer\n");
        f.read(header_buf, sizeof(header_buf));
        int l1,l2,l3,l4;
        if (sscanf(header_buf, "\nNN: InSize: %d, HiddenSize: %d, OutSize: %d, depth:%d \n",&l1,&l2,&l3,&l4) != 4) {
            printf("Error: Layer structure in header is corrupted or format mismatch.\n");
            assert(false);
        }
        if(InSize!=l1 ||
            HiddenSize!=l2 ||
            OutSize!=l3 ||
            depth!=l4
        ){
            printf("size doesn't match\n");
            printf("expected: %d %d %d %d loaded: %d %d %d %d\n",InSize,HiddenSize,OutSize,depth,l1,l2,l3,l4);
            assert(false);
        }
        input_layer.load(f);
        for(int i = 0;i<depth;i++){
            hidden_layer[i].load(f);
        }
        output_layer.load(f);
    }
};

template<int InSize,int OutSize>
struct NN_data{
    static constexpr int input_len = InSize, output_size = OutSize;
    dl x[InSize];
    dl y[OutSize];
};



template<int InSize,int OutSize,int MemorySize ,int Time>
struct layer_RNN_record{
    static constexpr int InBUF = InSize+MemorySize;
    static constexpr int OutBUF = OutSize+MemorySize;
    static constexpr int TimeBUF = Time;
    dl (*Vinput)[InBUF]; //[TimeBUF][InBUF]
    dl (*Voutput)[OutBUF]; //[TimeBUF][OutBUF]
    layer_RNN_record(){
        Vinput = (dl(*)[InBUF])_aligned_malloc(TimeBUF * InBUF * sizeof(dl),32);
        Voutput = (dl(*)[OutBUF])_aligned_malloc(TimeBUF * OutBUF * sizeof(dl),32);
        memset(Vinput, 0, TimeBUF * InBUF * sizeof(dl));
        memset(Voutput, 0, TimeBUF * OutBUF * sizeof(dl));
    }
    ~layer_RNN_record(){
        _aligned_free(Vinput);
        _aligned_free(Voutput);
    }
    layer_RNN_record(layer_RNN_record&& other) noexcept {
        Vinput = other.Vinput;  other.Vinput = nullptr;
        Voutput = other.Voutput; other.Voutput = nullptr;
    }
    void place_mem(int t){
        if(t+1>=TimeBUF)return;
        memcpy(Vinput[t+1],Voutput[t],MemorySize * sizeof(dl));
    }
    void copy_out(dl (*output)[OutSize]){
        for(int i = 0;i<Time;i++){
            memcpy(output[i],Voutput[i]+MemorySize,sizeof(dl)*OutSize);
        }
    }
    layer_RNN_record(const layer_RNN_record&) = delete;
    layer_RNN_record& operator=(const layer_RNN_record&) = delete;
};
// single layer RNN
template<int InSize,int OutSize,int MemorySize,int DataLen>
struct layer_RNN{
    
    static constexpr int BUF = (InSize + MemorySize > OutSize + MemorySize)
                    ? (InSize + MemorySize) : (OutSize + MemorySize);
    static constexpr int input_size = InSize,
    output_size = OutSize,
    memory_size = MemorySize,
    data_len = DataLen;

    

    using Record = layer_RNN_record<InSize,OutSize,MemorySize,DataLen>;

    using _layer = Layer<InSize+MemorySize,OutSize+MemorySize,active_function::Tanh,init::He>;
    _layer ly;

    static constexpr int memory_required = _layer::memory_required+((MemorySize+3)&~3);

    dl *h0; // [MemorySize]
    dl *dlh0;
    ParamBlock& pbh;
    Record rd;
    layer_RNN(memory_allocator& mal):ly(mal),pbh(mal.alloc(MemorySize)){
        h0 = pbh.w;
        dlh0 = pbh.g;
    }
    inline void rev(dl*__restrict__ y,dl*__restrict__ grad_in, dl*__restrict__ grad_out){
        ly.rev(y,grad_in,grad_out);
    }
    
    void run(dl (*input)[InSize],dl (*output)[OutSize],int len){
        run(rd,input,output,len);
    }

    void run(Record& target_rd,dl (*input)[InSize],dl (*output)[OutSize],int len){
        assert(len<=DataLen);
        memcpy(target_rd.Vinput[0],h0,MemorySize*sizeof(dl));
        for(int t = 0;t<len;t++){
            if(input != nullptr)memcpy(target_rd.Vinput[t]+MemorySize,input[t],InSize*sizeof(dl));
            ly.run(target_rd.Vinput[t],target_rd.Voutput[t]);
            target_rd.place_mem(t);
            if(output != nullptr){
                memcpy(output[t],target_rd.Voutput[t]+MemorySize,OutSize*sizeof(dl));
            }
        }
    }

    void gtrain(Record& copied_rd,dl (*grad_in)[OutSize+MemorySize],dl (*grad_out)[InSize+MemorySize],int len){
        //[len][OutSize+MemorySize]
        //[len][InSize+MemorySize]
        // #pragma omp simd
        // for(int i = 0;i<len;i++){
        //     ly.train(copied_rd.Vinput[i],copied_rd.Voutput[i],grad_in[i],grad_out[i]);
        // }
        dl gradient[BUF]={},gradient2[BUF]={};
        dl* now_gradient = gradient,*last_gradient = gradient2;
        for(int t = len-1;t>=0;t--){
            // for(int i = 0;i<OutSize;i++){
            //     last_gradient[i+MemorySize] = grad_in[t][i+MemorySize];
            // }
            memcpy(last_gradient+MemorySize,grad_in[t]+MemorySize,sizeof(dl) * OutSize);
            ly.train(copied_rd.Vinput[t],copied_rd.Voutput[t],last_gradient,now_gradient);
            memcpy(grad_out[t]+MemorySize,now_gradient+MemorySize,sizeof(dl) * InSize);
            swap(now_gradient,last_gradient);
        }
        for(int i = 0;i<MemorySize;i++){
            dlh0[i] += last_gradient[i];
        }
        pbh.counter++;
    }

    dl train(dl (*input)[InSize],dl (*real_value)[OutSize],int len){
        // dl (*output)[OutSize];
        run(rd,input,nullptr,len);
        dl gradient[BUF]={},gradient2[BUF]={};
        dl* now_gradient = gradient,*last_gradient = gradient2;
        dl loss = 0;
        for(int t = len-1;t>=0;t--){
            #pragma omp simd reduction(+:loss) 
            for(int i = 0;i<OutSize;i++){
                last_gradient[i+MemorySize] = 2.0*(rd.Voutput[t][i+MemorySize] - real_value[t][i]);
                loss += (rd.Voutput[t][i+MemorySize] - real_value[t][i])*(rd.Voutput[t][i+MemorySize] - real_value[t][i]);
            }
            ly.train(rd.Vinput[t],rd.Voutput[t],last_gradient,now_gradient);
            swap(now_gradient,last_gradient);
        }
        for(int i = 0;i<MemorySize;i++){
            dlh0[i] += last_gradient[i];
        }
        loss /= OutSize*len;
        return loss;
    }

    void save(ofstream& f){
        char header_buf[256]={};
        memset(header_buf,' ',sizeof(header_buf));
        snprintf(header_buf, sizeof(header_buf), 
            "\nlayer_RNN: InSize: %d, OutSize: %d, Memory:%d \n",InSize,OutSize,MemorySize);
        f.write(header_buf, sizeof(header_buf));
        ly.save(f);                         
        f.write((char*)h0, MemorySize*sizeof(dl));
    }

    void load(ifstream& f){
        char header_buf[256]={};
        printf("loading a layer\n");
        f.read(header_buf, sizeof(header_buf));
        int l1,l2,l3;
        if (sscanf(header_buf, "\nlayer_RNN: InSize: %d, OutSize: %d, Memory:%d \n",&l1,&l2,&l3) != 3) {
            printf("Error: Layer structure in header is corrupted or format mismatch.\n");
            assert(false);
        }
        if(InSize!=l1 ||
            OutSize!=l2 ||
            MemorySize!=l3
        ){
            printf("size doesn't match\n");
            printf("expected: %d %d %d loaded: %d %d %d\n",InSize,OutSize,MemorySize,l1,l2,l3);
            assert(false);
        }
        ly.load(f); 
        f.read((char*)h0, MemorySize*sizeof(dl)); 
    }

};


template<int InSize,int HidSize,int OutSize,int MemorySize,int Depth,int Time>
struct stacked_RNN_record{
    layer_RNN_record<InSize,HidSize,MemorySize,Time> ird;
    layer_RNN_record<HidSize,HidSize,MemorySize,Time> hrd[Depth];
    layer_RNN_record<HidSize,OutSize,MemorySize,Time> ord;

    static constexpr int InBUF = InSize+MemorySize;
    static constexpr int HidBUF = HidSize+MemorySize;
    static constexpr int OutBUF = OutSize+MemorySize;
    static constexpr int TimeBUF = Time;

    stacked_RNN_record(){

    }
    stacked_RNN_record(const stacked_RNN_record&) = delete;
    stacked_RNN_record& operator=(const stacked_RNN_record&) = delete;


    int counter = 0;
    void move(){
        if(counter == 0){
            memcpy(hrd[0].Vinput,ird.Voutput,Time * HidBUF * sizeof(dl));
        }else if(counter < Depth){
            memcpy(hrd[counter].Vinput,hrd[counter-1].Voutput,Time * HidBUF * sizeof(dl));
        }else if(counter == Depth){
            memcpy(ord.Vinput,hrd[counter-1].Voutput,Time * HidBUF * sizeof(dl));
        }else{
            assert(false);
        }
        counter++;
    }
};

template<int InSize,int HidSize,int OutSize,int MemorySize,int Depth,int DataLen>
struct stacked_RNN{
    static constexpr int BUF = (InSize + MemorySize > OutSize + MemorySize)
                    ? (InSize + MemorySize) : (OutSize + MemorySize);
    static constexpr int input_size = InSize,
    output_size = OutSize,
    memory_size = MemorySize,
    data_len = DataLen,
    depth = Depth;
    
    

    using Record = stacked_RNN_record<InSize,HidSize,OutSize,MemorySize,Depth,DataLen>;

    using _ilayer = layer_RNN<InSize,HidSize,MemorySize,DataLen>;
    using _hlayer = layer_RNN<HidSize,HidSize,MemorySize,DataLen>;
    using _olayer = layer_RNN<HidSize,OutSize,MemorySize,DataLen>;

    static constexpr int memory_required = _ilayer::memory_required + _hlayer::memory_required*Depth + _olayer::memory_required;
    
    _ilayer input_layer;
    _hlayer* hidden_layer;
    _olayer output_layer;
    
    Record rd;

    stacked_RNN(memory_allocator& mal):input_layer(mal),output_layer(mal){
        if constexpr(Depth>0){
            hidden_layer = (_hlayer*)_aligned_malloc(Depth * sizeof(_hlayer),64);
        }else{
            hidden_layer = nullptr;
        }
        
        for(int i = 0;i<Depth;i++){
            new (&hidden_layer[i]) _hlayer(mal);
        }
    }

    void run(dl (*input)[InSize],dl (*output)[OutSize],int len){
        run(rd,input,output,len);
    }

    void run(Record& target_rd,dl (*input)[InSize],dl (*output)[OutSize],int len){
        target_rd.counter = 0;
        assert(len<=DataLen);
        input_layer.run(target_rd.ird,input,nullptr,len);
        target_rd.move();
        for(int i = 0;i<Depth;i++){
            hidden_layer[i].run(target_rd.hrd[i],nullptr,nullptr,len);
            target_rd.move();
        }
        output_layer.run(target_rd.ord,nullptr,nullptr,len);
        target_rd.ord.copy_out(output);
    }

    dl train(dl (*input)[InSize],dl (*real_value)[OutSize],int len){
        dl (*pred_value)[OutSize]; // [DataLen][OutSize]
        pred_value = (dl(*)[OutSize])_aligned_malloc(DataLen * OutSize * sizeof(dl),64);
        run(rd,input,pred_value,len);
        dl (*grad)[MemorySize+OutSize]; // [DataLen][MemorySize+OutSize]
        dl (*grad2)[MemorySize+HidSize]; // [DataLen][MemorySize+OutSize]
        grad = (dl(*)[MemorySize+OutSize])_aligned_malloc(DataLen * (MemorySize+OutSize) * sizeof(dl),64);
        grad2 = (dl(*)[MemorySize+HidSize])_aligned_malloc(DataLen * (MemorySize+HidSize) * sizeof(dl),64);
        memset(grad, 0, DataLen*(MemorySize+OutSize)*sizeof(dl));
        memset(grad2, 0, DataLen*(MemorySize+HidSize)*sizeof(dl));
        dl loss = 0;
        for(int i = 0;i<len;i++){
            for(int j = 0;j<OutSize;j++){
                grad[i][j+MemorySize] = 2.0*(pred_value[i][j] - real_value[i][j]);
                loss += (pred_value[i][j] - real_value[i][j])*(pred_value[i][j] - real_value[i][j]);
            }
        }
        loss /= len*OutSize;
        output_layer.gtrain(rd.ord,grad,grad2,len);
        dl (*grad_last)[MemorySize+HidSize] = grad2;
        dl (*grad_now)[MemorySize+HidSize] = (dl(*)[MemorySize+HidSize])_aligned_malloc(DataLen * (MemorySize+HidSize) * sizeof(dl),64);
        for(int i = Depth-1;i>=0;i--){
            hidden_layer[i].gtrain(rd.hrd[i],grad_last,grad_now,len);
            swap(grad_now,grad_last);
        }

        dl (*grad3)[MemorySize+InSize] = (dl(*)[MemorySize+InSize])_aligned_malloc(DataLen * (MemorySize+InSize) * sizeof(dl),64);
        input_layer.gtrain(rd.ird,grad_last,grad3,len);


        _aligned_free(pred_value);
        _aligned_free(grad_last);
        _aligned_free(grad_now);
        _aligned_free(grad);
        // _aligned_free(grad2);   is grad_last or grad_now, don't double free
        _aligned_free(grad3);
        return loss;
    }

    void save(ofstream& f){
        char header_buf[256]={};
        memset(header_buf,' ',sizeof(header_buf));
        snprintf(header_buf, sizeof(header_buf), 
            "\nstacked_RNN: InSize: %d, HidSize:%d, OutSize: %d, Memory:%d, Depth:%d \n",InSize,HidSize,OutSize,MemorySize,Depth);
        f.write(header_buf, sizeof(header_buf));
        
        input_layer.save(f);
        for(int i = 0;i<Depth;i++){
            hidden_layer[i].save(f);
        }
        output_layer.save(f);
    }

    void load(ifstream& f){
        char header_buf[256]={};
        printf("loading a layer\n");
        f.read(header_buf, sizeof(header_buf));
        int l1,l2,l3,l4,l5;
        if (sscanf(header_buf, "\nstacked_RNN: InSize: %d, HidSize:%d, OutSize: %d, Memory:%d, Depth:%d \n",&l1,&l2,&l3,&l4,&l5) != 5) {
            printf("Error: Layer structure in header is corrupted or format mismatch.\n");
            assert(false);
        }
        if(InSize!=l1 ||
            HidSize!=l2 ||
            OutSize!=l3 ||
            MemorySize!=l4 ||
            Depth!=l5
        ){
            printf("size doesn't match\n");
            printf("expected: %d %d %d %d %d loaded: %d %d %d %d %d\n",InSize,HidSize,OutSize,MemorySize,Depth,l1,l2,l3,l4,l5);
            assert(false);
        }
        
        input_layer.load(f);
        for(int i = 0;i<Depth;i++){
            hidden_layer[i].load(f);
        }
        output_layer.load(f);
    }


    ~stacked_RNN(){
        for(int i = 0; i < Depth; i++)
            hidden_layer[i].~_hlayer();
        if constexpr(Depth > 0) _aligned_free(hidden_layer);
    }

};


template<int InSize,int OutSize,int Len>
struct RNN_data{
    dl x[Len][InSize]={};
    dl y[Len][OutSize]={};
    int len;
};
#define RNN_PRED_LEN 100
struct model{
    using model_type = stacked_RNN<1,10,1,10,4,RNN_PRED_LEN>;
    static constexpr int memsize = model_type::memory_required;
    static constexpr int total_size = memsize;
    using Data = RNN_data<model::model_type::input_size,model::model_type::output_size,model::model_type::data_len>;
    TrainConfig& config;
    memory_allocator mal;
    model_type rnn;
    Trainer<total_size> tr;
    GradientFixer<total_size> gfix;

    model(TrainConfig &tconfig): 
       config(tconfig),
        mal{total_size, 1000},
        rnn{mal},
        tr{mal.g_block, mal.w_block, tconfig},
        gfix{mal.g_block, tconfig}{
        //pass
    }

    dl train(Data& d){
        return rnn.train(d.x,d.y,d.len);
    }

    void run(Data& d,dl (*out)[model::model_type::output_size]){
        rnn.run(d.x,out,d.len);
    }

    void save(string name){
        printf("file saving...\n");
        string file_name = "rnn_" + name + ".rnn";
        ofstream f(file_name, ios::binary);
        rnn.save(f);
        printf("file saved\n");
    }

    void load(string name){
        printf("file loading...\n");
        ifstream f(name, ios::binary);
        rnn.load(f);
        printf("file loaded\n");
    }

    void active_train(){
        mal.norm();
        gfix.fix();
        tr.train();
    }
};

using Data = RNN_data<model::model_type::input_size,model::model_type::output_size,model::model_type::data_len>;

inline dl border_(dl x,dl l,dl r){
    assert(l<r);
    return min(max(l,x),r);
}

Data gen_data(){
    Data d;
    d.len = RNN_PRED_LEN;
    static normal_distribution<dl> distribution(0.0, 0.2);
    d.y[0][0] = 0;
    for(int i = 0;i<RNN_PRED_LEN;i++){
        d.x[i][0] = border_(distribution(gen),-1.0,1.0);
        d.y[i][0] = fixed_sin(
            (i>=3?d.x[i-3][0]:0)+
            (i>=2?d.x[i-2][0]:0)
        )*0.8;
        
        
    }
    return d;
}


template<int batch>
struct hard_sample_mine{
    vector<queue<int>> qu{batch};
    dl avg = 0.5;
    dl late_avg = 0.75;
    int exist_data = 0;
    hard_sample_mine(int d){
        exist_data = d;
        for(int i = 0;i<d;i++){
            qu[0].push(i);
        }
    }
    void reset(int d){
        exist_data = d;
        avg = 0.5;
        late_avg = 0.75;
        for(int i = 0;i<batch;i++){
            while(!qu[i].empty()) qu[i].pop();
        }
        for(int i = 0;i<d;i++){
            qu[0].push(i);
        }
    }
    void extend(int d){
        for(int i = exist_data;i<d;i++){
            qu[0].push(i);
        }
        exist_data = d;
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
    static constexpr int n = 1000;
    TrainConfig tcf;
    model m(tcf);
    
    auto train_data_ptr = make_unique<array<Data, n>>();
    auto& train_data = *train_data_ptr;
    hard_sample_mine<5> hsm(n);
    for(int i = 0;i<n;i++){
        train_data[i] = gen_data();
    }
    int train_counter = 0;
    dl& lr = tcf.lr;
    // m.load("rnn_test3.rnn");
    // lr = 0.00005;
    dl last_loss_decay = -10;
    dl last_loss = 100;
    int combo = 0;
    for(int t = 0;t<50;t++){
        if(last_loss_decay == t-1){
            if(combo>=3)lr*=1.05;
            combo++;
        }else if(last_loss_decay <= t-3){
            lr*=0.9;
            combo = 0;
        }else{
            combo = 0;
        }

        dl loss = 0;
        for(int i = 0;i<n*10;i++){
            auto [return_id,batch_id] = hsm.get_index(2);
            dl hardness;
            loss+=hardness = m.train(train_data[return_id]);
            hsm.push(return_id,batch_id,hardness);
            train_counter++;
            if((train_counter&7)==0){
                m.active_train();
            }
        }
        loss /= (dl)(n * 10);
        if(loss<last_loss){
            last_loss_decay = t;
            last_loss = loss;
        }
        // last_loss = loss;
        printf("st: %d, loss = %.15lf, lr = %.15lf\n",t,loss,lr);
    }

    //test
    for(int t = 0;t<3;t++){
        Data d = gen_data();
        dl pred[RNN_PRED_LEN][1];
        m.run(d,pred);
        printf("REAL:");
        for(int i = 0;i<RNN_PRED_LEN;i++){
            printf(" %lf",d.y[i][0]);
        }printf("\n\n");
        printf("PRED:");
        for(int i = 0;i<RNN_PRED_LEN;i++){
            printf(" %lf",pred[i][0]);
        }printf("\n\n");
    }

    // m.save("test3");
}