#define main previous_dp4a_main
#include "dp4a_benchmark.cu"
#undef main
#include <cublas_v2.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <atomic>
#include <limits>
#include <utility>
#include <dlfcn.h>
#define BLAS(call) do { auto s=(call); if(s!=CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cuBLAS error %d\n",int(s)); exit(1); } } while(0)
static int mismatch(const std::vector<int32_t>& a,const std::vector<int32_t>& b) { int n=0; for(size_t i=0;i<a.size();++i) n+=(a[i]!=b[i]); return n;
} int main() {
    void* lib=dlopen("libnvidia-ml.so.1",RTLD_LAZY); if(!lib) lib=dlopen("/usr/lib/wsl/lib/libnvidia-ml.so.1",RTLD_LAZY);
    using Init=int(*)(); using Handle=int(*)(unsigned,void**); using Power=int(*)(void*,unsigned*);
    auto init=lib?reinterpret_cast<Init>(dlsym(lib,"nvmlInit_v2")):nullptr; auto device=lib?reinterpret_cast<Handle>(dlsym(lib,"nvmlDeviceGetHandleByIndex_v2")):nullptr; auto power=lib?reinterpret_cast<Power>(dlsym(lib,"nvmlDeviceGetPowerUsage")):nullptr;
    void* gpu_handle=nullptr; bool sensor=!getenv("DISABLE_POWER")&&init&&device&&power&&init()==0&&device(0,&gpu_handle)==0;
    auto ticks=[](){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();};
    cublasHandle_t handle; BLAS(cublasCreate(&handle)); cudaDeviceProp gpu; CHECK_CUDA(cudaGetDeviceProperties(&gpu,0)); int version=0; BLAS(cublasGetVersion(handle,&version));
    std::ofstream raw("../results/transfer_energy_raw.csv"), report("../results/TRANSFER_ENERGY.md"); if(!raw || !report) { std::cerr<<"Cannot open result files\n"; return 3; }
    std::ofstream power_raw("../results/transfer_power_raw.csv"); if(!power_raw) return 3; raw.precision(12); power_raw.precision(12); power_raw<<"batch,method,scenario,round,steady_s,power_w\n"; raw<<"batch,method,scenario,round,repeats,latency_ms,mismatches,seconds,gpu_mean_w,gpu_est_j,gpu_est_mj_per_request,power_samples,power_coverage\n"; report<<"# Transfer and Energy Comparison\n\nGPU: "<<gpu.name<<" cuBLAS version: "<<version<<"\n\nSame int8 inputs, int32 outputs; seed 12345. All errors must be zero.\n";
    report<<"\nThree samples per row; median per-operation time. At least 5 seconds/sample after 1 second warm-up; all timings are host wall averages, including dispatch and synchronization.\n"; report<<"Pageable versus CUDA-allocated pinned memory; allocations, initialization and CPU reference are outside timing. GPU energy is estimated from sampled board power in the raw CSV, not whole-PC energy. nan means unavailable or insufficient coverage. Other GPU applications contribute.\n";
    report<<"\n| Batch | Scenario | Pageable ms | Pinned ms | Pinned speedup |\n|---|---|---:|---:|---:|\n"; const char* modes[]={"compute_no_transfers","weights_resident","upload_all"};
    for(int batch : {1,16,128}) { std::vector<int8_t> x(batch*4096),w(512*4096);
        std::vector<int32_t> expected(batch*512),out(batch*512); std::mt19937 rng(12345); std::uniform_int_distribution<int> dist(-8,7);
        for(auto& v:x) v=dist(rng); for(auto& v:w) v=dist(rng); cpu_ref(x,w,expected,batch,512,4096);
        out=expected; out[0]^=1; if(mismatch(out,expected)!=1) return 2; int8_t *dx,*dw; int32_t* dy;
        CHECK_CUDA(cudaMalloc(&dx,x.size())); CHECK_CUDA(cudaMalloc(&dw,w.size())); CHECK_CUDA(cudaMalloc(&dy,out.size()*4));
        CHECK_CUDA(cudaMemcpy(dx,x.data(),x.size(),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
        auto launch=[&](int method) { { int32_t alpha=1,beta=0; BLAS(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,512,batch,4096,&alpha,dw,CUDA_R_8I,4096,dx,CUDA_R_8I,4096,&beta,dy,CUDA_R_32I,512,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT)); } CHECK_CUDA(cudaGetLastError());
        }; std::vector<double> times[2][3];
        for(int round=0;round<3;++round) for(int mode=0;mode<3;++mode) for(int order=0;order<2;++order) { int method=(order+round)%2,repeats=mode==0?50:8;
            int8_t *hx=x.data(),*hw=w.data(); int32_t* hy=out.data();
            if(method) { CHECK_CUDA(cudaMallocHost(&hx,x.size())); CHECK_CUDA(cudaMallocHost(&hw,w.size())); CHECK_CUDA(cudaMallocHost(&hy,out.size()*4)); std::memcpy(hx,x.data(),x.size()); std::memcpy(hw,w.data(),w.size()); }
            auto request=[&]() { if(mode) CHECK_CUDA(cudaMemcpy(dx,hx,x.size(),cudaMemcpyHostToDevice));
                if(mode==2) CHECK_CUDA(cudaMemcpy(dw,hw,w.size(),cudaMemcpyHostToDevice)); launch(method);
                if(mode) CHECK_CUDA(cudaMemcpy(hy,dy,out.size()*4,cudaMemcpyDeviceToHost)); };
            double warm=ticks(); do { for(int i=0;i<256;++i) request(); CHECK_CUDA(cudaDeviceSynchronize()); } while(ticks()-warm<1.0);
            std::atomic<bool> stop(false); std::vector<std::pair<double,double>> samples;
            std::thread monitor([&](){ while(!stop.load()) { unsigned mw=0; double before=ticks(); int status=sensor?power(gpu_handle,&mw):-1; double after=ticks(); if(status==0&&mw>0) samples.emplace_back((before+after)/2,mw/1000.0); std::this_thread::sleep_for(std::chrono::milliseconds(100)); } });
            repeats=0; double t0=ticks(); do { for(int i=0;i<256;++i) request(); CHECK_CUDA(cudaDeviceSynchronize()); repeats+=256; } while(ticks()-t0<5.0);
            double t1=ticks(),seconds=t1-t0; stop=true; monitor.join(); double ms=seconds*1000/repeats;
            double integral=0,span=0,max_gap=0; int points=0; std::pair<double,double> prev;
            for(auto p:samples) { power_raw<<batch<<','<<(method?"pinned":"pageable")<<','<<modes[mode]<<','<<round<<','<<p.first<<','<<p.second<<'\n'; if(p.first<t0||p.first>t1) continue; if(points) { double dt=p.first-prev.first; integral+=(prev.second+p.second)*0.5*dt; span+=dt; max_gap=std::max(max_gap,dt); } prev=p; ++points; }
            double coverage=span/seconds,mean_w=std::numeric_limits<double>::quiet_NaN(); if(points>=20&&coverage>=0.9&&max_gap<=0.5) mean_w=integral/span; double gpu_j=mean_w*seconds;
            if(mode==0) CHECK_CUDA(cudaMemcpy(out.data(),dy,out.size()*4,cudaMemcpyDeviceToHost)); else if(method) std::copy(hy,hy+out.size(),out.begin());
            int errors=mismatch(out,expected); raw<<batch<<','<<(method==0?"pageable":"pinned")<<','<<modes[mode]<<','<<round<<','<<repeats<<','<<ms<<','<<errors<<','<<seconds<<','<<mean_w<<','<<gpu_j<<','<<gpu_j*1000/repeats<<','<<points<<','<<coverage<<'\n';
            if(errors) { std::cerr<<"Correctness failed\n"; return 2; } times[method][mode].push_back(ms); raw.flush(); power_raw.flush(); if(method) { CHECK_CUDA(cudaFreeHost(hx)); CHECK_CUDA(cudaFreeHost(hw)); CHECK_CUDA(cudaFreeHost(hy)); }
        } for(int mode=0;mode<3;++mode) {
            for(int method=0;method<2;++method) std::sort(times[method][mode].begin(),times[method][mode].end()); double a=times[0][mode][1],b=times[1][mode][1];
            report<<"| "<<batch<<" | "<<modes[mode]<<" | "<<a<<" | "<<b<<" | "<<a/b<<"x |\n"; std::cout<<"batch="<<batch<<" "<<modes[mode]<<" pageable="<<a<<"ms pinned="<<b<<"ms pinned_speedup="<<a/b<<"x\n";
        } CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(dw)); CHECK_CUDA(cudaFree(dy));
    } report<<"\nAll correctness checks passed, including an injected-error control. cuBLAS uses its default algorithm heuristic; no exhaustive tuning.\n";
    report<<"Resident: upload inputs + compute + download output. Upload-all also uploads weights. All requests await completion.\n"; BLAS(cublasDestroy(handle)); std::cout<<"All correctness checks passed. Results saved.\n"; return 0;
}
