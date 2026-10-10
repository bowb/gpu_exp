#define main previous_dp4a_main
#include "dp4a_benchmark.cu"
#undef main
#include <cublas_v2.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#define BLAS(call) do { auto s=(call); if(s!=CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cuBLAS error %d\n",int(s)); exit(1); } } while(0)
static int mismatch(const std::vector<int32_t>& a,const std::vector<int32_t>& b) { int n=0; for(size_t i=0;i<a.size();++i) n+=(a[i]!=b[i]); return n;
} int main() {
    cublasHandle_t handle; BLAS(cublasCreate(&handle)); cudaDeviceProp gpu; CHECK_CUDA(cudaGetDeviceProperties(&gpu,0)); int version=0; BLAS(cublasGetVersion(handle,&version));
    std::ofstream raw("../results/gpu_compare_raw.csv"), report("../results/GPU_COMPARISON.md"); if(!raw || !report) { std::cerr<<"Cannot open result files\n"; return 3; }
    raw<<"batch,method,scenario,round,repeats,latency_ms,mismatches\n"; report<<"# GPU Comparison\n\nGPU: "<<gpu.name<<" cuBLAS version: "<<version<<"\n\nSame int8 inputs, int32 outputs; seed 12345. All errors must be zero.\n";
    report<<"\nSeven samples per row; median per-operation time. Compute: 50 operations/sample. Transfers: 8 requests/sample.\n"; report<<"Pageable host memory; allocations, initialization and CPU reference are outside timing. Energy is not measured.\n";
    report<<"\n| Batch | Scenario | DP4A ms | cuBLAS ms | DP4A speedup vs cuBLAS |\n|---|---|---:|---:|---:|\n"; const char* modes[]={"compute_only","weights_resident","upload_all"};
    for(int batch : {1,4,16,128}) { std::vector<int8_t> x(batch*4096),w(512*4096);
        std::vector<int32_t> expected(batch*512),out(batch*512); std::mt19937 rng(12345); std::uniform_int_distribution<int> dist(-8,7);
        for(auto& v:x) v=dist(rng); for(auto& v:w) v=dist(rng); cpu_ref(x,w,expected,batch,512,4096);
        out=expected; out[0]^=1; if(mismatch(out,expected)!=1) return 2; int8_t *dx,*dw; int32_t* dy;
        CHECK_CUDA(cudaMalloc(&dx,x.size())); CHECK_CUDA(cudaMalloc(&dw,w.size())); CHECK_CUDA(cudaMalloc(&dy,out.size()*4));
        CHECK_CUDA(cudaMemcpy(dx,x.data(),x.size(),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
        auto launch=[&](int method) { if(method==0) dp4a_matmul_kernel<<<dim3(32,(batch+15)/16),dim3(16,16)>>>(dx,dw,dy,batch,512,4096);
            else { int32_t alpha=1,beta=0; BLAS(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,512,batch,4096,&alpha,dw,CUDA_R_8I,4096,dx,CUDA_R_8I,4096,&beta,dy,CUDA_R_32I,512,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT)); } CHECK_CUDA(cudaGetLastError());
        }; std::vector<double> times[2][3];
        for(int round=0;round<7;++round) for(int mode=0;mode<3;++mode) for(int order=0;order<2;++order) { int method=(order+round)%2,repeats=mode==0?50:8;
            auto request=[&]() { if(mode) CHECK_CUDA(cudaMemcpy(dx,x.data(),x.size(),cudaMemcpyHostToDevice));
                if(mode==2) CHECK_CUDA(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice)); launch(method);
                if(mode) CHECK_CUDA(cudaMemcpy(out.data(),dy,out.size()*4,cudaMemcpyDeviceToHost)); };
            for(int i=0;i<5;++i) request(); CHECK_CUDA(cudaDeviceSynchronize()); double ms;
            if(mode==0) { cudaEvent_t a,b; CHECK_CUDA(cudaEventCreate(&a)); CHECK_CUDA(cudaEventCreate(&b));
                CHECK_CUDA(cudaEventRecord(a)); for(int i=0;i<repeats;++i) request(); CHECK_CUDA(cudaEventRecord(b)); CHECK_CUDA(cudaEventSynchronize(b)); float elapsed; CHECK_CUDA(cudaEventElapsedTime(&elapsed,a,b));
                ms=elapsed/repeats; CHECK_CUDA(cudaEventDestroy(a)); CHECK_CUDA(cudaEventDestroy(b)); } else {
                auto a=std::chrono::steady_clock::now(); for(int i=0;i<repeats;++i) request(); ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count()/repeats;
            } CHECK_CUDA(cudaMemcpy(out.data(),dy,out.size()*4,cudaMemcpyDeviceToHost));
            int errors=mismatch(out,expected); raw<<batch<<','<<(method==0?"dp4a":"cublas")<<','<<modes[mode]<<','<<round<<','<<repeats<<','<<ms<<','<<errors<<'\n';
            if(errors) { std::cerr<<"Correctness failed\n"; return 2; } times[method][mode].push_back(ms);
        } for(int mode=0;mode<3;++mode) {
            for(int method=0;method<2;++method) std::sort(times[method][mode].begin(),times[method][mode].end()); double a=times[0][mode][3],b=times[1][mode][3];
            report<<"| "<<batch<<" | "<<modes[mode]<<" | "<<a<<" | "<<b<<" | "<<b/a<<"x |\n"; std::cout<<"batch="<<batch<<" "<<modes[mode]<<" DP4A="<<a<<"ms cuBLAS="<<b<<"ms DP4A_speedup="<<b/a<<"x\n";
        } CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(dw)); CHECK_CUDA(cudaFree(dy));
    } report<<"\nAll correctness checks passed, including an injected-error control. cuBLAS uses its default algorithm heuristic; no exhaustive tuning.\n";
    report<<"Resident: upload inputs + compute + download output. Upload-all also uploads weights. All requests await completion.\n"; BLAS(cublasDestroy(handle)); std::cout<<"All correctness checks passed. Results saved.\n"; return 0;
}
