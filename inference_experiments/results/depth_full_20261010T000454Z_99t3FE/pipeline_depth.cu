#define main previous_dp4a_main
#include "dp4a_benchmark.cu"
#undef main
#include <cublas_v2.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#define BLAS(call) do { auto s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS error " << int(s) << '\n'; std::exit(1); } } while (0)

static double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Settings {
    std::string output;
    double seconds = 3, warmup = 0.5;
    int trials = 5;
    std::vector<int> batches{1, 16, 128};
};

static Settings parse(int argc, char** argv) {
    Settings s;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (i + 1 == argc) throw std::runtime_error("Missing option value");
        std::string value = argv[++i];
        if (key == "--output") s.output = value;
        else if (key == "--seconds") s.seconds = std::stod(value);
        else if (key == "--warmup") s.warmup = std::stod(value);
        else if (key == "--trials") s.trials = std::stoi(value);
        else if (key == "--batches") {
            s.batches.clear();
            std::stringstream list(value);
            std::string item;
            while (std::getline(list, item, ',')) s.batches.push_back(std::stoi(item));
        } else throw std::runtime_error("Unknown option: " + key);
    }
    if (s.output.empty() || !std::isfinite(s.seconds) || s.seconds <= 0 ||
        !std::isfinite(s.warmup) || s.warmup < 0 || s.trials < 1 ||
        s.batches.empty()) throw std::runtime_error("Invalid settings");
    for (int b : s.batches) if (b < 1 || b > 128) throw std::runtime_error("Invalid batch");
    auto batches = s.batches;
    std::sort(batches.begin(), batches.end());
    if (std::adjacent_find(batches.begin(), batches.end()) != batches.end())
        throw std::runtime_error("Duplicate batch");
    return s;
}

struct PowerSensor {
    using Init = int(*)();
    using Handle = int(*)(unsigned, void**);
    using Query = int(*)(void*, unsigned*);
    using Shutdown = int(*)();
    void* lib = nullptr;
    void* device = nullptr;
    Query query = nullptr;
    Shutdown shutdown = nullptr;
    bool initialized = false, available = false;
    PowerSensor() {
        lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib) lib = dlopen("/usr/lib/wsl/lib/libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib) return;
        auto init = reinterpret_cast<Init>(dlsym(lib, "nvmlInit_v2"));
        auto handle = reinterpret_cast<Handle>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
        query = reinterpret_cast<Query>(dlsym(lib, "nvmlDeviceGetPowerUsage"));
        shutdown = reinterpret_cast<Shutdown>(dlsym(lib, "nvmlShutdown"));
        initialized = init && init() == 0;
        unsigned mw = 0;
        available = initialized && handle && query && handle(0, &device) == 0 &&
            query(device, &mw) == 0 && mw > 0;
    }
    ~PowerSensor() {
        if (initialized && shutdown) shutdown();
        if (lib) dlclose(lib);
    }
};

struct Config {
    std::string mode;
    int depth;
    bool asynchronous, pinned, serialized, sampling;
};

struct Slot {
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    cublasHandle_t blas = nullptr;
    int8_t* dx = nullptr;
    int32_t* dy = nullptr;
    int32_t* hy = nullptr;
    std::vector<int32_t> pageable_output;
    uint64_t id = 0;
    int bank = 0;
    double submitted = 0;
    bool pending = false;
};

struct Latency { uint64_t id; int bank, slot; double ms; };

int main(int argc, char** argv) {
    Settings settings;
    try { settings = parse(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 3; }
    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp gpu;
    CHECK_CUDA(cudaGetDeviceProperties(&gpu, 0));
    int cublas_version = 0, runtime_version = 0, driver_version = 0;
    BLAS(cublasGetVersion(nullptr, &cublas_version));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
    CHECK_CUDA(cudaDriverGetVersion(&driver_version));
    PowerSensor sensor;
    std::ofstream raw(settings.output + "/raw.csv"), power(settings.output + "/power.csv"),
        latency(settings.output + "/latency.csv"), metadata(settings.output + "/benchmark_metadata.txt");
    if (!raw || !power || !latency || !metadata) { std::cerr << "Cannot open outputs\n"; return 3; }
    raw << std::setprecision(17);
    power << std::setprecision(17);
    latency << std::setprecision(17);
    metadata << "GPU: " << gpu.name << "\ncuBLAS: " << cublas_version
        << "\nCUDA compiler: " << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__
        << '.' << __CUDACC_VER_BUILD__ << "\nCUDA runtime: " << runtime_version
        << "\nCUDA driver API: " << driver_version << "\ncopy engines: " << gpu.asyncEngineCount
        << "\nconcurrent kernels supported: " << gpu.concurrentKernels
        << "\npower query available: " << sensor.available << "\ndata seed: 12345"
        << "\norder seed: 20261010\ninput banks: 9\ninput features: 4096\noutput features: 512"
        << "\ntrials: " << settings.trials << "\ntimed seconds: " << settings.seconds
        << "\nwarmup seconds: " << settings.warmup << "\n";
    raw << "batch,mode,depth,telemetry,round,order,completed_requests,t0_s,t1_s,seconds,requests_s,inputs_s,amortized_ms,host_p50_ms,host_p95_ms,latency_samples,gpu_mean_w,gpu_est_j,gpu_est_mj_per_request,power_samples,power_coverage,power_max_gap_s,pinned_host_bytes,device_tensor_bytes,checked_requests,mismatches,stale_control_passed,injected_control_passed\n";
    power << "batch,mode,depth,telemetry,round,steady_s,power_w,in_window\n";
    latency << "batch,mode,depth,telemetry,round,request_id,input_bank,slot,host_observed_ms\n";
    const int banks = 9;
    std::vector<std::vector<Config>> groups{
        {{"serial_pageable", 1, false, false, true, true}},
        {{"serial_pinned", 1, false, true, true, true}},
        {{"serialized_8", 8, true, true, true, true}}
    };
    for (int depth : {1, 2, 4, 8}) groups.push_back({
        {"pipeline", depth, true, true, false, true},
        {"pipeline", depth, true, true, false, false}
    });
    std::mt19937 order_rng(20261010);
    int cells_done = 0;
    const int total_cells = int(settings.batches.size()) * settings.trials * 11;
    for (int batch : settings.batches) {
        const size_t xb = size_t(batch) * 4096, count = size_t(batch) * 512;
        const size_t yb = count * sizeof(int32_t), wb = size_t(512) * 4096;
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(-8, 7);
        std::vector<int8_t> weights(wb);
        for (auto& v : weights) v = dist(rng);
        std::vector<std::vector<int8_t>> inputs(banks, std::vector<int8_t>(xb));
        std::vector<std::vector<int32_t>> expected(banks, std::vector<int32_t>(count));
        for (int bank = 0; bank < banks; ++bank) {
            for (auto& v : inputs[bank]) v = dist(rng);
            cpu_ref(inputs[bank], weights, expected[bank], batch, 512, 4096);
        }
        int8_t* dw = nullptr;
        CHECK_CUDA(cudaMalloc(&dw, wb));
        CHECK_CUDA(cudaMemcpy(dw, weights.data(), wb, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
        for (int round = 0; round < settings.trials; ++round) {
            std::vector<int> group_order{0, 1, 2, 3, 4, 5, 6};
            std::shuffle(group_order.begin(), group_order.end(), order_rng);
            int ordinal = 0;
            for (int group_index : group_order) {
                auto configs = groups[group_index];
                if (configs.size() == 2 && (round + group_index) % 2)
                    std::reverse(configs.begin(), configs.end());
                for (const Config& config : configs) {
                    ++ordinal;
                    const std::string telemetry = config.sampling ? "sampling" : "off";
                    std::vector<int8_t*> hx(banks);
                    std::vector<Slot> slots(config.depth);
                    for (int bank = 0; bank < banks; ++bank) {
                        hx[bank] = inputs[bank].data();
                        if (config.pinned) {
                            CHECK_CUDA(cudaMallocHost(&hx[bank], xb));
                            std::memcpy(hx[bank], inputs[bank].data(), xb);
                        }
                    }
                    for (auto& slot : slots) {
                        CHECK_CUDA(cudaMalloc(&slot.dx, xb));
                        CHECK_CUDA(cudaMalloc(&slot.dy, yb));
                        if (config.pinned) CHECK_CUDA(cudaMallocHost(&slot.hy, yb));
                        else { slot.pageable_output.resize(count); slot.hy = slot.pageable_output.data(); }
                        if (config.asynchronous) {
                            CHECK_CUDA(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
                            CHECK_CUDA(cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming));
                        }
                        BLAS(cublasCreate(&slot.blas));
                        BLAS(cublasSetStream(slot.blas, slot.stream));
                    }
                    CHECK_CUDA(cudaDeviceSynchronize());
                    uint64_t completed = 0, checked = 0;
                    std::vector<Latency> latencies;
                    latencies.reserve(262144);
                    auto errors = [&](const Slot& slot) {
                        int bad = 0;
                        for (size_t i = 0; i < count; ++i) bad += slot.hy[i] != expected[slot.bank][i];
                        return bad;
                    };
                    auto fail = [&](const char* reason) {
                        std::cerr << reason << " batch=" << batch << " mode=" << config.mode
                            << " depth=" << config.depth << " round=" << round << '\n';
                        std::exit(2);
                    };
                    auto finish = [&](Slot& slot, bool check, bool collect) {
                        if (!slot.pending) return;
                        if (config.asynchronous) CHECK_CUDA(cudaEventSynchronize(slot.event));
                        double ms = (now_seconds() - slot.submitted) * 1000;
                        if (check) { ++checked; if (errors(slot)) fail("Output mismatch"); }
                        if (collect && slot.id % 31 == 0)
                            latencies.push_back({slot.id, slot.bank, int(slot.id % config.depth), ms});
                        ++completed;
                        slot.pending = false;
                    };
                    auto submit = [&](Slot& slot, uint64_t id) {
                        if (slot.pending) fail("Unsafe buffer reuse");
                        slot.id = id;
                        slot.bank = int(id % banks);
                        slot.submitted = now_seconds();
                        slot.pending = true;
                        if (config.asynchronous)
                            CHECK_CUDA(cudaMemcpyAsync(slot.dx, hx[slot.bank], xb, cudaMemcpyHostToDevice, slot.stream));
                        else CHECK_CUDA(cudaMemcpy(slot.dx, hx[slot.bank], xb, cudaMemcpyHostToDevice));
                        int32_t alpha = 1, beta = 0;
                        BLAS(cublasGemmEx(slot.blas, CUBLAS_OP_T, CUBLAS_OP_N, 512, batch, 4096,
                            &alpha, dw, CUDA_R_8I, 4096, slot.dx, CUDA_R_8I, 4096, &beta,
                            slot.dy, CUDA_R_32I, 512, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
                        CHECK_CUDA(cudaGetLastError());
                        if (config.asynchronous) {
                            CHECK_CUDA(cudaMemcpyAsync(slot.hy, slot.dy, yb, cudaMemcpyDeviceToHost, slot.stream));
                            CHECK_CUDA(cudaEventRecord(slot.event, slot.stream));
                        } else CHECK_CUDA(cudaMemcpy(slot.hy, slot.dy, yb, cudaMemcpyDeviceToHost));
                    };
                    auto step = [&](uint64_t id, bool check, bool collect) {
                        auto& slot = slots[id % config.depth];
                        finish(slot, check, collect);
                        submit(slot, id);
                        if (config.serialized) finish(slot, check, collect);
                    };
                    auto drain = [&](uint64_t n, bool check, bool collect) {
                        uint64_t first = n > uint64_t(config.depth) ? n - config.depth : 0;
                        for (uint64_t i = first; i < n; ++i) finish(slots[i % config.depth], check, collect);
                        for (const auto& slot : slots) if (slot.pending) fail("Undrained request");
                    };
                    step(0, true, false);
                    drain(1, true, false);
                    auto& control = slots[0];
                    control.bank = 1;
                    if (errors(control) == 0) fail("Stale-output control failed");
                    control.bank = 0;
                    control.hy[0] ^= 1;
                    if (errors(control) != 1) fail("Injected-error control failed");
                    control.hy[0] ^= 1;
                    checked = 0;
                    auto validate = [&]() {
                        for (uint64_t i = 0; i < 128; ++i) step(i, true, false);
                        drain(128, true, false);
                    };
                    validate();
                    uint64_t n = 0;
                    double warm = now_seconds();
                    while (now_seconds() - warm < settings.warmup) step(n++, false, false);
                    drain(n, false, false);
                    completed = 0;
                    std::atomic<bool> stop(false);
                    std::vector<std::pair<double, double>> samples;
                    samples.reserve(size_t(settings.seconds * 12 + 10));
                    // Keep the same sleeping worker in both arms; only NVML queries differ.
                    std::thread monitor([&]() {
                        while (!stop.load()) {
                            if (config.sampling && sensor.available) {
                                unsigned mw = 0;
                                double before = now_seconds();
                                int status = sensor.query(sensor.device, &mw);
                                double after = now_seconds();
                                if (status == 0 && mw > 0)
                                    samples.emplace_back((before + after) / 2, mw / 1000.0);
                            }
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        }
                    });
                    n = 0;
                    double t0 = now_seconds();
                    do { step(n++, false, true); } while (now_seconds() - t0 < settings.seconds);
                    drain(n, false, true);
                    double t1 = now_seconds(), seconds = t1 - t0;
                    stop = true;
                    monitor.join();
                    if (completed != n) fail("Completion count mismatch");
                    for (auto& slot : slots) { ++checked; if (errors(slot)) fail("Final buffer mismatch"); }
                    validate();
                    double integral = 0, span = 0, max_gap = 0;
                    int points = 0;
                    std::pair<double, double> previous;
                    for (auto sample : samples) {
                        bool in_window = sample.first >= t0 && sample.first <= t1;
                        power << batch << ',' << config.mode << ',' << config.depth << ',' << telemetry
                            << ',' << round << ',' << sample.first << ',' << sample.second << ',' << in_window << '\n';
                        if (!in_window) continue;
                        if (points) {
                            double dt = sample.first - previous.first;
                            span += dt;
                            integral += dt * (sample.second + previous.second) / 2;
                            max_gap = std::max(max_gap, dt);
                        }
                        previous = sample;
                        ++points;
                    }
                    double coverage = span / seconds, mean_w = std::numeric_limits<double>::quiet_NaN();
                    if (points >= 20 && coverage >= 0.9 && max_gap <= 0.5) mean_w = integral / span;
                    double joules = mean_w * seconds, mj = joules * 1000 / n;
                    std::vector<double> values;
                    values.reserve(latencies.size());
                    for (auto sample : latencies) {
                        latency << batch << ',' << config.mode << ',' << config.depth << ',' << telemetry
                            << ',' << round << ',' << sample.id << ',' << sample.bank << ',' << sample.slot
                            << ',' << sample.ms << '\n';
                        values.push_back(sample.ms);
                    }
                    std::sort(values.begin(), values.end());
                    double p50 = values[(values.size() - 1) / 2];
                    double p95 = values[size_t(0.95 * (values.size() - 1))];
                    double rate = n / seconds;
                    raw << batch << ',' << config.mode << ',' << config.depth << ',' << telemetry << ','
                        << round << ',' << ordinal << ',' << n << ',' << t0 << ',' << t1 << ',' << seconds
                        << ',' << rate << ',' << batch * rate << ',' << 1000 / rate << ',' << p50 << ',' << p95
                        << ',' << values.size() << ',' << mean_w << ',' << joules << ',' << mj << ',' << points
                        << ',' << coverage << ',' << max_gap << ',' << (config.pinned ? banks * xb + config.depth * yb : 0)
                        << ',' << wb + config.depth * (xb + yb) << ',' << checked << ",0,1,1\n";
                    raw.flush(); power.flush(); latency.flush();
                    std::cout << "cell " << ++cells_done << '/' << total_cells << " batch=" << batch
                        << " mode=" << config.mode << " depth=" << config.depth << " telemetry=" << telemetry
                        << " trial=" << round + 1 << '/' << settings.trials << " requests/s=" << rate << std::endl;
                    for (auto& slot : slots) {
                        BLAS(cublasDestroy(slot.blas));
                        if (config.asynchronous) {
                            CHECK_CUDA(cudaEventDestroy(slot.event));
                            CHECK_CUDA(cudaStreamDestroy(slot.stream));
                        }
                        CHECK_CUDA(cudaFree(slot.dx)); CHECK_CUDA(cudaFree(slot.dy));
                        if (config.pinned) CHECK_CUDA(cudaFreeHost(slot.hy));
                    }
                    if (config.pinned) for (auto ptr : hx) CHECK_CUDA(cudaFreeHost(ptr));
                }
            }
        }
        CHECK_CUDA(cudaFree(dw));
    }
    if (cells_done != total_cells) { std::cerr << "Incomplete sweep\n"; return 2; }
    std::cout << "COMPLETE: all correctness and negative controls passed; " << cells_done << " trials\n";
    return 0;
}
