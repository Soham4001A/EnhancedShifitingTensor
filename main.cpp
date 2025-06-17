// =================================================================================
// Universal Compilation Instructions (GCC/Clang/MSVC)
// =================================================================================
// Linux (GCC) / macOS (Clang):
//   g++ -O3 -std=c++20 -march=native -pthread -o shift main.cpp
//
// Windows (MSVC):
//   cl.exe /O2 /EHsc /std:c++20 /arch:AVX2 /Fe:shift.exe main.cpp
// =================================================================================

#include <algorithm>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>

// --- Platform-specific includes & helpers ---
#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
#elif defined(__aarch64__)
  #include <arm_neon.h>
#endif
#if defined(_WIN32)
  #include <malloc.h>
#endif
#if defined(__GNUC__) || defined(__clang__)
  #define TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#else
  #define TARGET_AVX2_FMA
#endif

// --- Base hyper-parameters (Realistic for a ~7B parameter model's FFN layer) ---
static constexpr int B_base = 32;
static constexpr int I_base = 4096;   // Represents d_model
static constexpr int O_base = 16384;  // Represents d_ffn (4 * d_model)
static constexpr int Nwarmup = 2;
static constexpr int Nbench  = 10;

// --- Helper: aligned malloc/free ---
void* aligned_malloc(size_t bytes, size_t align = 64) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, align);
#else
    void* p; if (posix_memalign(&p, align, bytes) != 0) return nullptr; return p;
#endif
}
void aligned_free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

// ===================================================================================
//  BASELINE LIBRARY APPROACH: PRE-PACKED WEIGHTS
// ===================================================================================
struct PackedWeights {
    float* W_packed;
    PackedWeights(const float* W, int i_dim, int o_dim) {
#if defined(__x86_64__) || defined(_M_X64)
        constexpr int O_TILE = 8;
#elif defined(__aarch64__)
        constexpr int O_TILE = 4;
#endif
        W_packed = static_cast<float*>(aligned_malloc(size_t(i_dim) * o_dim * sizeof(float)));
        for (int o_base = 0; o_base < o_dim; o_base += O_TILE) {
            for (int i = 0; i < i_dim; ++i) {
                for (int o_off = 0; o_off < O_TILE; ++o_off) {
                    const float* src = W + size_t(o_base + o_off) * i_dim + i;
                    float* dst = W_packed + size_t(o_base/O_TILE)*i_dim*O_TILE + size_t(i)*O_TILE + o_off;
                    *dst = *src;
                }
            }
        }
    }
    ~PackedWeights() { aligned_free(W_packed); }
};

#if defined(__x86_64__) || defined(_M_X64)
TARGET_AVX2_FMA
__attribute__((noinline))
void forward_packed(const float* A, const PackedWeights& P, float* Y, int b_dim, int i_dim, int o_dim) {
    constexpr int O_TILE = 8;
    for (int b = 0; b < b_dim; ++b) {
        for (int o_base = 0; o_base < o_dim; o_base += O_TILE) {
            __m256 vsum = _mm256_setzero_ps();
            const float* a_ptr = A + (size_t)b * i_dim;
            const float* w_pack_ptr = P.W_packed + size_t(o_base/O_TILE)*i_dim*O_TILE;
            for (int i = 0; i < i_dim; ++i) {
                __m256 va_splat = _mm256_set1_ps(*(a_ptr + i));
                __m256 vw_packed = _mm256_load_ps(w_pack_ptr + (size_t)i*O_TILE);
                vsum = _mm256_fmadd_ps(va_splat, vw_packed, vsum);
            }
            _mm256_store_ps(Y + (size_t)b*o_dim + o_base, vsum);
        }
    }
}
#elif defined(__aarch64__)
__attribute__((noinline))
void forward_packed(const float* A, const PackedWeights& P, float* Y, int b_dim, int i_dim, int o_dim) {
    constexpr int O_TILE = 4;
    for (int b = 0; b < b_dim; ++b) {
        for (int o_base = 0; o_base < o_dim; o_base += O_TILE) {
            float32x4_t vsum = vdupq_n_f32(0.f);
            const float* a_ptr = A + (size_t)b * i_dim;
            const float* w_pack_ptr = P.W_packed + size_t(o_base/O_TILE)*i_dim*O_TILE;
            for (int i = 0; i < i_dim; ++i) {
                float32x4_t va_splat = vld1q_dup_f32(a_ptr + i);
                float32x4_t vw_packed = vld1q_f32(w_pack_ptr + (size_t)i*O_TILE);
                vsum = vfmaq_f32(vsum, va_splat, vw_packed);
            }
            vst1q_f32(Y + (size_t)b*o_dim + o_base, vsum);
        }
    }
}
#endif


// ===================================================================================
//  V3 INNOVATION: PERSISTENT THREAD, VECTORIZED SHIFT, AND PREFETCHING
// ===================================================================================
#if defined(__x86_64__) || defined(_M_X64)

// State for communication between main thread and persistent helper thread
struct WorkerState {
    const float* W_ptr = nullptr;
    float* target_buf = nullptr;
    int o_base = 0, i_base = 0, i_dim = 0;

    std::mutex mtx;
    std::condition_variable main_cv, worker_cv;
    std::atomic<bool> work_ready{false};
    std::atomic<bool> work_done{true};
    std::atomic<bool> terminate{false};
};

TARGET_AVX2_FMA
static void shift_panel_vectorized(const float* w_src, float* w_dst, int i_dim) {
    constexpr int O_TILE = 8;
    constexpr int I_BLOCK = 256;
    for (int i_off = 0; i_off < I_BLOCK; ++i_off) {
        // This is a gather operation: load 8 floats from scattered locations.
        // It manually transposes the data as it loads.
        __m256 row = _mm256_set_ps(
            w_src[7 * i_dim + i_off], w_src[6 * i_dim + i_off],
            w_src[5 * i_dim + i_off], w_src[4 * i_dim + i_off],
            w_src[3 * i_dim + i_off], w_src[2 * i_dim + i_off],
            w_src[1 * i_dim + i_off], w_src[0 * i_dim + i_off]
        );
        // Store them into a contiguous block in the destination buffer.
        _mm256_store_ps(w_dst + i_off * O_TILE, row);
    }
}

// The function the persistent helper thread will run
void shift_worker(WorkerState& state) {
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.worker_cv.wait(lock, [&]{ return state.work_ready.load() || state.terminate.load(); });

        if (state.terminate.load()) return;
        
        shift_panel_vectorized(state.W_ptr + (size_t)state.o_base * state.i_dim + state.i_base, state.target_buf, state.i_dim);

        state.work_done = true;
        state.work_ready = false;
        lock.unlock();
        state.main_cv.notify_one();
    }
}

TARGET_AVX2_FMA
__attribute__((noinline))
void forward_pipelined_v3(const float* A, const float* W, float* Y, int b_dim, int i_dim, int o_dim) {
    constexpr int O_TILE = 8;
    constexpr int I_BLOCK = 256;
    alignas(64) float w_buf_A[I_BLOCK * O_TILE], w_buf_B[I_BLOCK * O_TILE];
    
    WorkerState state;
    state.i_dim = i_dim;
    std::thread worker(shift_worker, std::ref(state));

    for (int o_base = 0; o_base < o_dim; o_base += O_TILE) {
        for(int b=0; b<b_dim; ++b) std::memset(Y + (size_t)b*o_dim + o_base, 0, O_TILE * sizeof(float));

        float* w_compute_buf = w_buf_A;
        float* w_shift_buf = w_buf_B;
        
        // Prime the pipeline: Synchronously shift the first block
        shift_panel_vectorized(W + (size_t)o_base * i_dim, w_compute_buf, i_dim);

        for (int i_base = 0; i_base < i_dim; i_base += I_BLOCK) {
            // STEP 1: LAUNCH SHIFT FOR NEXT BLOCK (N+1) using the persistent thread
            if (i_base + I_BLOCK < i_dim) {
                // Prefetch data for block N+2 to hide L2/L3 latency
                _mm_prefetch(W + (size_t)o_base * i_dim + i_base + 2*I_BLOCK, _MM_HINT_T0);

                std::unique_lock<std::mutex> lock(state.mtx);
                state.target_buf = w_shift_buf;
                state.W_ptr = W; // Pass base pointer
                state.o_base = o_base;
                state.i_base = i_base + I_BLOCK;
                state.work_done = false;
                state.work_ready = true;
                lock.unlock();
                state.worker_cv.notify_one();
            }

            // STEP 2: COMPUTE ON CURRENT BLOCK (N)
            for (int b = 0; b < b_dim; ++b) {
                const float* a_ptr = A + (size_t)b * i_dim + i_base;
                float* y_ptr = Y + (size_t)b * o_dim + o_base;
                __m256 vsum = _mm256_load_ps(y_ptr);
                for (int i_off = 0; i_off < I_BLOCK; ++i_off) {
                    vsum = _mm256_fmadd_ps(_mm256_set1_ps(*(a_ptr + i_off)), _mm256_load_ps(w_compute_buf + i_off * O_TILE), vsum);
                }
                _mm256_store_ps(y_ptr, vsum);
            }
            
            // STEP 3: SYNCHRONIZE with helper thread and SWAP buffers
            if (i_base + I_BLOCK < i_dim) {
                std::unique_lock<std::mutex> lock(state.mtx);
                state.main_cv.wait(lock, [&]{ return state.work_done.load(); });
            }
            std::swap(w_compute_buf, w_shift_buf);
        }
    }
    
    // Cleanly terminate the worker thread
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.terminate = true;
        state.work_ready = true; // Wake up the worker one last time
        lock.unlock();
        state.worker_cv.notify_one();
    }
    worker.join();
}
#elif defined(__aarch64__)
// An ARM NEON v3 implementation would go here, following the same logic
// with pthreads/std::thread and NEON intrinsics for the shift.
void forward_pipelined_v3(const float* A, const float* W, float* Y, int b_dim, int i_dim, int o_dim) {
    // Placeholder for ARM
    printf("V3 Pipelined not implemented for ARM yet.\n");
}
#endif

// Main function to run the benchmark
int main() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__)
    for (int scale : {1, 2, 4}) {
        printf("\n--- Running at Scale: %dx (B=%d, I=%d, O=%d) ---\n",
               scale, B_base * scale, I_base * scale, O_base * scale);

        const int B = B_base * scale;
        const int I = I_base * scale;
        const int O = O_base * scale;

        auto A = static_cast<float*>(aligned_malloc(size_t(B)*I*sizeof(float)));
        auto W = static_cast<float*>(aligned_malloc(size_t(I)*O*sizeof(float)));
        auto Y = static_cast<float*>(aligned_malloc(size_t(B)*O*sizeof(float)));
        
        std::mt19937 rng(42); std::uniform_real_distribution<float> d(-1,1);
        for(size_t i=0; i<size_t(B)*I; ++i) A[i] = d(rng);
        for(size_t i=0; i<size_t(I)*O; ++i) W[i] = d(rng);

        double packed_ms = 0;
        {
            printf("Benchmarking 'Packed' (Library-style) Approach...\n");
            PackedWeights P(W, I, O);
            auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < Nbench; ++k) forward_packed(A, P, Y, B, I, O);
            auto t2 = std::chrono::steady_clock::now();
            packed_ms = std::chrono::duration<double, std::milli>(t2-t1).count();
        }

        double pipelined_ms = 0;
        {
            printf("Benchmarking 'Enhanced Pipelined (V3)' Approach...\n");
            auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < Nbench; ++k) forward_pipelined_v3(A, W, Y, B, I, O);
            auto t2 = std::chrono::steady_clock::now();
            pipelined_ms = std::chrono::duration<double, std::milli>(t2-t1).count();
        }

        printf("\n--- Results (Scale %dx) ---\n", scale);
        const double flops = double(B) * I * O * 2;
        double gflops_packed = flops / (packed_ms / Nbench * 1e6);
        double gflops_pipelined = flops / (pipelined_ms / Nbench * 1e6);
        printf("Packed (Library Style)       : %.2f GFLOP/s\n", gflops_packed);
        printf("Enhanced Pipelined (V3 Idea) : %.2f GFLOP/s\n", gflops_pipelined);
        printf("\nSpeed-up of V3 Pipelined vs Packed: %.2fx\n", gflops_pipelined / gflops_packed);

        aligned_free(A);
        aligned_free(W);
        aligned_free(Y);
    }
#else
    puts("This benchmark requires NEON or AVX support.");
#endif
    return 0;
}