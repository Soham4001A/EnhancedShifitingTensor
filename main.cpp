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
#include <future> // For std::async to achieve true overlap

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

// --- Base hyper-parameters ---
static constexpr int B_base = 256;
static constexpr int I_base = 4096;
static constexpr int O_base = 4096;
static constexpr int Nwarmup = 2; // Reduced for faster testing at scale
static constexpr int Nbench  = 10; // Reduced for faster testing at scale

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
//  V2 INNOVATION: ENHANCED PIPELINE WITH TRUE OVERLAP AND AMORTIZED SHIFTING
// ===================================================================================
#if defined(__x86_64__) || defined(_M_X64)
TARGET_AVX2_FMA
__attribute__((noinline))
void forward_pipelined_enhanced(const float* A, const float* W, float* Y, int b_dim, int i_dim, int o_dim) {
    constexpr int O_TILE = 8;
    constexpr int I_BLOCK = 256;
    alignas(64) float w_buf_A[I_BLOCK * O_TILE], w_buf_B[I_BLOCK * O_TILE];
    float* w_shift_buf = w_buf_A, *w_compute_buf = w_buf_B; // Start with swapped roles

    auto shift_panel = [&](float* target_buf, int o, int i) {
        for (int i_off = 0; i_off < I_BLOCK; ++i_off) {
            for (int o_off = 0; o_off < O_TILE; ++o_off) {
                target_buf[i_off * O_TILE + o_off] = W[(size_t)(o + o_off) * i_dim + (i + i_off)];
            }
        }
    };
    
    // New loop order: o -> b -> i. This amortizes the shifting cost.
    for (int o_base = 0; o_base < o_dim; o_base += O_TILE) {
        for(int b=0; b<b_dim; ++b) {
            std::memset(Y + (size_t)b*o_dim + o_base, 0, O_TILE * sizeof(float));
        }

        // Prime the pipeline: synchronously shift the first block
        shift_panel(w_compute_buf, o_base, 0);

        for (int i_base = 0; i_base < i_dim; i_base += I_BLOCK) {
            std::future<void> shift_future;
            // STEP 1: LAUNCH ASYNC SHIFT FOR NEXT BLOCK
            if (i_base + I_BLOCK < i_dim) {
                shift_future = std::async(std::launch::async, shift_panel, w_shift_buf, o_base, i_base + I_BLOCK);
            }

            // STEP 2: COMPUTE ON CURRENT BLOCK (for the whole batch)
            for (int b = 0; b < b_dim; ++b) {
                const float* a_ptr = A + (size_t)b * i_dim + i_base;
                float* y_ptr = Y + (size_t)b * o_dim + o_base;
                __m256 vsum = _mm256_load_ps(y_ptr); // Load previous sum
                for (int i_off = 0; i_off < I_BLOCK; ++i_off) {
                    __m256 va_splat = _mm256_set1_ps(*(a_ptr + i_off));
                    __m256 vw_packed = _mm256_load_ps(w_compute_buf + i_off * O_TILE);
                    vsum = _mm256_fmadd_ps(va_splat, vw_packed, vsum);
                }
                _mm256_store_ps(y_ptr, vsum); // Store updated sum
            }
            
            // STEP 3: SYNCHRONIZE and SWAP
            if (shift_future.valid()) {
                shift_future.get();
            }
            std::swap(w_compute_buf, w_shift_buf);
        }
    }
}
#elif defined(__aarch64__)
// The ARM version would follow the same std::async logic
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
            printf("Benchmarking 'Enhanced Pipelined' Approach...\n");
            auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < Nbench; ++k) forward_pipelined_enhanced(A, W, Y, B, I, O);
            auto t2 = std::chrono::steady_clock::now();
            pipelined_ms = std::chrono::duration<double, std::milli>(t2-t1).count();
        }

        printf("\n--- Results (Scale %dx) ---\n", scale);
        const double flops = double(B) * I * O * 2;
        double gflops_packed = flops / (packed_ms / Nbench * 1e6);
        double gflops_pipelined = flops / (pipelined_ms / Nbench * 1e6);
        printf("Packed (Library Style)       : %.2f GFLOP/s\n", gflops_packed);
        printf("Enhanced Pipelined (V2 Idea) : %.2f GFLOP/s\n", gflops_pipelined);
        printf("\nSpeed-up of V2 Pipelined vs Packed: %.2fx\n", gflops_pipelined / gflops_packed);

        aligned_free(A);
        aligned_free(W);
        aligned_free(Y);
    }
#else
    puts("This benchmark requires NEON or AVX support.");
#endif
    return 0;
}