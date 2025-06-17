// =================================================================================
// Universal Compilation Instructions (GCC/Clang/MSVC)
// =================================================================================
// Linux (GCC) / macOS (Clang):
//   g++ -O3 -std=c++20 -march=native -o shift main.cpp
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
static constexpr int Nwarmup = 5;
static constexpr int Nbench  = 20;

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
            const float* a_ptr = A + b * i_dim;
            const float* w_pack_ptr = P.W_packed + size_t(o_base/O_TILE)*i_dim*O_TILE;
            for (int i = 0; i < i_dim; ++i) {
                __m256 va_splat = _mm256_set1_ps(*(a_ptr + i));
                __m256 vw_packed = _mm256_load_ps(w_pack_ptr + size_t(i)*O_TILE);
                vsum = _mm256_fmadd_ps(va_splat, vw_packed, vsum);
            }
            _mm256_store_ps(Y + b*o_dim + o_base, vsum);
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
            const float* a_ptr = A + b * i_dim;
            const float* w_pack_ptr = P.W_packed + size_t(o_base/O_TILE)*i_dim*O_TILE;
            for (int i = 0; i < i_dim; ++i) {
                float32x4_t va_splat = vld1q_dup_f32(a_ptr + i);
                float32x4_t vw_packed = vld1q_f32(w_pack_ptr + size_t(i)*O_TILE);
                vsum = vfmaq_f32(vsum, va_splat, vw_packed);
            }
            vst1q_f32(Y + b*o_dim + o_base, vsum);
        }
    }
}
#endif

// ===================================================================================
//  YOUR INNOVATION: ON-THE-FLY PIPELINED "SHIFTING"
// ===================================================================================
#if defined(__x86_64__) || defined(_M_X64)
TARGET_AVX2_FMA
__attribute__((noinline))
void forward_pipelined(const float* A, const float* W, float* Y, int b_dim, int i_dim, int o_dim) {
    constexpr int O_TILE = 8;
    constexpr int I_BLOCK = 256;
    alignas(64) float w_buf_A[I_BLOCK * O_TILE], w_buf_B[I_BLOCK * O_TILE];
    float* w_compute_buf = w_buf_A, *w_shift_buf = w_buf_B;

    auto shift_block = [&](const float* w_src, float* w_dst, int i_len) {
        for (int i = 0; i < i_len; ++i) {
            for(int o=0; o < O_TILE; ++o) w_dst[i * O_TILE + o] = w_src[o * i_dim + i];
        }
    };

    for (int b = 0; b < b_dim; ++b) {
        for (int o_base = 0; o_base < o_dim; o_base += O_TILE) {
            __m256 vsum = _mm256_setzero_ps();
            const float* a_ptr = A + b * i_dim;
            const float* w_base_ptr = W + o_base * i_dim;
            
            shift_block(w_base_ptr, w_compute_buf, I_BLOCK);
            
            for (int i_base = 0; i_base < i_dim; i_base += I_BLOCK) {
                if (i_base + I_BLOCK < i_dim) {
                    shift_block(w_base_ptr + i_base + I_BLOCK, w_shift_buf, I_BLOCK);
                }
                for (int i_off = 0; i_off < I_BLOCK; ++i_off) {
                    __m256 va_splat = _mm256_set1_ps(*(a_ptr + i_base + i_off));
                    __m256 vw_packed = _mm256_load_ps(w_compute_buf + i_off * O_TILE);
                    vsum = _mm256_fmadd_ps(va_splat, vw_packed, vsum);
                }
                std::swap(w_compute_buf, w_shift_buf);
            }
            _mm256_store_ps(Y + b*o_dim + o_base, vsum);
        }
    }
}
#elif defined(__aarch64__)
__attribute__((noinline))
void forward_pipelined(const float* A, const float* W, float* Y, int b_dim, int i_dim, int o_dim) {
    constexpr int O_TILE = 4;
    constexpr int I_BLOCK = 256;
    alignas(64) float w_buf_A[I_BLOCK * O_TILE], w_buf_B[I_BLOCK * O_TILE];
    float* w_compute_buf = w_buf_A, *w_shift_buf = w_buf_B;

    auto shift_block = [&](const float* w_src, float* w_dst, int i_len) {
        for (int i = 0; i < i_len; ++i) {
            for(int o=0; o < O_TILE; ++o) w_dst[i * O_TILE + o] = w_src[o * i_dim + i];
        }
    };

    for (int b = 0; b < b_dim; ++b) {
        for (int o_base = 0; o_base < o_dim; o_base += O_TILE) {
            float32x4_t vsum = vdupq_n_f32(0.f);
            const float* a_ptr = A + b * i_dim;
            const float* w_base_ptr = W + o_base * i_dim;
            
            shift_block(w_base_ptr, w_compute_buf, I_BLOCK);
            
            for (int i_base = 0; i_base < i_dim; i_base += I_BLOCK) {
                if (i_base + I_BLOCK < i_dim) {
                    shift_block(w_base_ptr + i_base + I_BLOCK, w_shift_buf, I_BLOCK);
                }
                for (int i_off = 0; i_off < I_BLOCK; ++i_off) {
                    float32x4_t va_splat = vld1q_dup_f32(a_ptr + i_base + i_off);
                    float32x4_t vw_packed = vld1q_f32(w_compute_buf + i_off * O_TILE);
                    vsum = vfmaq_f32(vsum, va_splat, vw_packed);
                }
                std::swap(w_compute_buf, w_shift_buf);
            }
            vst1q_f32(Y + b*o_dim + o_base, vsum);
        }
    }
}
#endif

// Main function to run the benchmark
int main() {
    printf("--- Running Matrix Multiplication Benchmark ---\n");
    printf("--- Scale: B=%d, I=%d, O=%d ---\n", B_base, I_base, O_base);
    const int B = B_base, I = I_base, O = O_base;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__)
    auto A = static_cast<float*>(aligned_malloc(size_t(B)*I*sizeof(float)));
    auto W = static_cast<float*>(aligned_malloc(size_t(I)*O*sizeof(float)));
    auto Y = static_cast<float*>(aligned_malloc(size_t(B)*O*sizeof(float)));
    
    std::mt19937 rng(42); std::uniform_real_distribution<float> d(-1,1);
    for(size_t i=0; i<size_t(B)*I; ++i) A[i] = d(rng);
    for(size_t i=0; i<size_t(I)*O; ++i) W[i] = d(rng);

    // --- Time the Packed (Library) Approach ---
    double packed_ms = 0;
    {
        printf("\nBenchmarking 'Packed' (Library-style) Approach...\n");
        PackedWeights P(W, I, O); // One-time packing cost is outside the timer
        auto t1 = std::chrono::steady_clock::now();
        for (int k = 0; k < Nbench; ++k) {
            forward_packed(A, P, Y, B, I, O);
        }
        auto t2 = std::chrono::steady_clock::now();
        packed_ms = std::chrono::duration<double, std::milli>(t2-t1).count();
    }

    // --- Time Your Pipelined Approach ---
    double pipelined_ms = 0;
    {
        printf("Benchmarking 'Pipelined' (Your Innovation) Approach...\n");
        auto t1 = std::chrono::steady_clock::now();
        for (int k = 0; k < Nbench; ++k) {
            forward_pipelined(A, W, Y, B, I, O);
        }
        auto t2 = std::chrono::steady_clock::now();
        pipelined_ms = std::chrono::duration<double, std::milli>(t2-t1).count();
    }

    // --- Report Results ---
    printf("\n--- Results ---\n");
    const double flops = double(B) * I * O * 2;
    double gflops_packed = flops / (packed_ms / Nbench * 1e6);
    double gflops_pipelined = flops / (pipelined_ms / Nbench * 1e6);
    printf("Packed (Library Style) : %.2f GFLOP/s\n", gflops_packed);
    printf("Pipelined (Your Idea)  : %.2f GFLOP/s\n", gflops_pipelined);
    printf("\nSpeed-up of Pipelined vs Packed: %.2fx\n", gflops_pipelined / gflops_packed);

    aligned_free(A);
    aligned_free(W);
    aligned_free(Y);
#else
    puts("This benchmark requires NEON or AVX support.");
#endif
    return 0;
}