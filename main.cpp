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

// --- Base hyper-parameters (Realistic for a ~7B parameter model's FFN layer) ---
static constexpr int B_base = 32;
static constexpr int I_base = 4096;
static constexpr int O_base = 16384;
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
#endif

// ===================================================================================
//  V5 INNOVATION: OUTER-PRODUCT (A-STATIONARY) KERNEL
// ===================================================================================
#if defined(__x86_64__) || defined(_M_X64)
TARGET_AVX2_FMA
__attribute__((noinline))
void forward_v5_outer_product(const float* A, const float* W, float* Y, int b_dim, int i_dim, int o_dim) {
    constexpr int O_REG_TILE = 4; // Number of __m256 registers for C tile (4*8=32)
    constexpr int O_TILE = O_REG_TILE * 8;
    
    for (int b = 0; b < b_dim; ++b) {
        for (int o_base = 0; o_base < o_dim; o_base += O_TILE) {
            // Accumulators for a 1x32 tile of C
            __m256 c_regs[O_REG_TILE];
            for(int i=0; i<O_REG_TILE; ++i) c_regs[i] = _mm256_setzero_ps();

            for (int i = 0; i < i_dim; ++i) {
                const float* a_ptr = A + (size_t)b * i_dim;
                // Keep 'a' stationary in a register
                __m256 va_splat = _mm256_set1_ps(*(a_ptr + i));

                // "Spray" the contribution of a_scalar across the output tile
                for (int o_reg = 0; o_reg < O_REG_TILE; ++o_reg) {
                    const float* w_ptr = W + (size_t)(o_base + o_reg * 8) * i_dim + i;
                    __m256 vw = _mm256_loadu_ps(w_ptr); // Unaligned load is safer
                    c_regs[o_reg] = _mm256_fmadd_ps(va_splat, vw, c_regs[o_reg]);
                }
            }
            // Store the final results
            for(int i=0; i<O_REG_TILE; ++i) {
                _mm256_storeu_ps(Y + (size_t)b*o_dim + o_base + i*8, c_regs[i]);
            }
        }
    }
}
#endif

// ===================================================================================
// MAIN BENCHMARK FUNCTION
// ===================================================================================
int main() {
#if defined(__x86_64__) || defined(_M_X64)
    for (int scale : {1, 2, 3, 4, 5}) {
        printf("\n--- Running at Scale: %dx (B=%d, I=%d, O=%d) ---\n",
               scale, B_base * scale, I_base * scale, O_base * scale);

        const int B = B_base * scale;
        const int I = I_base * scale;
        const int O = O_base * scale;
        
        const size_t W_bytes = size_t(I) * O * sizeof(float);
        if (W_bytes > (8UL << 30)) {
            printf("SKIPPING: Weight matrix size (%.2f GB) exceeds safety limit.\n", (double)W_bytes / (1UL << 30));
            continue;
        }
        printf("    (Memory footprint: A=%.2f GB, W=%.2f GB, Y=%.2f GB)\n",
            (double)B*I*4/(1UL<<30), (double)I*O*4/(1UL<<30), (double)B*O*4/(1UL<<30));

        auto A = static_cast<float*>(aligned_malloc(size_t(B)*I*sizeof(float)));
        auto W = static_cast<float*>(aligned_malloc(W_bytes));
        auto Y = static_cast<float*>(aligned_malloc(size_t(B)*O*sizeof(float)));
        if (!A || !W || !Y) {
            printf("FATAL: Memory allocation failed.\n"); return 1;
        }
        
        std::mt19937 rng(42); std::uniform_real_distribution<float> d(-1,1);
        for(size_t i=0; i<size_t(B)*I; ++i) A[i] = d(rng);
        for(size_t i=0; i<W_bytes/sizeof(float); ++i) W[i] = d(rng);

        double packed_ms = 0;
        {
            printf("Benchmarking 'Packed' (Library-style) Approach...\n");
            PackedWeights P(W, I, O);
            auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < Nbench; ++k) forward_packed(A, P, Y, B, I, O);
            auto t2 = std::chrono::steady_clock::now();
            packed_ms = std::chrono::duration<double, std::milli>(t2-t1).count();
        }

        double outer_product_ms = 0;
        {
            printf("Benchmarking 'Outer-Product (V5)' Approach...\n");
            auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < Nbench; ++k) forward_v5_outer_product(A, W, Y, B, I, O);
            auto t2 = std::chrono::steady_clock::now();
            outer_product_ms = std::chrono::duration<double, std::milli>(t2-t1).count();
        }

        printf("\n--- Results (Scale %dx) ---\n", scale);
        const double flops = double(B) * I * O * 2;
        double gflops_packed = flops / (packed_ms / Nbench * 1e6);
        double gflops_outer = flops / (outer_product_ms / Nbench * 1e6);
        printf("Packed (Library Style)     : %.2f GFLOP/s\n", gflops_packed);
        printf("Outer-Product (V5 Idea)    : %.2f GFLOP/s\n", gflops_outer);
        printf("\nSpeed-up of V5 Outer-Product vs Packed: %.2fx\n", gflops_outer / gflops_packed);

        aligned_free(A);
        aligned_free(W);
        aligned_free(Y);
    }
#else
    puts("This benchmark is for x86 AVX2 only.");
#endif
    return 0;
}