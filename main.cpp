// clang++ -O3 -std=c++20 -march=armv8.4-a+simd -o shift main.cpp
// ./shift

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#if defined(__x86_64__) || defined(_M_X64) || defined(__AVX2__)
  #include <immintrin.h>   // AVX2 / AVX‑512 intrinsics
#elif defined(__aarch64__)
  #include <arm_neon.h>    // NEON intrinsics on Apple Silicon
#endif

// --- Base hyper-parameters (will be scaled in main) ----------------------
static constexpr int B_base = 256;
static constexpr int I_base = 4096;
static constexpr int O_base = 4096;
static constexpr int Nhots_max   = 256;
static constexpr int Hot_threshold = 8;
static constexpr int Nwarmup = 5;  // Reduced for faster testing at scale
static constexpr int Nbench  = 20; // Reduced for faster testing at scale

// Forward declarations
inline float dot_vec(const float* a, const float* w, int i_dim);

// --- helper: aligned malloc --------------------------------------------------
void* aligned_malloc(size_t bytes, size_t align = 64) {
    void* p; if(posix_memalign(&p, align, bytes)) return nullptr; return p;
}

// --- baseline forward pass ---------------------------------------------------
__attribute__((noinline))
void forward_baseline(const float* A, const float* W, float* Y, int b_dim, int i_dim, int o_dim) {
    for(int b=0;b<b_dim;++b){
        const float* a = A + b*i_dim;
        float* y = Y + b*o_dim;
        for (int o = 0; o < o_dim; ++o) {
            const float* wcol = W + size_t(o) * i_dim;
            float sum = dot_vec(a, wcol, i_dim);
            y[o] = sum;
        }
    }
}

// --- shifted forward pass ----------------------------------------------------
struct HotBuf {
    float* slab;
    int    size;
    std::vector<int> map;
    std::vector<int> acc;

    HotBuf(int o_dim, int i_dim) : map(o_dim, -1), acc(o_dim, 0) {
        slab = static_cast<float*>(aligned_malloc(size_t(Nhots_max)*i_dim*sizeof(float)));
        size = 0;
    }
    ~HotBuf(){ free(slab); }
};

inline void maybe_cache(int o, const float* W, HotBuf& H, int i_dim){
    if (H.map[o] != -1) return;
    if (++H.acc[o] < Hot_threshold) return;
    if (H.size >= Nhots_max) return;

    float* dst = H.slab + size_t(H.size)*i_dim;
    memcpy(dst, W + size_t(o)*i_dim, i_dim*sizeof(float));
    H.map[o] = H.size++;
}

__attribute__((noinline))
void forward_shifted(const float* A, const float* W, float* Y, HotBuf& H, int b_dim, int i_dim, int o_dim){
    for(int b=0;b<b_dim;++b){
        const float* a = A + b*i_dim;
        float* y = Y + b*o_dim;
        for(int o=0;o<o_dim;++o){
            const int hot_idx = H.map[o];
            const float* wcol = (hot_idx == -1)
                                ? (W + size_t(o)*i_dim)
                                : (H.slab + size_t(hot_idx)*i_dim);

            float sum = dot_vec(a, wcol, i_dim);
            y[o] = sum;

            maybe_cache(o, W, H, i_dim);
        }
    }
}

// --- Tiled forward pass (for data reuse) -------------------------------------
#if defined(__aarch64__)
__attribute__((noinline))
void forward_tiled(const float* A, const float* W, float* Y, int b_dim, int i_dim, int o_dim){
    constexpr int O_TILE = 4;
    assert(o_dim % O_TILE == 0 && "Output dimension must be a multiple of tile size for this simple implementation");

    for(int b = 0; b < b_dim; ++b){
        const float* a = A + b*i_dim;
        float* y = Y + b*o_dim;
        for(int o_base = 0; o_base < o_dim; o_base += O_TILE){
            float32x4_t vsum0=vdupq_n_f32(0.f), vsum1=vdupq_n_f32(0.f), vsum2=vdupq_n_f32(0.f), vsum3=vdupq_n_f32(0.f);

            const float* wcol0 = W + size_t(o_base + 0) * i_dim;
            const float* wcol1 = W + size_t(o_base + 1) * i_dim;
            const float* wcol2 = W + size_t(o_base + 2) * i_dim;
            const float* wcol3 = W + size_t(o_base + 3) * i_dim;

            for (int i = 0; i < i_dim; i += 4) {
                float32x4_t va = vld1q_f32(a + i);
                float32x4_t vw0 = vld1q_f32(wcol0 + i);
                float32x4_t vw1 = vld1q_f32(wcol1 + i);
                float32x4_t vw2 = vld1q_f32(wcol2 + i);
                float32x4_t vw3 = vld1q_f32(wcol3 + i);

                vsum0 = vfmaq_f32(vsum0, va, vw0);
                vsum1 = vfmaq_f32(vsum1, va, vw1);
                vsum2 = vfmaq_f32(vsum2, va, vw2);
                vsum3 = vfmaq_f32(vsum3, va, vw3);
            }
            y[o_base + 0] = vaddvq_f32(vsum0);
            y[o_base + 1] = vaddvq_f32(vsum1);
            y[o_base + 2] = vaddvq_f32(vsum2);
            y[o_base + 3] = vaddvq_f32(vsum3);
        }
    }
}
#endif
// NOTE: An equivalent x86 AVX implementation would be similar but tile by 8.

// -----------------------------------------------------------------------------
//  Architecture‑specific dot product
// -----------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64) || defined(__AVX2__)
inline float dot_vec(const float* a, const float* w, int i_dim) {
    __m256 vsum = _mm256_setzero_ps();
    for (int i = 0; i < i_dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        vsum = _mm256_fmadd_ps(va, vw, vsum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    return tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}
#elif defined(__aarch64__)
inline float dot_vec(const float* a, const float* w, int i_dim) {
    float32x4_t vsum = vdupq_n_f32(0.f);
    for (int i = 0; i < i_dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vw = vld1q_f32(w + i);
        vsum = vfmaq_f32(vsum, va, vw);
    }
    return vaddvq_f32(vsum);
}
#else
inline float dot_vec(const float* a, const float* w, int i_dim) {
    float sum = 0.f;
    for (int i = 0; i < i_dim; ++i) sum += a[i] * w[i];
    return sum;
}
#endif

// -----------------------------------------------------------------------------
//                     micro-driver / benchmark
// -----------------------------------------------------------------------------
enum class ForwardType { BASELINE, SHIFTED, TILED };

const char* to_string(ForwardType type) {
    switch (type) {
        case ForwardType::BASELINE: return "Baseline";
        case ForwardType::SHIFTED:  return "Shifted ";
        case ForwardType::TILED:    return "Tiled   ";
        default: return "Unknown";
    }
}

double run_and_time(ForwardType type, int b_dim, int i_dim, int o_dim) {
    size_t A_size = size_t(b_dim) * i_dim;
    size_t W_size = size_t(i_dim) * o_dim;
    size_t Y_size = size_t(b_dim) * o_dim;
    
    std::vector<float> A(A_size), W(W_size), Y(Y_size, 0.0f);
    volatile float checksum = 0.f;
    std::mt19937 rng(42); std::uniform_real_distribution<float> d(-1,1);
    std::generate(A.begin(), A.end(), [&]{ return d(rng); });
    std::generate(W.begin(), W.end(), [&]{ return d(rng); });

    HotBuf H(o_dim, i_dim);
    auto t0 = std::chrono::steady_clock::now();

    for(int k=0; k<Nwarmup; ++k) {
        switch(type) {
            case ForwardType::BASELINE: forward_baseline(A.data(), W.data(), Y.data(), b_dim, i_dim, o_dim); break;
            case ForwardType::SHIFTED:  forward_shifted(A.data(), W.data(), Y.data(), H, b_dim, i_dim, o_dim); break;
            #if defined(__aarch64__)
            case ForwardType::TILED:    forward_tiled(A.data(), W.data(), Y.data(), b_dim, i_dim, o_dim); break;
            #endif
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    for(int k=0; k<Nbench; ++k) {
        switch(type) {
            case ForwardType::BASELINE: forward_baseline(A.data(), W.data(), Y.data(), b_dim, i_dim, o_dim); break;
            case ForwardType::SHIFTED:  forward_shifted(A.data(), W.data(), Y.data(), H, b_dim, i_dim, o_dim); break;
            #if defined(__aarch64__)
            case ForwardType::TILED:    forward_tiled(A.data(), W.data(), Y.data(), b_dim, i_dim, o_dim); break;
            #endif
        }
    }
    checksum = checksum + Y[Y_size / 2];

    auto t2 = std::chrono::steady_clock::now();
    double warm_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    double run_ms  = std::chrono::duration<double, std::milli>(t2-t1).count();
    printf("%s: warm-up %.2f ms, bench %.2f ms", to_string(type), warm_ms, run_ms);
    (void)checksum;
    return run_ms;
}

int main(){
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__)
    for (int scale : {1, 3, 5}) {
        printf("\n--- Running at Scale: %dx (B=%d, I=%d, O=%d) ---\n",
               scale, B_base * scale, I_base * scale, O_base * scale);

        const int B = B_base * scale;
        const int I = I_base * scale;
        const int O = O_base * scale;

        double t_base  = run_and_time(ForwardType::BASELINE, B, I, O);
        double t_shift = run_and_time(ForwardType::SHIFTED, B, I, O);
        #if defined(__aarch64__)
        double t_tiled = run_and_time(ForwardType::TILED, B, I, O);
        #endif

        const double flops_per_matmul = double(B) * I * O * 2;
        double ns_base  = (t_base  * 1e6) / Nbench;
        double ns_shift = (t_shift * 1e6) / Nbench;
        
        double gflops_base  = flops_per_matmul * 1e-9 / (ns_base  * 1e-9);
        double gflops_shift = flops_per_matmul * 1e-9 / (ns_shift * 1e-9);

        printf("\n\n=== Results (Scale %dx) ===\n", scale);
        printf("Baseline: %.2f ns/op  (%.1f GFLOP/s)\n", ns_base,  gflops_base);
        printf("Shifted : %.2f ns/op  (%.1f GFLOP/s) | Speed-up vs Baseline: %.2fx\n", ns_shift, gflops_shift, ns_base / ns_shift);

        #if defined(__aarch64__)
        double ns_tiled = (t_tiled * 1e6) / Nbench;
        double gflops_tiled = flops_per_matmul * 1e-9 / (ns_tiled * 1e-9);
        printf("Tiled   : %.2f ns/op  (%.1f GFLOP/s) | Speed-up vs Baseline: %.2fx\n", ns_tiled, gflops_tiled, ns_base / ns_tiled);
        #endif
    }
#else
    puts("Compile with AVX2 or run on ARM-NEON.");
#endif
    return 0;
}