#include "operators.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <float.h>

// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef static_assert
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif

#if defined _MSC_VER || defined(__MINGW32__)

#if !defined(__MINGW32__)
#include <Windows.h>
#else
// ref: https://github.com/ggerganov/whisper.cpp/issues/168
#include <windows.h>
#include <errno.h>
#endif

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;

static void atomic_store(atomic_int* ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int* ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int* ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_sub(atomic_int* ptr, LONG dec) {
    return atomic_fetch_add(ptr, -(dec));
}

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t* out, void* unused, thread_ret_t(*func)(void*), void* arg) {
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void* unused) {
    return (int) WaitForSingleObject(thread, INFINITE);
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else
#include <pthread.h>
#include <stdatomic.h>

typedef void* thread_ret_t;
#endif

#ifdef __HAIKU__
#define static_assert(cond, msg) _Static_assert(cond, msg)
#endif

/*#define DRAGON_PERF*/
#define DRAGON_DEBUG 0
#define DRAGON_GELU_FP16
#define DRAGON_SILU_FP16

#define DRAGON_SOFT_MAX_UNROLL 4
#define DRAGON_VEC_DOT_UNROLL  2

#ifdef DRAGON_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define DRAGON_SOFT_MAX_ACCELERATE
#endif

#if UINTPTR_MAX == 0xFFFFFFFF
    #define DRAGON_MEM_ALIGN 4
#else
    #define DRAGON_MEM_ALIGN 16
#endif

#define UNUSED(x) (void)(x)
#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)

#define DRAGON_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "DRAGON_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#ifdef DRAGON_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#elif DRAGON_USE_OPENBLAS
#include <cblas.h>
#endif

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// floating point type used to accumulate sums
typedef double dragon_float;

// 16-bit float
// on Arm, we use __fp16
// on x86, we use uint16_t
#ifdef __ARM_NEON

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>

#define DRAGON_COMPUTE_FP16_TO_FP32(x) (x)
#define DRAGON_COMPUTE_FP32_TO_FP16(x) (x)

#define DRAGON_FP16_TO_FP32(x) (x)
#define DRAGON_FP32_TO_FP16(x) (x)

#else

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#else
#ifdef __POWER9_VECTOR__
#include <altivec.h>
#undef bool
#define bool _Bool
#else
#include <immintrin.h>
#endif
#endif

#ifdef __F16C__

#define DRAGON_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define DRAGON_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)

#else

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
	union {
		float as_value;
		uint32_t as_bits;
	} fp32;
	fp32.as_value = f;
	return fp32.as_bits;
}

static inline float dragon_compute_fp16_to_fp32(dragon_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline dragon_fp16_t dragon_compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define DRAGON_COMPUTE_FP16_TO_FP32(x) dragon_compute_fp16_to_fp32(x)
#define DRAGON_COMPUTE_FP32_TO_FP16(x) dragon_compute_fp32_to_fp16(x)

#endif // __F16C__

#endif // __ARM_NEON

//
// global data
//

// precomputed gelu table for f16 (128 KB)
static dragon_fp16_t table_gelu_f16[1 << 16];

// precomputed silu table for f16 (128 KB)
static dragon_fp16_t table_silu_f16[1 << 16];

// precomputed exp table for f16 (128 KB)
static dragon_fp16_t table_exp_f16[1 << 16];

// precomputed f32 table for f16 (256 KB)
static float table_f32_f16[1 << 16];

// On ARM NEON, it's quicker to directly convert x -> x instead of calling into dragon_lookup_fp16_to_fp32,
// so we define DRAGON_FP16_TO_FP32 and DRAGON_FP32_TO_FP16 elsewhere for NEON.
#if !defined(DRAGON_FP16_TO_FP32) || !defined(DRAGON_FP32_TO_FP16)

inline static float dragon_lookup_fp16_to_fp32(dragon_fp16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return table_f32_f16[s];
}

#define DRAGON_FP16_TO_FP32(x) dragon_lookup_fp16_to_fp32(x)
#define DRAGON_FP32_TO_FP16(x) DRAGON_COMPUTE_FP32_TO_FP16(x)

#endif

// note: do not use these inside operators.c
// these are meant to be used via the operators.h API
float dragon_fp16_to_fp32(dragon_fp16_t x) {
    return DRAGON_FP16_TO_FP32(x);
}

dragon_fp16_t dragon_fp32_to_fp16(float x) {
    return DRAGON_FP32_TO_FP16(x);
}

//
// timing
//

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq;
void dragon_time_init(void) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    timer_freq = frequency.QuadPart;
}
int64_t dragon_time_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (t.QuadPart * 1000) / timer_freq;
}
int64_t dragon_time_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (t.QuadPart * 1000000) / timer_freq;
}
#else
void dragon_time_init(void) {}
int64_t dragon_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t dragon_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t dragon_cycles(void) {
    return clock();
}

int64_t dragon_cycles_per_ms(void) {
    return CLOCKS_PER_SEC/1000;
}

#ifdef DRAGON_PERF
#define dragon_perf_time_ms()       dragon_time_ms()
#define dragon_perf_time_us()       dragon_time_us()
#define dragon_perf_cycles()        dragon_cycles()
#define dragon_perf_cycles_per_ms() dragon_cycles_per_ms()
#else
#define dragon_perf_time_ms()       0
#define dragon_perf_time_us()       0
#define dragon_perf_cycles()        0
#define dragon_perf_cycles_per_ms() 0
#endif

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/sizeof(float);

//
// quantization
//

#define QK 32

#if __AVX2__
// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytesFromNibbles( const uint8_t* rsi )
{
    // Load 16 bytes from memory
    __m128i tmp = _mm_loadu_si128( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m256i bytes = _mm256_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    __m256i high = _mm256_andnot_si256( lowMask, bytes );
    __m256i low = _mm256_and_si256( lowMask, bytes );
    high = _mm256_slli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );
    return bytes;
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
}
#endif


// method 5
// blocks of QK elements
// represented with a single float (delta) and QK/2 8-bit ints (i.e QK 4-bit signed integer factors)
void quantize_row_q4_0(const float * restrict x, void * restrict y, int k) {
    assert(k % QK == 0);

    const int nb = k / QK;
    const size_t bs = sizeof(float) + QK/2;

    uint8_t * restrict pd = ((uint8_t *)y + 0*bs);
    uint8_t * restrict pb = ((uint8_t *)y + 0*bs + sizeof(float));

    uint8_t pp[QK/2];

#if __ARM_NEON
#if QK == 32
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = vld1q_f32(x + i*32 + 4*l);
        for (int l = 0; l < 8; l++) asrcv[l] = vabsq_f32(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2*l] = vmaxq_f32(asrcv[2*l], asrcv[2*l+1]);
        for (int l = 0; l < 2; l++) amaxv[4*l] = vmaxq_f32(amaxv[4*l], amaxv[4*l+2]);
        for (int l = 0; l < 1; l++) amaxv[8*l] = vmaxq_f32(amaxv[8*l], amaxv[8*l+4]);

        amax = MAX(
                MAX(vgetq_lane_f32(amaxv[0], 0), vgetq_lane_f32(amaxv[0], 1)),
                MAX(vgetq_lane_f32(amaxv[0], 2), vgetq_lane_f32(amaxv[0], 3)));

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0/d : 0.0;

        *(float *)pd = d;
        pd += bs;

        for (int l = 0; l < 8; l++) {
            const float32x4_t v  = vmulq_n_f32(srcv[l], id);
            const float32x4_t vf = vaddq_f32(v, vdupq_n_f32(8.5f));
            const int32x4_t   vi = vcvtq_s32_f32(vf);

            pp[2*l + 0] = vgetq_lane_s32(vi, 0) | (vgetq_lane_s32(vi, 1) << 4);
            pp[2*l + 1] = vgetq_lane_s32(vi, 2) | (vgetq_lane_s32(vi, 3) << 4);
        }

        memcpy(pb, pp, sizeof(pp));
        pb += bs;
    }
#else
#error "not implemented for QK"
#endif
#elif defined(__AVX2__)
#if QK == 32
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 7.0f;
        *(float *)pd = d;
        pd += bs;
        const float id = ( maxScalar != 0.0f ) ? 7.0f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        // Apply offset to translate the range from [ -7 .. +7 ] into [ +1 .. +15 ]
        const __m256i off = _mm256_set1_epi8( 8 );
        i0 = _mm256_add_epi8( i0, off );

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles( i0 );
        _mm_storeu_si128( ( __m128i* )pb, res );
        pb += bs;
    }
#else
#error "not implemented for QK"
#endif
#elif defined(__wasm_simd128__)
#if QK == 32
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = wasm_v128_load(x + i*32 + 4*l);
        for (int l = 0; l < 8; l++) asrcv[l] = wasm_f32x4_abs(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2*l] = wasm_f32x4_max(asrcv[2*l], asrcv[2*l+1]);
        for (int l = 0; l < 2; l++) amaxv[4*l] = wasm_f32x4_max(amaxv[4*l], amaxv[4*l+2]);
        for (int l = 0; l < 1; l++) amaxv[8*l] = wasm_f32x4_max(amaxv[8*l], amaxv[8*l+4]);

        amax = MAX(
                MAX(wasm_f32x4_extract_lane(amaxv[0], 0), wasm_f32x4_extract_lane(amaxv[0], 1)),
                MAX(wasm_f32x4_extract_lane(amaxv[0], 2), wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0/d : 0.0;

        *(float *)pd = d;
        pd += bs;

        for (int l = 0; l < 8; l++) {
            const v128_t v  = wasm_f32x4_mul(srcv[l], wasm_f32x4_splat(id));
            const v128_t vf = wasm_f32x4_add(v, wasm_f32x4_splat(8.5f));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(vf);

            pp[2*l + 0] = wasm_i32x4_extract_lane(vi, 0) | (wasm_i32x4_extract_lane(vi, 1) << 4);
            pp[2*l + 1] = wasm_i32x4_extract_lane(vi, 2) | (wasm_i32x4_extract_lane(vi, 3) << 4);
        }

        memcpy(pb, pp, sizeof(pp));
        pb += bs;
    }
#else
#error "not implemented for QK"
#endif
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int l = 0; l < QK; l++) {
            const float v = x[i*QK + l];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        *(float *)pd = d;
        pd += bs;

        for (int l = 0; l < QK; l += 2) {
            const float v0 = x[i*QK + l + 0]*id;
            const float v1 = x[i*QK + l + 1]*id;

            const uint8_t vi0 = ((int8_t) (round(v0))) + 8;
            const uint8_t vi1 = ((int8_t) (round(v1))) + 8;

            assert(vi0 >= 0 && vi0 < 16);
            assert(vi1 >= 0 && vi1 < 16);

            pp[l/2] = vi0 | (vi1 << 4);
        }

        memcpy(pb, pp, sizeof(pp));
        pb += bs;
    }
#endif
}

// method 4
// blocks of QK elements
// represented with 2 floats (min + delta) and QK/2 8-bit ints (i.e QK 4-bit unsigned integer factors)
void quantize_row_q4_1(const float * restrict x, void * restrict y, int k) {
    assert(k % QK == 0);

    const int nb = k / QK;
    const size_t bs = 2*sizeof(float) + QK/2;

    uint8_t * restrict pd = ((uint8_t *)y + 0*bs);
    uint8_t * restrict pm = ((uint8_t *)y + 0*bs +   sizeof(float));
    uint8_t * restrict pb = ((uint8_t *)y + 0*bs + 2*sizeof(float));

    uint8_t pp[QK/2];

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int l = 0; l < QK; l++) {
            const float v = x[i*QK + l];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        *(float *)pm = min;
        *(float *)pd = d;
        pm += bs;
        pd += bs;

        for (int l = 0; l < QK; l += 2) {
            const float v0 = (x[i*QK + l + 0] - min)*id;
            const float v1 = (x[i*QK + l + 1] - min)*id;

            const uint8_t vi0 = round(v0);
            const uint8_t vi1 = round(v1);

            assert(vi0 >= 0 && vi0 < 16);
            assert(vi1 >= 0 && vi1 < 16);

            pp[l/2] = vi0 | (vi1 << 4);
        }

        memcpy(pb, pp, sizeof(pp));
        pb += bs;
    }
}

// TODO: vectorize
void dequantize_row_q4_0(const void * restrict x, float * restrict y, int k) {
    assert(k % QK == 0);

    const int nb = k / QK;
    const size_t bs = sizeof(float) + QK/2;

    const uint8_t * restrict pd = ((const uint8_t *)x + 0*bs);
    const uint8_t * restrict pb = ((const uint8_t *)x + 0*bs + sizeof(float));

    // scalar
    for (int i = 0; i < nb; i++) {
        const float d = *(const float *) (pd + i*bs);

        const uint8_t * restrict pp = pb + i*bs;

        for (int l = 0; l < QK; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8)*d;
            const float v1 = (vi1 - 8)*d;

            //printf("d = %f, vi = %d, vi0 = %d, vi1 = %d, v0 = %f, v1 = %f\n", d, vi, vi0, vi1, v0, v1);

            y[i*QK + l + 0] = v0;
            y[i*QK + l + 1] = v1;

            assert(!isnan(y[i*QK + l + 0]));
            assert(!isnan(y[i*QK + l + 1]));
        }
    }
}

void dequantize_row_q4_1(const void * restrict x, float * restrict y, int k) {
    assert(k % QK == 0);

    const int nb = k / QK;
    const size_t bs = 2*sizeof(float) + QK/2;

    const uint8_t * restrict pd = ((const uint8_t *)x + 0*bs);
    const uint8_t * restrict pm = ((const uint8_t *)x + 0*bs + sizeof(float));
    const uint8_t * restrict pb = ((const uint8_t *)x + 0*bs + 2*sizeof(float));

    for (int i = 0; i < nb; i++) {
        const float d = *(const float *) (pd + i*bs);
        const float m = *(const float *) (pm + i*bs);

        const uint8_t * restrict pp = pb + i*bs;

        for (int l = 0; l < QK; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = vi0*d + m;
            const float v1 = vi1*d + m;

            y[i*QK + l + 0] = v0;
            y[i*QK + l + 1] = v1;

            assert(!isnan(y[i*QK + l + 0]));
            assert(!isnan(y[i*QK + l + 1]));
        }
    }
}

//
// simd mappings
//

// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// DRAGON_F32_STEP / DRAGON_F16_STEP
//   number of elements to process in a single step
//
// DRAGON_F32_EPR / DRAGON_F16_EPR
//   number of elements to fit in a single register
//

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)

#define DRAGON_SIMD

// F32 NEON

#define DRAGON_F32_STEP 16
#define DRAGON_F32_EPR  4

#define DRAGON_F32x4              float32x4_t
#define DRAGON_F32x4_ZERO         vdupq_n_f32(0.0f)
#define DRAGON_F32x4_SET1(x)      vdupq_n_f32(x)
#define DRAGON_F32x4_LOAD         vld1q_f32
#define DRAGON_F32x4_STORE        vst1q_f32
#define DRAGON_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define DRAGON_F32x4_ADD          vaddq_f32
#define DRAGON_F32x4_MUL          vmulq_f32
#if defined(__ARM_FEATURE_QRDMX)
    #define DRAGON_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#else
    #define DRAGON_F32x4_REDUCE_ONE(x) \
    (vgetq_lane_f32(x, 0) +          \
     vgetq_lane_f32(x, 1) +          \
     vgetq_lane_f32(x, 2) +          \
     vgetq_lane_f32(x, 3))
#endif
#define DRAGON_F32x4_REDUCE(res, x)              \
{                                              \
    for (int i = 0; i < DRAGON_F32_ARR/2; ++i) { \
        x[2*i] = vaddq_f32(x[2*i], x[2*i+1]);  \
    }                                          \
    for (int i = 0; i < DRAGON_F32_ARR/4; ++i) { \
        x[4*i] = vaddq_f32(x[4*i], x[4*i+2]);  \
    }                                          \
    for (int i = 0; i < DRAGON_F32_ARR/8; ++i) { \
        x[8*i] = vaddq_f32(x[8*i], x[8*i+4]);  \
    }                                          \
    res = DRAGON_F32x4_REDUCE_ONE(x[0]);         \
}

#define DRAGON_F32_VEC        DRAGON_F32x4
#define DRAGON_F32_VEC_ZERO   DRAGON_F32x4_ZERO
#define DRAGON_F32_VEC_SET1   DRAGON_F32x4_SET1
#define DRAGON_F32_VEC_LOAD   DRAGON_F32x4_LOAD
#define DRAGON_F32_VEC_STORE  DRAGON_F32x4_STORE
#define DRAGON_F32_VEC_FMA    DRAGON_F32x4_FMA
#define DRAGON_F32_VEC_ADD    DRAGON_F32x4_ADD
#define DRAGON_F32_VEC_MUL    DRAGON_F32x4_MUL
#define DRAGON_F32_VEC_REDUCE DRAGON_F32x4_REDUCE

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define DRAGON_F16_STEP 32
    #define DRAGON_F16_EPR  8

    #define DRAGON_F16x8              float16x8_t
    #define DRAGON_F16x8_ZERO         vdupq_n_f16(0.0f)
    #define DRAGON_F16x8_SET1(x)      vdupq_n_f16(x)
    #define DRAGON_F16x8_LOAD         vld1q_f16
    #define DRAGON_F16x8_STORE        vst1q_f16
    #define DRAGON_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #define DRAGON_F16x8_ADD          vaddq_f16
    #define DRAGON_F16x8_MUL          vmulq_f16
    #define DRAGON_F16x8_REDUCE(res, x)                             \
    {                                                             \
        for (int i = 0; i < DRAGON_F16_ARR/2; ++i) {                \
            x[2*i] = vaddq_f16(x[2*i], x[2*i+1]);                 \
        }                                                         \
        for (int i = 0; i < DRAGON_F16_ARR/4; ++i) {                \
            x[4*i] = vaddq_f16(x[4*i], x[4*i+2]);                 \
        }                                                         \
        for (int i = 0; i < DRAGON_F16_ARR/8; ++i) {                \
            x[8*i] = vaddq_f16(x[8*i], x[8*i+4]);                 \
        }                                                         \
        const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 (x[0])); \
        const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(x[0])); \
        res = vaddvq_f32(vaddq_f32(t0, t1));                      \
    }

    #define DRAGON_F16_VEC                DRAGON_F16x8
    #define DRAGON_F16_VEC_ZERO           DRAGON_F16x8_ZERO
    #define DRAGON_F16_VEC_SET1           DRAGON_F16x8_SET1
    #define DRAGON_F16_VEC_LOAD(p, i)     DRAGON_F16x8_LOAD(p)
    #define DRAGON_F16_VEC_STORE(p, r, i) DRAGON_F16x8_STORE(p, r[i])
    #define DRAGON_F16_VEC_FMA            DRAGON_F16x8_FMA
    #define DRAGON_F16_VEC_ADD            DRAGON_F16x8_ADD
    #define DRAGON_F16_VEC_MUL            DRAGON_F16x8_MUL
    #define DRAGON_F16_VEC_REDUCE         DRAGON_F16x8_REDUCE
#else
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #define DRAGON_F16_STEP 16
    #define DRAGON_F16_EPR  4

    #define DRAGON_F32Cx4              float32x4_t
    #define DRAGON_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #define DRAGON_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #define DRAGON_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16(x))
    #define DRAGON_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #define DRAGON_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #define DRAGON_F32Cx4_ADD          vaddq_f32
    #define DRAGON_F32Cx4_MUL          vmulq_f32
    #define DRAGON_F32Cx4_REDUCE       DRAGON_F32x4_REDUCE

    #define DRAGON_F16_VEC                DRAGON_F32Cx4
    #define DRAGON_F16_VEC_ZERO           DRAGON_F32Cx4_ZERO
    #define DRAGON_F16_VEC_SET1           DRAGON_F32Cx4_SET1
    #define DRAGON_F16_VEC_LOAD(p, i)     DRAGON_F32Cx4_LOAD(p)
    #define DRAGON_F16_VEC_STORE(p, r, i) DRAGON_F32Cx4_STORE(p, r[i])
    #define DRAGON_F16_VEC_FMA            DRAGON_F32Cx4_FMA
    #define DRAGON_F16_VEC_ADD            DRAGON_F32Cx4_ADD
    #define DRAGON_F16_VEC_MUL            DRAGON_F32Cx4_MUL
    #define DRAGON_F16_VEC_REDUCE         DRAGON_F32Cx4_REDUCE
#endif

#elif defined(__AVX__)

#define DRAGON_SIMD

// F32 AVX

#define DRAGON_F32_STEP 32
#define DRAGON_F32_EPR  8

#define DRAGON_F32x8         __m256
#define DRAGON_F32x8_ZERO    _mm256_setzero_ps()
#define DRAGON_F32x8_SET1(x) _mm256_set1_ps(x)
#define DRAGON_F32x8_LOAD    _mm256_loadu_ps
#define DRAGON_F32x8_STORE   _mm256_storeu_ps
#if defined(__FMA__)
    #define DRAGON_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
    #define DRAGON_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define DRAGON_F32x8_ADD     _mm256_add_ps
#define DRAGON_F32x8_MUL     _mm256_mul_ps
#define DRAGON_F32x8_REDUCE(res, x)                                 \
{                                                                 \
    for (int i = 0; i < DRAGON_F32_ARR/2; ++i) {                    \
        x[2*i] = _mm256_add_ps(x[2*i], x[2*i+1]);                 \
    }                                                             \
    for (int i = 0; i < DRAGON_F32_ARR/4; ++i) {                    \
        x[4*i] = _mm256_add_ps(x[4*i], x[4*i+2]);                 \
    }                                                             \
    for (int i = 0; i < DRAGON_F32_ARR/8; ++i) {                    \
        x[8*i] = _mm256_add_ps(x[8*i], x[8*i+4]);                 \
    }                                                             \
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));                     \
}
// TODO: is this optimal ?

#define DRAGON_F32_VEC        DRAGON_F32x8
#define DRAGON_F32_VEC_ZERO   DRAGON_F32x8_ZERO
#define DRAGON_F32_VEC_SET1   DRAGON_F32x8_SET1
#define DRAGON_F32_VEC_LOAD   DRAGON_F32x8_LOAD
#define DRAGON_F32_VEC_STORE  DRAGON_F32x8_STORE
#define DRAGON_F32_VEC_FMA    DRAGON_F32x8_FMA
#define DRAGON_F32_VEC_ADD    DRAGON_F32x8_ADD
#define DRAGON_F32_VEC_MUL    DRAGON_F32x8_MUL
#define DRAGON_F32_VEC_REDUCE DRAGON_F32x8_REDUCE

// F16 AVX

#define DRAGON_F16_STEP 32
#define DRAGON_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead
// we take advantage of the _mm256_cvt intrinsics to convert F16 <-> F32

#define DRAGON_F32Cx8             __m256
#define DRAGON_F32Cx8_ZERO        _mm256_setzero_ps()
#define DRAGON_F32Cx8_SET1(x)     _mm256_set1_ps(x)
#define DRAGON_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x)))
#define DRAGON_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#define DRAGON_F32Cx8_FMA         DRAGON_F32x8_FMA
#define DRAGON_F32Cx8_ADD         _mm256_add_ps
#define DRAGON_F32Cx8_MUL         _mm256_mul_ps
#define DRAGON_F32Cx8_REDUCE      DRAGON_F32x8_REDUCE

#define DRAGON_F16_VEC                DRAGON_F32Cx8
#define DRAGON_F16_VEC_ZERO           DRAGON_F32Cx8_ZERO
#define DRAGON_F16_VEC_SET1           DRAGON_F32Cx8_SET1
#define DRAGON_F16_VEC_LOAD(p, i)     DRAGON_F32Cx8_LOAD(p)
#define DRAGON_F16_VEC_STORE(p, r, i) DRAGON_F32Cx8_STORE(p, r[i])
#define DRAGON_F16_VEC_FMA            DRAGON_F32Cx8_FMA
#define DRAGON_F16_VEC_ADD            DRAGON_F32Cx8_ADD
#define DRAGON_F16_VEC_MUL            DRAGON_F32Cx8_MUL
#define DRAGON_F16_VEC_REDUCE         DRAGON_F32Cx8_REDUCE

#elif defined(__POWER9_VECTOR__)

#define DRAGON_SIMD

// F32 POWER9

#define DRAGON_F32_STEP 32
#define DRAGON_F32_EPR  4

#define DRAGON_F32x4              vector float
#define DRAGON_F32x4_ZERO         0.0f
#define DRAGON_F32x4_SET1         vec_splats
#define DRAGON_F32x4_LOAD(p)      vec_xl(0, p)
#define DRAGON_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#define DRAGON_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#define DRAGON_F32x4_ADD          vec_add
#define DRAGON_F32x4_MUL          vec_mul
#define DRAGON_F32x4_REDUCE(res, x)              \
{                                              \
    for (int i = 0; i < DRAGON_F32_ARR/2; ++i) { \
        x[2*i] = vec_add(x[2*i], x[2*i+1]);    \
    }                                          \
    for (int i = 0; i < DRAGON_F32_ARR/4; ++i) { \
        x[4*i] = vec_add(x[4*i], x[4*i+2]);    \
    }                                          \
    for (int i = 0; i < DRAGON_F32_ARR/8; ++i) { \
        x[8*i] = vec_add(x[8*i], x[8*i+4]);    \
    }                                          \
    res = vec_extract(x[0], 0) +               \
          vec_extract(x[0], 1) +               \
          vec_extract(x[0], 2) +               \
          vec_extract(x[0], 3);                \
}

#define DRAGON_F32_VEC        DRAGON_F32x4
#define DRAGON_F32_VEC_ZERO   DRAGON_F32x4_ZERO
#define DRAGON_F32_VEC_SET1   DRAGON_F32x4_SET1
#define DRAGON_F32_VEC_LOAD   DRAGON_F32x4_LOAD
#define DRAGON_F32_VEC_STORE  DRAGON_F32x4_STORE
#define DRAGON_F32_VEC_FMA    DRAGON_F32x4_FMA
#define DRAGON_F32_VEC_ADD    DRAGON_F32x4_ADD
#define DRAGON_F32_VEC_MUL    DRAGON_F32x4_MUL
#define DRAGON_F32_VEC_REDUCE DRAGON_F32x4_REDUCE

// F16 POWER9
#define DRAGON_F16_STEP       DRAGON_F32_STEP
#define DRAGON_F16_EPR        DRAGON_F32_EPR
#define DRAGON_F16_VEC        DRAGON_F32x4
#define DRAGON_F16_VEC_ZERO   DRAGON_F32x4_ZERO
#define DRAGON_F16_VEC_SET1   DRAGON_F32x4_SET1
#define DRAGON_F16_VEC_FMA    DRAGON_F32x4_FMA
#define DRAGON_F16_VEC_REDUCE DRAGON_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#define DRAGON_F16_VEC_LOAD(p, i) (i & 0x1) ?                   \
  vec_extract_fp32_from_shorth(vec_xl(0, p - DRAGON_F16_EPR)) : \
  vec_extract_fp32_from_shortl(vec_xl(0, p))
#define DRAGON_ENDIAN_BYTE(i) ((unsigned char *)&(uint16_t){1})[i]
#define DRAGON_F16_VEC_STORE(p, r, i)                             \
  if (i & 0x1)                                                  \
    vec_xst(vec_pack_to_short_fp32(r[i - DRAGON_ENDIAN_BYTE(1)],  \
                                   r[i - DRAGON_ENDIAN_BYTE(0)]), \
            0, p - DRAGON_F16_EPR)

#elif defined(__wasm_simd128__)

#define DRAGON_SIMD

// F32 WASM

#define DRAGON_F32_STEP 16
#define DRAGON_F32_EPR  4

#define DRAGON_F32x4              v128_t
#define DRAGON_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#define DRAGON_F32x4_SET1(x)      wasm_f32x4_splat(x)
#define DRAGON_F32x4_LOAD         wasm_v128_load
#define DRAGON_F32x4_STORE        wasm_v128_store
#define DRAGON_F32x4_FMA(a, b, c) wasm_f32x4_add(wasm_f32x4_mul(b, c), a)
#define DRAGON_F32x4_ADD          wasm_f32x4_add
#define DRAGON_F32x4_MUL          wasm_f32x4_mul
#define DRAGON_F32x4_REDUCE(res, x)                  \
{                                                  \
    for (int i = 0; i < DRAGON_F32_ARR/2; ++i) {     \
        x[2*i] = wasm_f32x4_add(x[2*i], x[2*i+1]); \
    }                                              \
    for (int i = 0; i < DRAGON_F32_ARR/4; ++i) {     \
        x[4*i] = wasm_f32x4_add(x[4*i], x[4*i+2]); \
    }                                              \
    for (int i = 0; i < DRAGON_F32_ARR/8; ++i) {     \
        x[8*i] = wasm_f32x4_add(x[8*i], x[8*i+4]); \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#define DRAGON_F32_VEC        DRAGON_F32x4
#define DRAGON_F32_VEC_ZERO   DRAGON_F32x4_ZERO
#define DRAGON_F32_VEC_SET1   DRAGON_F32x4_SET1
#define DRAGON_F32_VEC_LOAD   DRAGON_F32x4_LOAD
#define DRAGON_F32_VEC_STORE  DRAGON_F32x4_STORE
#define DRAGON_F32_VEC_FMA    DRAGON_F32x4_FMA
#define DRAGON_F32_VEC_ADD    DRAGON_F32x4_ADD
#define DRAGON_F32_VEC_MUL    DRAGON_F32x4_MUL
#define DRAGON_F32_VEC_REDUCE DRAGON_F32x4_REDUCE

// F16 WASM

#define DRAGON_F16_STEP 16
#define DRAGON_F16_EPR  4

inline static v128_t __wasm_f16x4_load(const dragon_fp16_t * p) {
    float tmp[4];

    tmp[0] = DRAGON_FP16_TO_FP32(p[0]);
    tmp[1] = DRAGON_FP16_TO_FP32(p[1]);
    tmp[2] = DRAGON_FP16_TO_FP32(p[2]);
    tmp[3] = DRAGON_FP16_TO_FP32(p[3]);

    return wasm_v128_load(tmp);
}

inline static void __wasm_f16x4_store(dragon_fp16_t * p, v128_t x) {
    float tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = DRAGON_FP32_TO_FP16(tmp[0]);
    p[1] = DRAGON_FP32_TO_FP16(tmp[1]);
    p[2] = DRAGON_FP32_TO_FP16(tmp[2]);
    p[3] = DRAGON_FP32_TO_FP16(tmp[3]);
}

#define DRAGON_F16x4             v128_t
#define DRAGON_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#define DRAGON_F16x4_SET1(x)     wasm_f32x4_splat(x)
#define DRAGON_F16x4_LOAD(x)     __wasm_f16x4_load(x)
#define DRAGON_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#define DRAGON_F16x4_FMA         DRAGON_F32x4_FMA
#define DRAGON_F16x4_ADD         wasm_f32x4_add
#define DRAGON_F16x4_MUL         wasm_f32x4_mul
#define DRAGON_F16x4_REDUCE(res, x)                  \
{                                                  \
    for (int i = 0; i < DRAGON_F16_ARR/2; ++i) {     \
        x[2*i] = wasm_f32x4_add(x[2*i], x[2*i+1]); \
    }                                              \
    for (int i = 0; i < DRAGON_F16_ARR/4; ++i) {     \
        x[4*i] = wasm_f32x4_add(x[4*i], x[4*i+2]); \
    }                                              \
    for (int i = 0; i < DRAGON_F16_ARR/8; ++i) {     \
        x[8*i] = wasm_f32x4_add(x[8*i], x[8*i+4]); \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#define DRAGON_F16_VEC                DRAGON_F16x4
#define DRAGON_F16_VEC_ZERO           DRAGON_F16x4_ZERO
#define DRAGON_F16_VEC_SET1           DRAGON_F16x4_SET1
#define DRAGON_F16_VEC_LOAD(p, i)     DRAGON_F16x4_LOAD(p)
#define DRAGON_F16_VEC_STORE(p, r, i) DRAGON_F16x4_STORE(p, r[i])
#define DRAGON_F16_VEC_FMA            DRAGON_F16x4_FMA
#define DRAGON_F16_VEC_ADD            DRAGON_F16x4_ADD
#define DRAGON_F16_VEC_MUL            DRAGON_F16x4_MUL
#define DRAGON_F16_VEC_REDUCE         DRAGON_F16x4_REDUCE

#elif defined(__SSE3__)

#define DRAGON_SIMD

// F32 SSE

#define DRAGON_F32_STEP 32
#define DRAGON_F32_EPR  4

#define DRAGON_F32x4         __m128
#define DRAGON_F32x4_ZERO    _mm_setzero_ps()
#define DRAGON_F32x4_SET1(x) _mm_set1_ps(x)
#define DRAGON_F32x4_LOAD    _mm_loadu_ps
#define DRAGON_F32x4_STORE   _mm_storeu_ps
#if defined(__FMA__)
    // TODO: Does this work?
    #define DRAGON_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#else
    #define DRAGON_F32x4_FMA(a, b, c) _mm_add_ps(_mm_mul_ps(b, c), a)
#endif
#define DRAGON_F32x4_ADD     _mm_add_ps
#define DRAGON_F32x4_MUL     _mm_mul_ps
#define DRAGON_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    for (int i = 0; i < DRAGON_F32_ARR/2; ++i) {                    \
        x[2*i] = _mm_add_ps(x[2*i], x[2*i+1]);                    \
    }                                                             \
    for (int i = 0; i < DRAGON_F32_ARR/4; ++i) {                    \
        x[4*i] = _mm_add_ps(x[4*i], x[4*i+2]);                    \
    }                                                             \
    for (int i = 0; i < DRAGON_F32_ARR/8; ++i) {                    \
        x[8*i] = _mm_add_ps(x[8*i], x[8*i+4]);                    \
    }                                                             \
    const __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = _mm_cvtss_f32(_mm_hadd_ps(t0, t0));                     \
}
// TODO: is this optimal ?

#define DRAGON_F32_VEC        DRAGON_F32x4
#define DRAGON_F32_VEC_ZERO   DRAGON_F32x4_ZERO
#define DRAGON_F32_VEC_SET1   DRAGON_F32x4_SET1
#define DRAGON_F32_VEC_LOAD   DRAGON_F32x4_LOAD
#define DRAGON_F32_VEC_STORE  DRAGON_F32x4_STORE
#define DRAGON_F32_VEC_FMA    DRAGON_F32x4_FMA
#define DRAGON_F32_VEC_ADD    DRAGON_F32x4_ADD
#define DRAGON_F32_VEC_MUL    DRAGON_F32x4_MUL
#define DRAGON_F32_VEC_REDUCE DRAGON_F32x4_REDUCE

// F16 SSE

#define DRAGON_F16_STEP 32
#define DRAGON_F16_EPR  4

static inline __m128 __sse_f16x4_load(dragon_fp16_t *x) {
    float tmp[4];

    tmp[0] = DRAGON_FP16_TO_FP32(x[0]);
    tmp[1] = DRAGON_FP16_TO_FP32(x[1]);
    tmp[2] = DRAGON_FP16_TO_FP32(x[2]);
    tmp[3] = DRAGON_FP16_TO_FP32(x[3]);

    return _mm_loadu_ps(tmp);
}

static inline void __sse_f16x4_store(dragon_fp16_t *x, __m128 y) {
    float arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = DRAGON_FP32_TO_FP16(arr[0]);
    x[1] = DRAGON_FP32_TO_FP16(arr[1]);
    x[2] = DRAGON_FP32_TO_FP16(arr[2]);
    x[3] = DRAGON_FP32_TO_FP16(arr[3]);
}

#define DRAGON_F32Cx4             __m128
#define DRAGON_F32Cx4_ZERO        _mm_setzero_ps()
#define DRAGON_F32Cx4_SET1(x)     _mm_set1_ps(x)
#define DRAGON_F32Cx4_LOAD(x)     __sse_f16x4_load(x)
#define DRAGON_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#define DRAGON_F32Cx4_FMA         DRAGON_F32x4_FMA
#define DRAGON_F32Cx4_ADD         _mm_add_ps
#define DRAGON_F32Cx4_MUL         _mm_mul_ps
#define DRAGON_F32Cx4_REDUCE      DRAGON_F32x4_REDUCE

#define DRAGON_F16_VEC                 DRAGON_F32Cx4
#define DRAGON_F16_VEC_ZERO            DRAGON_F32Cx4_ZERO
#define DRAGON_F16_VEC_SET1            DRAGON_F32Cx4_SET1
#define DRAGON_F16_VEC_LOAD(p, i)      DRAGON_F32Cx4_LOAD(p)
#define DRAGON_F16_VEC_STORE(p, r, i)  DRAGON_F32Cx4_STORE(p, r[i])
#define DRAGON_F16_VEC_FMA             DRAGON_F32Cx4_FMA
#define DRAGON_F16_VEC_ADD             DRAGON_F32Cx4_ADD
#define DRAGON_F16_VEC_MUL             DRAGON_F32Cx4_MUL
#define DRAGON_F16_VEC_REDUCE          DRAGON_F32Cx4_REDUCE

#endif

// DRAGON_F32_ARR / DRAGON_F16_ARR
//   number of registers to use per step
#ifdef DRAGON_SIMD
#define DRAGON_F32_ARR (DRAGON_F32_STEP/DRAGON_F32_EPR)
#define DRAGON_F16_ARR (DRAGON_F16_STEP/DRAGON_F16_EPR)
#endif

//
// fundamental operations
//

inline static void dragon_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void dragon_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void dragon_vec_set_i32(const int n, int32_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void dragon_vec_set_f16(const int n, dragon_fp16_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void dragon_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
inline static void dragon_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void dragon_vec_acc1_f32(const int n, float * y, const float   v)                  { for (int i = 0; i < n; ++i) y[i] += v;           }
inline static void dragon_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
inline static void dragon_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void dragon_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void dragon_vec_neg_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = -x[i];       }
inline static void dragon_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
inline static void dragon_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }

inline static void dragon_vec_dot_f32(const int n, float * restrict s, const float * restrict x, const float * restrict y) {
    dragon_float sumf = 0.0;

#ifdef DRAGON_SIMD
    const int np = (n & ~(DRAGON_F32_STEP - 1));

    DRAGON_F32_VEC sum[DRAGON_F32_ARR] = { DRAGON_F32_VEC_ZERO };

    DRAGON_F32_VEC ax[DRAGON_F32_ARR];
    DRAGON_F32_VEC ay[DRAGON_F32_ARR];

    for (int i = 0; i < np; i += DRAGON_F32_STEP) {
        for (int j = 0; j < DRAGON_F32_ARR; j++) {
            ax[j] = DRAGON_F32_VEC_LOAD(x + i + j*DRAGON_F32_EPR);
            ay[j] = DRAGON_F32_VEC_LOAD(y + i + j*DRAGON_F32_EPR);

            sum[j] = DRAGON_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    DRAGON_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i]*y[i];
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        sumf += x[i]*y[i];
    }
#endif

    *s = sumf;
}

inline static void dragon_vec_dot_f16(const int n, float * restrict s, dragon_fp16_t * restrict x, dragon_fp16_t * restrict y) {
    dragon_float sumf = 0.0;

#if defined(DRAGON_SIMD)
    const int np = (n & ~(DRAGON_F16_STEP - 1));

    DRAGON_F16_VEC sum[DRAGON_F16_ARR] = { DRAGON_F16_VEC_ZERO };

    DRAGON_F16_VEC ax[DRAGON_F16_ARR];
    DRAGON_F16_VEC ay[DRAGON_F16_ARR];

    for (int i = 0; i < np; i += DRAGON_F16_STEP) {
        for (int j = 0; j < DRAGON_F16_ARR; j++) {
            ax[j] = DRAGON_F16_VEC_LOAD(x + i + j*DRAGON_F16_EPR, j);
            ay[j] = DRAGON_F16_VEC_LOAD(y + i + j*DRAGON_F16_EPR, j);

            sum[j] = DRAGON_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    DRAGON_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += DRAGON_FP16_TO_FP32(x[i])*DRAGON_FP16_TO_FP32(y[i]);
    }
#else
    for (int i = 0; i < n; ++i) {
        sumf += DRAGON_FP16_TO_FP32(x[i])*DRAGON_FP16_TO_FP32(y[i]);
    }
#endif

    *s = sumf;
}

inline static void dragon_vec_dot_q4_0(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const int nb = n / QK;

    assert(n % QK == 0);
    assert(nb % 2 == 0);

    const size_t bs = sizeof(float) + QK/2;

    const uint8_t * restrict pd0 = ((const uint8_t *)x + 0*bs);
    const uint8_t * restrict pd1 = ((const uint8_t *)y + 0*bs);

    const uint8_t * restrict pb0 = ((const uint8_t *)x + 0*bs + sizeof(float));
    const uint8_t * restrict pb1 = ((const uint8_t *)y + 0*bs + sizeof(float));

    float sumf = 0.0;

#ifdef __ARM_NEON
#if QK == 32
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int i = 0; i < nb; i += 2) {
        const float d0_0 = *(const float *) (pd0 + i*bs);
        const float d1_0 = *(const float *) (pd1 + i*bs);
        const float d0_1 = *(const float *) (pd0 + (i + 1)*bs);
        const float d1_1 = *(const float *) (pd1 + (i + 1)*bs);

        //printf("d0_0: %f, d1_0: %f, d0_1: %f, d1_1: %f\n", d0_0, d1_0, d0_1, d1_1);

        const uint8_t * restrict p0 = pb0 + i*bs;
        const uint8_t * restrict p1 = pb1 + i*bs;

        const uint8x16_t m4b = vdupq_n_u8(0xf);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(p0);
        const uint8x16_t v1_0 = vld1q_u8(p1);
        const uint8x16_t v0_1 = vld1q_u8(p0 + bs);
        const uint8x16_t v1_1 = vld1q_u8(p1 + bs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        const int8x16_t v1_0l = vreinterpretq_s8_u8(vandq_u8(v1_0, m4b));

        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v1_0h = vreinterpretq_s8_u8(vshrq_n_u8(v1_0, 4));

        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        const int8x16_t v1_1l = vreinterpretq_s8_u8(vandq_u8(v1_1, m4b));

        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));
        const int8x16_t v1_1h = vreinterpretq_s8_u8(vshrq_n_u8(v1_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v1_0ls = vsubq_s8(v1_0l, s8b);

        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v1_0hs = vsubq_s8(v1_0h, s8b);

        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v1_1ls = vsubq_s8(v1_1l, s8b);

        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);
        const int8x16_t v1_1hs = vsubq_s8(v1_1h, s8b);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int16x8_t
        int32x4_t p_0 = vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0ls);
        int32x4_t p_1 = vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1ls);

        p_0 = vdotq_s32(p_0, v0_0hs, v1_0hs);
        p_1 = vdotq_s32(p_1, v0_1hs, v1_1hs);

        // scalar
#if defined(__ARM_FEATURE_QRDMX)
        sum0 += d0_0*d1_0*vaddvq_s32(p_0);
        sum1 += d0_1*d1_1*vaddvq_s32(p_1);
#else
        sum0 += d0_0*d1_0*(vgetq_lane_s32(p_0, 0) + vgetq_lane_s32(p_0, 1) + vgetq_lane_s32(p_0, 2) + vgetq_lane_s32(p_0, 3));
        sum1 += d0_1*d1_1*(vgetq_lane_s32(p_1, 0) + vgetq_lane_s32(p_1, 1) + vgetq_lane_s32(p_1, 2) + vgetq_lane_s32(p_1, 3));
#endif
#else
	    const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0ls));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0ls));

        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0hs));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0hs));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1ls), vget_low_s8 (v1_1ls));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1ls));

        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hs), vget_low_s8 (v1_1hs));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1hs));

        const int16x8_t pl_0 = vaddq_s16(pl0l, pl0h);
        const int16x8_t ph_0 = vaddq_s16(ph0l, ph0h);

        const int16x8_t pl_1 = vaddq_s16(pl1l, pl1h);
        const int16x8_t ph_1 = vaddq_s16(ph1l, ph1h);

        const int16x8_t p_0 = vaddq_s16(pl_0, ph_0);
        const int16x8_t p_1 = vaddq_s16(pl_1, ph_1);

        // scalar
#if defined(__ARM_FEATURE_QRDMX)
        sum0 += d0_0*d1_0*vaddvq_s16(p_0);
        sum1 += d0_1*d1_1*vaddvq_s16(p_1);
#else
        sum0 += d0_0*d1_0*(vgetq_lane_s16(p_0, 0) + vgetq_lane_s16(p_0, 1) + vgetq_lane_s16(p_0, 2) + vgetq_lane_s16(p_0, 3) + vgetq_lane_s16(p_0, 4) + vgetq_lane_s16(p_0, 5) + vgetq_lane_s16(p_0, 6) + vgetq_lane_s16(p_0, 7));
        sum1 += d0_1*d1_1*(vgetq_lane_s16(p_1, 0) + vgetq_lane_s16(p_1, 1) + vgetq_lane_s16(p_1, 2) + vgetq_lane_s16(p_1, 3) + vgetq_lane_s16(p_1, 4) + vgetq_lane_s16(p_1, 5) + vgetq_lane_s16(p_1, 6) + vgetq_lane_s16(p_1, 7));
#endif
#endif
    }

    sumf = sum0 + sum1;
#else
#error "not implemented for QK"
#endif
#elif defined(__AVX2__)
#if QK == 32
    const size_t countBlocks = nb;

    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        const float * d0_0 = (const float *) (pd0 + i*bs);
        const float * d1_0 = (const float *) (pd1 + i*bs);

        const uint8_t * restrict p0 = pb0 + i*bs;
        const uint8_t * restrict p1 = pb1 + i*bs;

        // Compute combined scale for the block
        const __m256 scale = _mm256_mul_ps( _mm256_broadcast_ss( d0_0 ), _mm256_broadcast_ss( d1_0 ) );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        __m256i bx = bytesFromNibbles( p0 );
        __m256i by = bytesFromNibbles( p1 );

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        bx = _mm256_sub_epi8( bx, off );
        by = _mm256_sub_epi8( by, off );

        // Sign-extend first 16 signed bytes into int16_t
        __m256i x16 = _mm256_cvtepi8_epi16( _mm256_castsi256_si128( bx ) );
        __m256i y16 = _mm256_cvtepi8_epi16( _mm256_castsi256_si128( by ) );
        // Compute products of int16_t integers, add pairwise
        __m256i i32 = _mm256_madd_epi16( x16, y16 );

        // Sign-extend last 16 signed bytes into int16_t vectors
        x16 = _mm256_cvtepi8_epi16( _mm256_extracti128_si256( bx, 1 ) );
        y16 = _mm256_cvtepi8_epi16( _mm256_extracti128_si256( by, 1 ) );
        // Accumulate products of int16_t integers
        i32 = _mm256_add_epi32( i32, _mm256_madd_epi16( x16, y16 ) );

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps( i32 );
        // Apply the scale, and accumulate
        acc = _mm256_fmadd_ps( scale, p, acc );
    }

    // Return horizontal sum of the acc vector
    __m128 res = _mm256_extractf128_ps( acc, 1 );
    res = _mm_add_ps( res, _mm256_castps256_ps128( acc ) );
    res = _mm_add_ps( res, _mm_movehl_ps( res, res ) );
    res = _mm_add_ss( res, _mm_movehdup_ps( res ) );

    sumf = _mm_cvtss_f32( res );
#else
#error "not implemented for QK"
#endif
#elif defined(__wasm_simd128__)
#if QK == 32
    // wasm simd
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int i = 0; i < nb; i += 2) {
        const float d0_0 = *(const float *) (pd0 + i*bs);
        const float d1_0 = *(const float *) (pd1 + i*bs);
        const float d0_1 = *(const float *) (pd0 + (i + 1)*bs);
        const float d1_1 = *(const float *) (pd1 + (i + 1)*bs);

        const uint8_t * restrict p0 = pb0 + i*bs;
        const uint8_t * restrict p1 = pb1 + i*bs;

        const v128_t m4b = wasm_u8x16_splat(0xf);
        const v128_t s8b = wasm_i8x16_splat(0x8);

        const v128_t v0_0 = wasm_v128_load(p0);
        const v128_t v0_1 = wasm_v128_load(p0 + bs);
        const v128_t v1_0 = wasm_v128_load(p1);
        const v128_t v1_1 = wasm_v128_load(p1 + bs);

        // 4-bit -> 8-bit
        const v128_t v0_0l = wasm_v128_and(v0_0, m4b);
        const v128_t v1_0l = wasm_v128_and(v1_0, m4b);

        const v128_t v0_0h = wasm_u8x16_shr(v0_0, 4);
        const v128_t v1_0h = wasm_u8x16_shr(v1_0, 4);

        const v128_t v0_1l = wasm_v128_and(v0_1, m4b);
        const v128_t v1_1l = wasm_v128_and(v1_1, m4b);

        const v128_t v0_1h = wasm_u8x16_shr(v0_1, 4);
        const v128_t v1_1h = wasm_u8x16_shr(v1_1, 4);

        // sub 8
        const v128_t v0_0ls = wasm_i8x16_sub(v0_0l, s8b);
        const v128_t v1_0ls = wasm_i8x16_sub(v1_0l, s8b);

        const v128_t v0_0hs = wasm_i8x16_sub(v0_0h, s8b);
        const v128_t v1_0hs = wasm_i8x16_sub(v1_0h, s8b);

        const v128_t v0_1ls = wasm_i8x16_sub(v0_1l, s8b);
        const v128_t v1_1ls = wasm_i8x16_sub(v1_1l, s8b);

        const v128_t v0_1hs = wasm_i8x16_sub(v0_1h, s8b);
        const v128_t v1_1hs = wasm_i8x16_sub(v1_1h, s8b);

        // dot product into int16x8_t
        const v128_t pl0l = wasm_i16x8_mul(wasm_i16x8_extend_low_i8x16(v0_0ls), wasm_i16x8_extend_low_i8x16(v1_0ls));
        const v128_t pl0h = wasm_i16x8_mul(wasm_i16x8_extend_high_i8x16(v0_0ls), wasm_i16x8_extend_high_i8x16(v1_0ls));

        const v128_t ph0l = wasm_i16x8_mul(wasm_i16x8_extend_low_i8x16(v0_0hs), wasm_i16x8_extend_low_i8x16(v1_0hs));
        const v128_t ph0h = wasm_i16x8_mul(wasm_i16x8_extend_high_i8x16(v0_0hs), wasm_i16x8_extend_high_i8x16(v1_0hs));

        const v128_t pl1l = wasm_i16x8_mul(wasm_i16x8_extend_low_i8x16(v0_1ls), wasm_i16x8_extend_low_i8x16(v1_1ls));
        const v128_t pl1h = wasm_i16x8_mul(wasm_i16x8_extend_high_i8x16(v0_1ls), wasm_i16x8_extend_high_i8x16(v1_1ls));

        const v128_t ph1l = wasm_i16x8_mul(wasm_i16x8_extend_low_i8x16(v0_1hs), wasm_i16x8_extend_low_i8x16(v1_1hs));
        const v128_t ph1h = wasm_i16x8_mul(wasm_i16x8_extend_high_i8x16(v0_1hs), wasm_i16x8_extend_high_i8x16(v1_1hs));

        const v128_t pl_0 = wasm_i16x8_add(pl0l, pl0h);
        const v128_t ph_0 = wasm_i16x8_add(ph0l, ph0h);

        const v128_t pl_1 = wasm_i16x8_add(pl1l, pl1h);
        const v128_t ph_1 = wasm_i16x8_add(ph1l, ph1h);

        const v128_t p_0 = wasm_i16x8_add(pl_0, ph_0);
        const v128_t p_1 = wasm_i16x8_add(pl_1, ph_1);

        sum0 += d0_0*d1_0*(
                wasm_i16x8_extract_lane(p_0, 0) + wasm_i16x8_extract_lane(p_0, 1) +
                wasm_i16x8_extract_lane(p_0, 2) + wasm_i16x8_extract_lane(p_0, 3) +
                wasm_i16x8_extract_lane(p_0, 4) + wasm_i16x8_extract_lane(p_0, 5) +
                wasm_i16x8_extract_lane(p_0, 6) + wasm_i16x8_extract_lane(p_0, 7));
        sum1 += d0_1*d1_1*(
                wasm_i16x8_extract_lane(p_1, 0) + wasm_i16x8_extract_lane(p_1, 1) +
                wasm_i16x8_extract_lane(p_1, 2) + wasm_i16x8_extract_lane(p_1, 3) +
                wasm_i16x8_extract_lane(p_1, 4) + wasm_i16x8_extract_lane(p_1, 5) +
                wasm_i16x8_extract_lane(p_1, 6) + wasm_i16x8_extract_lane(p_1, 7));
    }

    sumf = sum0 + sum1;
#else
#error "not implemented for QK"
#endif
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const float d0 = *(const float *) (pd0 + i*bs);
        const float d1 = *(const float *) (pd1 + i*bs);

        const uint8_t * restrict p0 = pb0 + i*bs;
        const uint8_t * restrict p1 = pb1 + i*bs;

        for (int j = 0; j < QK/2; j++) {
            const uint8_t v0 = p0[j];
            const uint8_t v1 = p1[j];

            const float f0 = d0*((int8_t) (v0 & 0xf) - 8);
            const float f1 = d0*((int8_t) (v0 >> 4)  - 8);

            const float f2 = d1*((int8_t) (v1 & 0xf) - 8);
            const float f3 = d1*((int8_t) (v1 >> 4)  - 8);

            sumf += f0*f2 + f1*f3;
        }
    }
#endif

    *s = sumf;
}

inline static void dragon_vec_dot_q4_1(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const int nb = n / QK;

    const size_t bs = 2*sizeof(float) + QK/2;

    const uint8_t * restrict pd0 = ((const uint8_t *)x + 0*bs);
    const uint8_t * restrict pd1 = ((const uint8_t *)y + 0*bs);

    const uint8_t * restrict pm0 = ((const uint8_t *)x + 0*bs + sizeof(float));
    const uint8_t * restrict pm1 = ((const uint8_t *)y + 0*bs + sizeof(float));

    const uint8_t * restrict pb0 = ((const uint8_t *)x + 0*bs + 2*sizeof(float));
    const uint8_t * restrict pb1 = ((const uint8_t *)y + 0*bs + 2*sizeof(float));

    float sumf = 0.0;

#if defined(__AVX2__)
#if QK == 32
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    // Accumulator for constant offsets
    float acc_offset = 0.0f;

    // Main loop
    for (int i = 0; i < nb; ++i) {
        const float * m0 = (const float *) (pm0 + i*bs);
        const float * m1 = (const float *) (pm1 + i*bs);

        const float * d0 = (const float *) (pd0 + i*bs);
        const float * d1 = (const float *) (pd1 + i*bs);

        const uint8_t * restrict p0 = pb0 + i*bs;
        const uint8_t * restrict p1 = pb1 + i*bs;

        const __m256 d0v = _mm256_broadcast_ss( d0 );
        const __m256 d1v = _mm256_broadcast_ss( d1 );
        const __m256 m0v = _mm256_broadcast_ss( m0 );
        const __m256 m1v = _mm256_broadcast_ss( m1 );


        // Compute combined scale for the block
        const __m256 scale_01 = _mm256_mul_ps( d0v, d1v );

        // Compute cross scales for the block
        const __m256 scale_0 = _mm256_mul_ps( d0v, m1v );
        const __m256 scale_1 = _mm256_mul_ps( m0v, d1v );
        const __m256 cross_scales = _mm256_blend_ps( scale_0, scale_1, 0b10101010 );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        __m256i bx = bytesFromNibbles( p0 );
        __m256i by = bytesFromNibbles( p1 );

        // Now we have a vector with bytes in [ 0 .. 15 ] interval.

        // Sign-extend first 16 signed bytes into int16_t
        __m256i x16 = _mm256_cvtepi8_epi16( _mm256_castsi256_si128( bx ) );
        __m256i y16 = _mm256_cvtepi8_epi16( _mm256_castsi256_si128( by ) );
        // Compute products of int16_t integers, add pairwise
        __m256i i32 = _mm256_madd_epi16( x16, y16 );

        // Sign-extend last 16 signed bytes into int16_t vectors
        __m256i x16_h = _mm256_cvtepi8_epi16( _mm256_extracti128_si256( bx, 1 ) );
        __m256i y16_h = _mm256_cvtepi8_epi16( _mm256_extracti128_si256( by, 1 ) );
        // Accumulate products of int16_t integers
        i32 = _mm256_add_epi32( i32, _mm256_madd_epi16( x16_h, y16_h ) );

        // compute sums of unsigned bytes in bx, by in blocks of 8.
        // This results in a layout like X100 0000 X200 0000 X300 0000 X400 0000,
        // which we then interleave as X100 Y100 X200 Y200 X300 Y300 X400 Y400.
        // so if we then cast to 8 singles, we get 8 floats like [ x0_7, y0_7, x8_15, y8_15, x16_23, y16_23, x24_31, y24_31 ]
        __m256i xsumi = _mm256_sad_epu8( bx, _mm256_setzero_si256() );
        __m256i ysumi = _mm256_sad_epu8( by, _mm256_setzero_si256() );
        __m256i sumsi = _mm256_or_si256( xsumi, _mm256_slli_si256( ysumi, 4 ) );
        __m256  sums  = _mm256_cvtepi32_ps( sumsi );

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps( i32 );
        // Apply the scale, and accumulate
        // acc += d0*d1*x*y + d0*m1*x + d1*m0*y
        acc = _mm256_fmadd_ps( scale_01, p, acc );
        acc = _mm256_fmadd_ps( cross_scales, sums, acc );
        // acc_offset += m0*m1 (for each entry in the block)
        acc_offset += (*m0)*(*m1);
    }

    // Return horizontal sum of the acc vector
    __m128 res = _mm256_extractf128_ps( acc, 1 );
    res = _mm_add_ps( res, _mm256_castps256_ps128( acc ) );
    res = _mm_add_ps( res, _mm_movehl_ps( res, res ) );
    res = _mm_add_ss( res, _mm_movehdup_ps( res ) );

    sumf = _mm_cvtss_f32( res ) + acc_offset * QK;
#else
#error "not implemented for QK"
#endif
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const float m0 = *(const float *) (pm0 + i*bs);
        const float m1 = *(const float *) (pm1 + i*bs);

        const float d0 = *(const float *) (pd0 + i*bs);
        const float d1 = *(const float *) (pd1 + i*bs);

        const uint8_t * restrict p0 = pb0 + i*bs;
        const uint8_t * restrict p1 = pb1 + i*bs;

        for (int j = 0; j < QK/2; j++) {
            const uint8_t v0 = p0[j];
            const uint8_t v1 = p1[j];

            const float f0 = d0*(v0 & 0xf) + m0;
            const float f1 = d0*(v0 >> 4)  + m0;

            const float f2 = d1*(v1 & 0xf) + m1;
            const float f3 = d1*(v1 >> 4)  + m1;

            sumf += f0*f2 + f1*f3;
        }
    }
#endif

    *s = sumf;
}

// compute DRAGON_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void dragon_vec_dot_f16_unroll(const int n, const int xs, float * restrict s, void * restrict xv, dragon_fp16_t * restrict y) {
    dragon_float sumf[DRAGON_VEC_DOT_UNROLL] = { 0.0 };

    dragon_fp16_t * restrict x[DRAGON_VEC_DOT_UNROLL];

    for (int i = 0; i < DRAGON_VEC_DOT_UNROLL; ++i) {
        x[i] = (dragon_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(DRAGON_SIMD)
    const int np = (n & ~(DRAGON_F16_STEP - 1));

    DRAGON_F16_VEC sum[DRAGON_VEC_DOT_UNROLL][DRAGON_F16_ARR] = { { DRAGON_F16_VEC_ZERO } };

    DRAGON_F16_VEC ax[DRAGON_F16_ARR];
    DRAGON_F16_VEC ay[DRAGON_F16_ARR];

    for (int i = 0; i < np; i += DRAGON_F16_STEP) {
        for (int j = 0; j < DRAGON_F16_ARR; j++) {
            ay[j] = DRAGON_F16_VEC_LOAD(y + i + j*DRAGON_F16_EPR, j);

            for (int k = 0; k < DRAGON_VEC_DOT_UNROLL; ++k) {
                ax[j] = DRAGON_F16_VEC_LOAD(x[k] + i + j*DRAGON_F16_EPR, j);

                sum[k][j] = DRAGON_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
            }
        }
    }

    // reduce sum0..sum3 to sum0
    for (int k = 0; k < DRAGON_VEC_DOT_UNROLL; ++k) {
        DRAGON_F16_VEC_REDUCE(sumf[k], sum[k]);
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        for (int j = 0; j < DRAGON_VEC_DOT_UNROLL; ++j) {
            sumf[j] += DRAGON_FP16_TO_FP32(x[j][i])*DRAGON_FP16_TO_FP32(y[i]);
        }
    }
#else
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < DRAGON_VEC_DOT_UNROLL; ++j) {
            sumf[j] += DRAGON_FP16_TO_FP32(x[j][i])*DRAGON_FP16_TO_FP32(y[i]);
        }
    }
#endif

    for (int i = 0; i < DRAGON_VEC_DOT_UNROLL; ++i) {
        s[i] = sumf[i];
    }
}

inline static void dragon_vec_mad_f32(const int n, float * restrict y, const float * restrict x, const float v) {
#if defined(DRAGON_SIMD)
    const int np = (n & ~(DRAGON_F32_STEP - 1));

    DRAGON_F32_VEC vx = DRAGON_F32_VEC_SET1(v);

    DRAGON_F32_VEC ax[DRAGON_F32_ARR];
    DRAGON_F32_VEC ay[DRAGON_F32_ARR];

    for (int i = 0; i < np; i += DRAGON_F32_STEP) {
        for (int j = 0; j < DRAGON_F32_ARR; j++) {
            ax[j] = DRAGON_F32_VEC_LOAD(x + i + j*DRAGON_F32_EPR);
            ay[j] = DRAGON_F32_VEC_LOAD(y + i + j*DRAGON_F32_EPR);
            ay[j] = DRAGON_F32_VEC_FMA(ay[j], ax[j], vx);

            DRAGON_F32_VEC_STORE(y + i + j*DRAGON_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] += x[i]*v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

inline static void dragon_vec_mad_f16(const int n, dragon_fp16_t * restrict y, dragon_fp16_t * restrict x, const float v) {
#if defined(DRAGON_SIMD)
    const int np = (n & ~(DRAGON_F16_STEP - 1));

    DRAGON_F16_VEC vx = DRAGON_F16_VEC_SET1(v);

    DRAGON_F16_VEC ax[DRAGON_F16_ARR];
    DRAGON_F16_VEC ay[DRAGON_F16_ARR];

    for (int i = 0; i < np; i += DRAGON_F16_STEP) {
        for (int j = 0; j < DRAGON_F16_ARR; j++) {
            ax[j] = DRAGON_F16_VEC_LOAD(x + i + j*DRAGON_F16_EPR, j);
            ay[j] = DRAGON_F16_VEC_LOAD(y + i + j*DRAGON_F16_EPR, j);
            ay[j] = DRAGON_F16_VEC_FMA(ay[j], ax[j], vx);

            DRAGON_F16_VEC_STORE(y + i + j*DRAGON_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        DRAGON_ASSERT(false);
        y[i] = DRAGON_FP32_TO_FP16(DRAGON_FP16_TO_FP32(y[i]) + DRAGON_FP16_TO_FP32(x[i])*v);
    }
#else
    for (int i = 0; i < n; ++i) {
        y[i] = DRAGON_FP32_TO_FP16(DRAGON_FP16_TO_FP32(y[i]) + DRAGON_FP16_TO_FP32(x[i])*v);
    }
#endif
}

inline static void dragon_vec_mad_q4_0(const int n, float * restrict y, void * restrict x, const float v) {
    assert(n % QK == 0);

    const int nb = n / QK;
    const size_t bs = sizeof(float) + QK/2;

    const uint8_t * restrict pd = ((const uint8_t *)x + 0*bs);
    const uint8_t * restrict pb = ((const uint8_t *)x + 0*bs + sizeof(float));

#if __ARM_NEON
#if QK == 32
    for (int i = 0; i < nb; ++i) {
        const float d0 = v*(*(const float *) (pd + i*bs));

        const uint8_t * restrict pp = pb + i*bs;

        const uint8x8_t m4b = vdup_n_u8(0xf);
        const int8x8_t  s8b = vdup_n_s8(0x8);

        const float32x4_t vd = vdupq_n_f32(d0);

        for (int j = 0; j < 2; j++) {
            const uint8x8_t vx = vld1_u8(pp + j*8);

            const int8x8_t vxl = vreinterpret_s8_u8(vand_u8(vx, m4b));
            const int8x8_t vxh = vreinterpret_s8_u8(vshr_n_u8(vx, 4));

            // sub 8
            const int8x8_t vxls = vsub_s8(vxl, s8b);
            const int8x8_t vxhs = vsub_s8(vxh, s8b);

            //const int8x8_t vxlt = vzip_s8(vxls, vxhs)[0];
            //const int8x8_t vxht = vzip_s8(vxls, vxhs)[1];
            const int8x8_t vxlt = vzip1_s8(vxls, vxhs);
            const int8x8_t vxht = vzip2_s8(vxls, vxhs);

            const int8x16_t vxq = vcombine_s8(vxlt, vxht);

            // convert to 2x int16x8_t
            const int16x8_t vxq0 = vmovl_s8(vget_low_s8 (vxq));
            const int16x8_t vxq1 = vmovl_s8(vget_high_s8(vxq));

            // convert to 4x float32x4_t
            const float32x4_t vx0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vxq0)));
            const float32x4_t vx1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vxq0)));
            const float32x4_t vx2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vxq1)));
            const float32x4_t vx3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vxq1)));

            const float32x4_t vy0 = vld1q_f32(y + i*32 + j*16 + 0);
            const float32x4_t vy1 = vld1q_f32(y + i*32 + j*16 + 4);
            const float32x4_t vy2 = vld1q_f32(y + i*32 + j*16 + 8);
            const float32x4_t vy3 = vld1q_f32(y + i*32 + j*16 + 12);

            const float32x4_t vr0 = vfmaq_f32(vy0, vx0, vd);
            const float32x4_t vr1 = vfmaq_f32(vy1, vx1, vd);
            const float32x4_t vr2 = vfmaq_f32(vy2, vx2, vd);
            const float32x4_t vr3 = vfmaq_f32(vy3, vx3, vd);

            vst1q_f32(y + i*32 + j*16 + 0,  vr0);
            vst1q_f32(y + i*32 + j*16 + 4,  vr1);
            vst1q_f32(y + i*32 + j*16 + 8,  vr2);
            vst1q_f32(y + i*32 + j*16 + 12, vr3);
        }
    }
#endif
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const float d = *(const float *) (pd + i*bs);

        const uint8_t * restrict pp = pb + i*bs;

        for (int l = 0; l < QK; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8)*d;
            const float v1 = (vi1 - 8)*d;

            y[i*QK + l + 0] += v0*v;
            y[i*QK + l + 1] += v1*v;

            assert(!isnan(y[i*QK + l + 0]));
            assert(!isnan(y[i*QK + l + 1]));
            assert(!isinf(y[i*QK + l + 0]));
            assert(!isinf(y[i*QK + l + 1]));
        }
    }
#endif
}

inline static void dragon_vec_mad_q4_1(const int n, float * restrict y, void * restrict x, const float v) {
    assert(n % QK == 0);

    const int nb = n / QK;
    const size_t bs = 2*sizeof(float) + QK/2;

    const uint8_t * restrict pd = ((const uint8_t *)x + 0*bs);
    const uint8_t * restrict pm = ((const uint8_t *)x + 0*bs +   sizeof(float)); 
    const uint8_t * restrict pb = ((const uint8_t *)x + 0*bs + 2*sizeof(float));

    for (int i = 0; i < nb; i++) {
        const float d = *(const float *) (pd + i*bs);
        const float m = *(const float *) (pm + i*bs);

        const uint8_t * restrict pp = pb + i*bs;

        for (int l = 0; l < QK; l += 2) {
            const uint8_t vi = pp[l/2];

            const uint8_t vi0 = vi & 0xf;
            const uint8_t vi1 = vi >> 4;

            const float v0 = d*vi0 + m;
            const float v1 = d*vi1 + m;

            y[i*QK + l + 0] += v0*v;
            y[i*QK + l + 1] += v1*v;

            assert(!isnan(y[i*QK + l + 0]));
            assert(!isnan(y[i*QK + l + 1]));
            assert(!isinf(y[i*QK + l + 0]));
            assert(!isinf(y[i*QK + l + 1]));
            //printf("mad: v0 %f v1 %f, i = %d, l = %d, d = %f, vi = %d, vi0 = %d, vi1 = %d\n", v0, v1, i, l, d, vi, vi0, vi1);
        }
    }
}

//inline static void dragon_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
inline static void dragon_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(DRAGON_SIMD)
    const int np = (n & ~(DRAGON_F32_STEP - 1));

    DRAGON_F32_VEC vx = DRAGON_F32_VEC_SET1(v);

    DRAGON_F32_VEC ay[DRAGON_F32_ARR];

    for (int i = 0; i < np; i += DRAGON_F32_STEP) {
        for (int j = 0; j < DRAGON_F32_ARR; j++) {
            ay[j] = DRAGON_F32_VEC_LOAD(y + i + j*DRAGON_F32_EPR);
            ay[j] = DRAGON_F32_VEC_MUL(ay[j], vx);

            DRAGON_F32_VEC_STORE(y + i + j*DRAGON_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void dragon_vec_norm_f32 (const int n, float * s, const float * x) { dragon_vec_dot_f32(n, s, x, x); *s = sqrt(*s);   }
inline static void dragon_vec_sqr_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
inline static void dragon_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrt(x[i]); }
inline static void dragon_vec_abs_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
inline static void dragon_vec_sgn_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
inline static void dragon_vec_step_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
inline static void dragon_vec_relu_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }

static const dragon_float GELU_COEF_A    = 0.044715;
static const dragon_float SQRT_2_OVER_PI = 0.79788456080286535587989211986876;

inline static float dragon_gelu_f32(float x) {
    return 0.5*x*(1.0 + tanh(SQRT_2_OVER_PI*x*(1.0 + GELU_COEF_A*x*x)));
}

inline static void dragon_vec_gelu_f16(const int n, dragon_fp16_t * y, const dragon_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = table_gelu_f16[i16[i]];
    }
}

#ifdef DRAGON_GELU_FP16
inline static void dragon_vec_gelu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        dragon_fp16_t fp16 = DRAGON_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = DRAGON_FP16_TO_FP32(table_gelu_f16[t]);
    }
}
#else
inline static void dragon_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = dragon_gelu_f32(x[i]);
    }
}
#endif

// Sigmoid Linear Unit (SiLU) function
inline static float dragon_silu_f32(float x) {
    return x/(1.0 + exp(-x));
}

inline static void dragon_vec_silu_f16(const int n, dragon_fp16_t * y, const dragon_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = table_silu_f16[i16[i]];
    }
}

#ifdef DRAGON_SILU_FP16
inline static void dragon_vec_silu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        dragon_fp16_t fp16 = DRAGON_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = DRAGON_FP16_TO_FP32(table_silu_f16[t]);
    }
}
#else
inline static void dragon_vec_silu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = dragon_silu_f32(x[i]);
    }
}
#endif

inline static void dragon_vec_sum_f32(const int n, float * s, const float * x) {
#ifndef DRAGON_USE_ACCELERATE
    dragon_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    *s = sum;
#else
    vDSP_sve(x, 1, s, n);
#endif
}

inline static void dragon_vec_max_f32(const int n, float * s, const float * x) {
#ifndef DRAGON_USE_ACCELERATE
    dragon_float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
#else
    vDSP_maxv(x, 1, s, n);
#endif
}

inline static void dragon_vec_norm_inv_f32(const int n, float * s, const float * x) { dragon_vec_norm_f32(n, s, x); *s = 1./(*s); }

//
// logging
//

#if (DRAGON_DEBUG >= 1)
#define DRAGON_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define DRAGON_PRINT_DEBUG(...)
#endif

#if (DRAGON_DEBUG >= 5)
#define DRAGON_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define DRAGON_PRINT_DEBUG_5(...)
#endif

#if (DRAGON_DEBUG >= 10)
#define DRAGON_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define DRAGON_PRINT_DEBUG_10(...)
#endif

#define DRAGON_PRINT(...) printf(__VA_ARGS__)

//
// data types
//

static const int DRAGON_BLCK_SIZE[DATA_TYPE_COUNT] = {
    QK,
    QK,
    1,
    1,
    1,
    1,
    1,
};

static_assert(DATA_TYPE_COUNT == 7, "DATA_TYPE_COUNT != 5");

static const size_t DRAGON_TYPE_SIZE[DATA_TYPE_COUNT] = {
    sizeof(float  )   + QK/2,
    sizeof(float  )*2 + QK/2,
    sizeof(int8_t ),
    sizeof(int16_t),
    sizeof(int32_t),
    sizeof(dragon_fp16_t),
    sizeof(float  ),
};

// don't forget to update the array above when adding new types
static_assert(DATA_TYPE_COUNT == 7, "DATA_TYPE_COUNT != 5");

static const char * DRAGON_OP_LABEL[DRAGON_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "SUM",
    "MEAN",
    "REPEAT",
    "ABS",
    "SGN",
    "NEG",
    "STEP",
    "RELU",
    "GELU",
    "SILU",
    "NORM",
    "RMS_NORM",

    "MUL_MAT",

    "SCALE",
    "CPY",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "DIAG_MASK_INF",
    "SOFT_MAX",
    "ROPE",
    "CONV_1D_1S",
    "CONV_1D_2S",

    "FLASH_ATTN",
    "FLASH_FF",
};

static_assert(DRAGON_OP_COUNT == 35, "DRAGON_OP_COUNT != 35");

static const char * DRAGON_OP_SYMBOL[DRAGON_OP_COUNT] = {
    "none",

    "x",
    "x+y",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "Σx",
    "Σx/n",
    "repeat(x)",
    "abs(x)",
    "sgn(x)",
    "-x",
    "step(x)",
    "relu(x)",
    "gelu(x)",
    "silu(x)",
    "norm(x)",
    "rms_norm(x)",

    "X*Y",

    "x*v",
    "x-\\>y",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "diag_mask_inf(x)",
    "soft_max(x)",
    "rope(x)",
    "conv_1d_1s(x)",
    "conv_1d_2s(x)",

    "flash_attn(x)",
    "flash_ff(x)",
};

static_assert(DRAGON_OP_COUNT == 35, "DRAGON_OP_COUNT != 35");

//
// dragon object
//

struct dragon_object {
    size_t offs;
    size_t size;

    struct dragon_object * next;

    char padding[8];
};

static const size_t DRAGON_OBJECT_SIZE = sizeof(struct dragon_object);

static_assert(sizeof(struct dragon_object)%DRAGON_MEM_ALIGN == 0, "dragon_object size must be a multiple of DRAGON_MEM_ALIGN");
static_assert(sizeof(struct dragon_tensor)%DRAGON_MEM_ALIGN == 0, "dragon_tensor size must be a multiple of DRAGON_MEM_ALIGN");

//
// dragon context
//

struct dragon_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;

    int n_objects;

    struct dragon_object * objects_begin;
    struct dragon_object * objects_end;

    struct dragon_scratch scratch;
    struct dragon_scratch scratch_save;
};

struct dragon_context_container {
    bool used;

    struct dragon_context context;
};

//
// compute types
//

enum dragon_task_type {
    DRAGON_TASK_INIT = 0,
    DRAGON_TASK_COMPUTE,
    DRAGON_TASK_FINALIZE,
};

struct dragon_compute_params {
    enum dragon_task_type type;

    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;
};

//
// dragon state
//

struct dragon_state {
    struct dragon_context_container contexts[DRAGON_MAX_CONTEXTS];
};

// global state
static struct dragon_state g_state;
static atomic_int g_state_barrier = 0;

// barrier via spin lock
inline static void dragon_critical_section_start(void) {
    int processing = atomic_fetch_add(&g_state_barrier, 1);

    while (processing > 0) {
        // wait for other threads to finish
        atomic_fetch_sub(&g_state_barrier, 1);
        sched_yield(); // TODO: reconsider this
        processing = atomic_fetch_add(&g_state_barrier, 1);
    }
}

// TODO: make this somehow automatically executed
//       some sort of "sentry" mechanism
inline static void dragon_critical_section_end(void) {
    atomic_fetch_sub(&g_state_barrier, 1);
}

////////////////////////////////////////////////////////////////////////////////

void dragon_print_object(const struct dragon_object * obj) {
    DRAGON_PRINT(" - dragon_object: offset = %zu, size = %zu, next = %p\n",
            obj->offs, obj->size, (const void *) obj->next);
}

void dragon_print_objects(const struct dragon_context * ctx) {
    struct dragon_object * obj = ctx->objects_begin;

    DRAGON_PRINT("%s: objects in context %p:\n", __func__, (const void *) ctx);

    while (obj != NULL) {
        dragon_print_object(obj);
        obj = obj->next;
    }

    DRAGON_PRINT("%s: --- end ---\n", __func__);
}

int dragon_nelements(const struct dragon_tensor * tensor) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int dragon_nrows(const struct dragon_tensor * tensor) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t dragon_nbytes(const struct dragon_tensor * tensor) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return (dragon_nelements(tensor)*DRAGON_TYPE_SIZE[tensor->type])/DRAGON_BLCK_SIZE[tensor->type];
}

int dragon_blck_size(enum data_type type) {
    return DRAGON_BLCK_SIZE[type];
}

size_t dragon_type_size(enum data_type type) {
    return DRAGON_TYPE_SIZE[type];
}

float dragon_type_sizef(enum data_type type) {
    return ((float)(DRAGON_TYPE_SIZE[type]))/DRAGON_BLCK_SIZE[type];
}

size_t dragon_element_size(const struct dragon_tensor * tensor) {
    return DRAGON_TYPE_SIZE[tensor->type];
}

static inline bool dragon_is_scalar(const struct dragon_tensor * tensor) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool dragon_is_vector(const struct dragon_tensor * tensor) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool dragon_is_matrix(const struct dragon_tensor * tensor) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool dragon_can_mul_mat(const struct dragon_tensor * t0, const struct dragon_tensor * t1) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return
        (t0->ne[0]  == t1->ne[0])  &&
        (t0->ne[2]  == t1->ne[2])  &&
        (t0->ne[3]  == t1->ne[3]);
}

static inline bool dragon_is_contiguous(const struct dragon_tensor * tensor) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == DRAGON_TYPE_SIZE[tensor->type] &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/DRAGON_BLCK_SIZE[tensor->type] &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

static inline bool dragon_is_padded_1d(const struct dragon_tensor * tensor) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == DRAGON_TYPE_SIZE[tensor->type] &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

static inline bool dragon_are_same_shape(const struct dragon_tensor * t0, const struct dragon_tensor * t1) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return
        (t0->ne[0] == t1->ne[0] ) &&
        (t0->ne[1] == t1->ne[1] ) &&
        (t0->ne[2] == t1->ne[2] ) &&
        (t0->ne[3] == t1->ne[3] );
}

// check if t1 can be represented as a repeatition of t0
static inline bool dragon_can_repeat(const struct dragon_tensor * t0, const struct dragon_tensor * t1) {
    static_assert(DRAGON_MAX_DIMS == 4, "DRAGON_MAX_DIMS is not 4 - update this function");

    return
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

static inline int dragon_up32(int n) {
    return (n + 31) & ~31;
}

static inline int dragon_up64(int n) {
    return (n + 63) & ~63;
}

static inline int dragon_up(int n, int m) {
    // assert m is a power of 2
    DRAGON_ASSERT((m & (m - 1)) == 0);
    return (n + m - 1) & ~(m - 1);
}

// assert that pointer is aligned to DRAGON_MEM_ALIGN
#define dragon_assert_aligned(ptr) \
    assert(((uintptr_t) (ptr))%DRAGON_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

struct dragon_context * dragon_init(struct dragon_init_params params) {
    // make this function thread safe
    dragon_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = dragon_time_us(); UNUSED(t_start);

            dragon_fp16_t ii;
            for (int i = 0; i < (1 << 16); ++i) {
                uint16_t ui = i;
                memcpy(&ii, &ui, sizeof(ii));
                const float f = table_f32_f16[i] = DRAGON_COMPUTE_FP16_TO_FP32(ii);
                table_gelu_f16[i] = DRAGON_FP32_TO_FP16(dragon_gelu_f32(f));
                table_silu_f16[i] = DRAGON_FP32_TO_FP16(dragon_silu_f32(f));
                table_exp_f16[i]  = DRAGON_FP32_TO_FP16(exp(f));
            }

            const uint64_t t_end = dragon_time_us(); UNUSED(t_end);

            DRAGON_PRINT_DEBUG("%s: GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        // initialize g_state
        {
            const uint64_t t_start = dragon_time_us(); UNUSED(t_start);

            g_state = (struct dragon_state) {
                /*.contexts =*/ { { 0 } },
            };

            for (int i = 0; i < DRAGON_MAX_CONTEXTS; ++i) {
                g_state.contexts[i].used = false;
            }

            const uint64_t t_end = dragon_time_us(); UNUSED(t_end);

            DRAGON_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        is_first_call = false;
    }

    // find non-used context in g_state
    struct dragon_context * ctx = NULL;

    for (int i = 0; i < DRAGON_MAX_CONTEXTS; i++) {
        if (!g_state.contexts[i].used) {
            g_state.contexts[i].used = true;
            ctx = &g_state.contexts[i].context;

            DRAGON_PRINT_DEBUG("%s: found unused context %d\n", __func__, i);
            break;
        }
    }

    if (ctx == NULL) {
        DRAGON_PRINT_DEBUG("%s: no unused context found\n", __func__);

        dragon_critical_section_end();

        return NULL;
    }

    *ctx = (struct dragon_context) {
        /*.mem_size         =*/ params.mem_size,
        /*.mem_buffer       =*/ params.mem_buffer ? params.mem_buffer : malloc(params.mem_size),
        /*.mem_buffer_owned =*/ params.mem_buffer ? false : true,
        /*.n_objects        =*/ 0,
        /*.objects_begin    =*/ NULL,
        /*.objects_end      =*/ NULL,
        /*.scratch          =*/ { 0, 0, NULL, },
        /*.scratch_save     =*/ { 0, 0, NULL, },
    };

    dragon_assert_aligned(ctx->mem_buffer);

    DRAGON_PRINT_DEBUG("%s: context initialized\n", __func__);

    dragon_critical_section_end();

    return ctx;
}

void dragon_free(struct dragon_context * ctx) {
    // make this function thread safe
    dragon_critical_section_start();

    bool found = false;

    for (int i = 0; i < DRAGON_MAX_CONTEXTS; i++) {
        if (&g_state.contexts[i].context == ctx) {
            g_state.contexts[i].used = false;

            DRAGON_PRINT_DEBUG("%s: context %d with %d objects has been freed. memory used = %zu\n",
                    __func__, i, ctx->n_objects, ctx->objects_end->offs + ctx->objects_end->size);

            if (ctx->mem_buffer_owned) {
                free(ctx->mem_buffer);
            }

            found = true;
            break;
        }
    }

    if (!found) {
        DRAGON_PRINT_DEBUG("%s: context not found\n", __func__);
    }

    dragon_critical_section_end();
}

size_t dragon_used_mem(const struct dragon_context * ctx) {
    return ctx->objects_end->offs + ctx->objects_end->size;
}

size_t dragon_set_scratch(struct dragon_context * ctx, struct dragon_scratch scratch) {
    const size_t result = ctx->scratch.data ? ctx->scratch.offs : 0;

    ctx->scratch = scratch;

    return result;
}

////////////////////////////////////////////////////////////////////////////////

struct dragon_tensor * dragon_new_tensor_impl(
        struct dragon_context * ctx,
        enum   data_type type,
        int    n_dims,
        const int* ne,
        void*  data) {
    // always insert objects at the end of the context's memory pool
    struct dragon_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    size_t size_needed = 0;

    if (data == NULL) {
        size_needed += DRAGON_TYPE_SIZE[type]*(ne[0]/DRAGON_BLCK_SIZE[type]);
        for (int i = 1; i < n_dims; i++) {
            size_needed *= ne[i];
        }
        // align to DRAGON_MEM_ALIGN
        size_needed = ((size_needed + DRAGON_MEM_ALIGN - 1)/DRAGON_MEM_ALIGN)*DRAGON_MEM_ALIGN;
    }

    char * const mem_buffer = ctx->mem_buffer;
    struct dragon_object * const obj_new = (struct dragon_object *)(mem_buffer + cur_end);

    if (ctx->scratch.data == NULL || data != NULL) {
        size_needed += sizeof(struct dragon_tensor);

        if (cur_end + size_needed + DRAGON_OBJECT_SIZE > ctx->mem_size) {
            DRAGON_PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                    __func__, cur_end + size_needed + DRAGON_OBJECT_SIZE, ctx->mem_size);
            assert(false);
            return NULL;
        }

        *obj_new = (struct dragon_object) {
            .offs = cur_end + DRAGON_OBJECT_SIZE,
            .size = size_needed,
            .next = NULL,
        };
    } else {
        if (ctx->scratch.offs + size_needed > ctx->scratch.size) {
            DRAGON_PRINT("%s: not enough space in the scratch memory\n", __func__);
            assert(false);
            return NULL;
        }

        if (cur_end + sizeof(struct dragon_tensor) + DRAGON_OBJECT_SIZE > ctx->mem_size) {
            DRAGON_PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                    __func__, cur_end + sizeof(struct dragon_tensor) + DRAGON_OBJECT_SIZE, ctx->mem_size);
            assert(false);
            return NULL;
        }

        data = (char * const) ctx->scratch.data + ctx->scratch.offs;

        *obj_new = (struct dragon_object) {
            .offs = cur_end + DRAGON_OBJECT_SIZE,
            .size = sizeof(struct dragon_tensor),
            .next = NULL,
        };

        //printf("scratch offs = %zu, size_needed = %zu\n", ctx->scratch.offs, size_needed);

        ctx->scratch.offs += size_needed;
    }

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    struct dragon_tensor * const result = (struct dragon_tensor *)(mem_buffer + obj_new->offs);

    dragon_assert_aligned(result);

    *result = (struct dragon_tensor) {
        /*.type         =*/ type,
        /*.n_dims       =*/ n_dims,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ DRAGON_OP_NONE,
        /*.is_param     =*/ false,
        /*.grad         =*/ NULL,
        /*.src0         =*/ NULL,
        /*.src1         =*/ NULL,
        /*.opt          =*/ { NULL },
        /*.n_tasks      =*/ 0,
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
        /*.data         =*/ data == NULL ? (void *)(result + 1) : data,
        /*.pad          =*/ { 0 },
    };

    dragon_assert_aligned(result->data);

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = DRAGON_TYPE_SIZE[type];
    result->nb[1] = result->nb[0]*(result->ne[0]/DRAGON_BLCK_SIZE[type]);
    for (int i = 2; i < DRAGON_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    return result;
}

struct dragon_tensor * dragon_new_tensor(
        struct dragon_context * ctx,
        enum   data_type type,
        int    n_dims,
        const int * ne) {
    return dragon_new_tensor_impl(ctx, type, n_dims, ne, NULL);
}

struct dragon_tensor * dragon_new_tensor_1d(
        struct dragon_context * ctx,
        enum   data_type type,
        int    ne0) {
    return dragon_new_tensor(ctx, type, 1, &ne0);
}

struct dragon_tensor * dragon_new_tensor_2d(
        struct dragon_context * ctx,
        enum   data_type type,
        int    ne0,
        int    ne1) {
    const int ne[2] = { ne0, ne1 };
    return dragon_new_tensor(ctx, type, 2, ne);
}

struct dragon_tensor * dragon_new_tensor_3d(
        struct dragon_context * ctx,
        enum   data_type type,
        int    ne0,
        int    ne1,
        int    ne2) {
    const int ne[3] = { ne0, ne1, ne2 };
    return dragon_new_tensor(ctx, type, 3, ne);
}

struct dragon_tensor * dragon_new_tensor_4d(
        struct dragon_context * ctx,
        enum   data_type type,
        int    ne0,
        int    ne1,
        int    ne2,
        int    ne3) {
    const int ne[4] = { ne0, ne1, ne2, ne3 };
    return dragon_new_tensor(ctx, type, 4, ne);
}

struct dragon_tensor * dragon_new_i32(struct dragon_context * ctx, int32_t value) {
    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;

    struct dragon_tensor * result = dragon_new_tensor_1d(ctx, DATA_TYPE_I32, 1);

    ctx->scratch = ctx->scratch_save;

    dragon_set_i32(result, value);

    return result;
}

struct dragon_tensor * dragon_new_f32(struct dragon_context * ctx, float value) {
    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;

    struct dragon_tensor * result = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, 1);

    ctx->scratch = ctx->scratch_save;

    dragon_set_f32(result, value);

    return result;
}

struct dragon_tensor * dragon_dup_tensor(struct dragon_context * ctx, const struct dragon_tensor * src) {
    return dragon_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, NULL);
}

struct dragon_tensor * dragon_set_zero(struct dragon_tensor * tensor) {
    memset(tensor->data, 0, dragon_nbytes(tensor));
    return tensor;
}

struct dragon_tensor * dragon_set_i32 (struct dragon_tensor * tensor, int32_t value) {
    const int n     = dragon_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case DATA_TYPE_Q4_0:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_Q4_1:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(dragon_fp16_t));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_f16(nc, (dragon_fp16_t *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }

    return tensor;
}

struct dragon_tensor * dragon_set_f32(struct dragon_tensor * tensor, float value) {
    const int n     = dragon_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case DATA_TYPE_Q4_0:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_Q4_1:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(dragon_fp16_t));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_f16(nc, (dragon_fp16_t *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    dragon_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }

    return tensor;
}

int32_t dragon_get_i32_1d(const struct dragon_tensor * tensor, int i) {
    switch (tensor->type) {
        case DATA_TYPE_Q4_0:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_Q4_1:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_I8:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            } break;
        case DATA_TYPE_I16:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            } break;
        case DATA_TYPE_I32:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            } break;
        case DATA_TYPE_F16:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(dragon_fp16_t));
                return DRAGON_FP16_TO_FP32(((dragon_fp16_t *)(tensor->data))[i]);
            } break;
        case DATA_TYPE_F32:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            } break;
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }

    return 0.0f;
}

void dragon_set_i32_1d(const struct dragon_tensor * tensor, int i, int32_t value) {
    switch (tensor->type) {
        case DATA_TYPE_Q4_0:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_Q4_1:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_I8:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case DATA_TYPE_I16:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case DATA_TYPE_I32:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case DATA_TYPE_F16:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(dragon_fp16_t));
                ((dragon_fp16_t *)(tensor->data))[i] = DRAGON_FP32_TO_FP16(value);
            } break;
        case DATA_TYPE_F32:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

float dragon_get_f32_1d(const struct dragon_tensor * tensor, int i) {
    switch (tensor->type) {
        case DATA_TYPE_Q4_0:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_Q4_1:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_I8:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            } break;
        case DATA_TYPE_I16:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            } break;
        case DATA_TYPE_I32:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            } break;
        case DATA_TYPE_F16:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(dragon_fp16_t));
                return DRAGON_FP16_TO_FP32(((dragon_fp16_t *)(tensor->data))[i]);
            } break;
        case DATA_TYPE_F32:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            } break;
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }

    return 0.0f;
}

void dragon_set_f32_1d(const struct dragon_tensor * tensor, int i, float value) {
    switch (tensor->type) {
        case DATA_TYPE_Q4_0:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_Q4_1:
            {
                DRAGON_ASSERT(false);
            } break;
        case DATA_TYPE_I8:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case DATA_TYPE_I16:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case DATA_TYPE_I32:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case DATA_TYPE_F16:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(dragon_fp16_t));
                ((dragon_fp16_t *)(tensor->data))[i] = DRAGON_FP32_TO_FP16(value);
            } break;
        case DATA_TYPE_F32:
            {
                DRAGON_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

void * dragon_get_data(const struct dragon_tensor * tensor) {
    return tensor->data;
}

float * dragon_get_data_f32(const struct dragon_tensor * tensor) {
    assert(tensor->type == DATA_TYPE_F32);
    return (float *)(tensor->data);
}

struct dragon_tensor * dragon_view_tensor(
        struct dragon_context * ctx,
        const struct dragon_tensor * src) {
    return dragon_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src->data);
}

////////////////////////////////////////////////////////////////////////////////

// dragon_dup

struct dragon_tensor * dragon_dup_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_DUP;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_dup(
        struct dragon_context * ctx,
        struct dragon_tensor * a) {
    return dragon_dup_impl(ctx, a, false);
}

struct dragon_tensor * dragon_dup_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor * a) {
    return dragon_dup_impl(ctx, a, true);
}

// dragon_add

struct dragon_tensor * dragon_add_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b,
        bool inplace) {
    DRAGON_ASSERT(dragon_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_ADD;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct dragon_tensor * dragon_add(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    return dragon_add_impl(ctx, a, b, false);
}

struct dragon_tensor * dragon_add_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    return dragon_add_impl(ctx, a, b, true);
}

// dragon_sub

struct dragon_tensor * dragon_sub_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b,
        bool inplace) {
    DRAGON_ASSERT(dragon_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_SUB;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct dragon_tensor * dragon_sub(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    return dragon_sub_impl(ctx, a, b, false);
}

struct dragon_tensor * dragon_sub_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    return dragon_sub_impl(ctx, a, b, true);
}

// dragon_mul

struct dragon_tensor * dragon_mul_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b,
        bool inplace) {
    DRAGON_ASSERT(dragon_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    if (inplace) {
        DRAGON_ASSERT(is_node == false);
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_MUL;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct dragon_tensor * dragon_mul(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b) {
    return dragon_mul_impl(ctx, a, b, false);
}

struct dragon_tensor * dragon_mul_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b) {
    return dragon_mul_impl(ctx, a, b, true);
}

// dragon_div

struct dragon_tensor * dragon_div_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b,
        bool inplace) {
    DRAGON_ASSERT(dragon_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    if (inplace) {
        DRAGON_ASSERT(is_node == false);
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_DIV;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct dragon_tensor * dragon_div(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b) {
    return dragon_div_impl(ctx, a, b, false);
}

struct dragon_tensor * dragon_div_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b) {
    return dragon_div_impl(ctx, a, b, true);
}

// dragon_sqr

struct dragon_tensor * dragon_sqr_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_SQR;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_sqr(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_sqr_impl(ctx, a, false);
}

struct dragon_tensor * dragon_sqr_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_sqr_impl(ctx, a, true);
}

// dragon_sqrt

struct dragon_tensor * dragon_sqrt_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_SQRT;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_sqrt(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_sqrt_impl(ctx, a, false);
}

struct dragon_tensor * dragon_sqrt_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_sqrt_impl(ctx, a, true);
}

// dragon_sum

struct dragon_tensor * dragon_sum(
        struct dragon_context * ctx,
        struct dragon_tensor * a) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct dragon_tensor * result = dragon_new_tensor_1d(ctx, a->type, 1);

    result->op   = DRAGON_OP_SUM;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// dragon_mean

struct dragon_tensor * dragon_mean(
        struct dragon_context * ctx,
        struct dragon_tensor * a) {
    bool is_node = false;

    if (a->grad) {
        DRAGON_ASSERT(false); // TODO: implement
        is_node = true;
    }

    int ne[DRAGON_MAX_DIMS] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    struct dragon_tensor * result = dragon_new_tensor(ctx, DATA_TYPE_F32, a->n_dims, ne);

    result->op   = DRAGON_OP_MEAN;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// dragon_repeat

struct dragon_tensor * dragon_repeat(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    DRAGON_ASSERT(dragon_can_repeat(a, b));

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    if (dragon_are_same_shape(a, b) && !is_node) {
        return a;
    }

    struct dragon_tensor * result = dragon_new_tensor(ctx, a->type, b->n_dims, b->ne);

    result->op   = DRAGON_OP_REPEAT;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// dragon_abs

struct dragon_tensor * dragon_abs_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_ABS;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_abs(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_abs_impl(ctx, a, false);
}

struct dragon_tensor * dragon_abs_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_abs_impl(ctx, a, true);
}


// dragon_sgn

struct dragon_tensor * dragon_sgn_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_SGN;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_sgn(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_sgn_impl(ctx, a, false);
}

struct dragon_tensor * dragon_sgn_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_sgn_impl(ctx, a, true);
}

// dragon_neg

struct dragon_tensor * dragon_neg_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_NEG;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_neg(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_neg_impl(ctx, a, false);
}

struct dragon_tensor * dragon_neg_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_neg_impl(ctx, a, true);
}

// dragon_step

struct dragon_tensor * dragon_step_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_STEP;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_step(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_step_impl(ctx, a, false);
}

struct dragon_tensor * dragon_step_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_step_impl(ctx, a, true);
}

// dragon_relu

struct dragon_tensor * dragon_relu_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_RELU;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_relu(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_relu_impl(ctx, a, false);
}

struct dragon_tensor * dragon_relu_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_relu_impl(ctx, a, true);
}

// dragon_gelu

struct dragon_tensor * dragon_gelu_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_GELU;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_gelu(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_gelu_impl(ctx, a, false);
}

struct dragon_tensor * dragon_gelu_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_gelu_impl(ctx, a, true);
}

// dragon_silu

struct dragon_tensor * dragon_silu_impl(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_SILU;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_silu(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_silu_impl(ctx, a, false);
}

struct dragon_tensor * dragon_silu_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_silu_impl(ctx, a, true);
}

// dragon_norm

struct dragon_tensor * dragon_norm_impl(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_NORM;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store epsilon here?

    return result;
}

struct dragon_tensor * dragon_norm(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_norm_impl(ctx, a, false);
}

struct dragon_tensor * dragon_norm_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_norm_impl(ctx, a, true);
}

struct dragon_tensor * dragon_rms_norm_impl(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);

    result->op   = DRAGON_OP_RMS_NORM;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store epsilon here?

    return result;
}

struct dragon_tensor * dragon_rms_norm(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_rms_norm_impl(ctx, a, false);
}

struct dragon_tensor * dragon_rms_norm_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    return dragon_rms_norm_impl(ctx, a, true);
}

// dragon_mul_mat

struct dragon_tensor * dragon_mul_mat(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b) {
    DRAGON_ASSERT(dragon_can_mul_mat(a, b));

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    const int ne[4] = { a->ne[1], b->ne[1], a->ne[2], b->ne[3] };
    struct dragon_tensor * result = dragon_new_tensor(ctx, DATA_TYPE_F32, MIN(a->n_dims, b->n_dims), ne);

    result->op   = DRAGON_OP_MUL_MAT;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// dragon_scale

struct dragon_tensor * dragon_scale_impl(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b,
        bool inplace) {
    DRAGON_ASSERT(dragon_is_scalar(b));
    DRAGON_ASSERT(dragon_is_padded_1d(a));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);
    struct dragon_tensor * result = dragon_view_tensor(ctx, a);

    result->op   = DRAGON_OP_SCALE;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct dragon_tensor * dragon_scale(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    return dragon_scale_impl(ctx, a, b, false);
}

struct dragon_tensor * dragon_scale_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    return dragon_scale_impl(ctx, a, b, true);
}

// dragon_cpy

struct dragon_tensor * dragon_cpy_impl(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b,
        bool inplace) {
    DRAGON_ASSERT(dragon_nelements(a) == dragon_nelements(b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // make a view of the destination
    struct dragon_tensor * result = dragon_view_tensor(ctx, b);

    result->op   = DRAGON_OP_CPY;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct dragon_tensor * dragon_cpy(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    return dragon_cpy_impl(ctx, a, b, false);
}

struct dragon_tensor * dragon_cpy_inplace(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    return dragon_cpy_impl(ctx, a, b, true);
}

// dragon_reshape

struct dragon_tensor * dragon_reshape(
        struct dragon_context * ctx,
        struct dragon_tensor * a,
        struct dragon_tensor * b) {
    DRAGON_ASSERT(dragon_is_contiguous(a));
    DRAGON_ASSERT(dragon_is_contiguous(b));
    DRAGON_ASSERT(dragon_nelements(a) == dragon_nelements(b));

    bool is_node = false;

    if (a->grad || b->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct dragon_tensor * result = dragon_new_tensor_impl(ctx, a->type, b->n_dims, b->ne, a->data);

    result->op   = DRAGON_OP_RESHAPE;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_reshape_2d(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   ne0,
        int                   ne1) {
    DRAGON_ASSERT(dragon_is_contiguous(a));
    DRAGON_ASSERT(dragon_nelements(a) == ne0*ne1);

    bool is_node = false;

    if (a->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int ne[2] = { ne0, ne1 };
    struct dragon_tensor * result = dragon_new_tensor_impl(ctx, a->type, 2, ne, a->data);

    result->op   = DRAGON_OP_RESHAPE;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct dragon_tensor * dragon_reshape_3d(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2) {
    DRAGON_ASSERT(dragon_is_contiguous(a));
    DRAGON_ASSERT(dragon_nelements(a) == ne0*ne1*ne2);

    bool is_node = false;

    if (a->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int ne[3] = { ne0, ne1, ne2 };
    struct dragon_tensor * result = dragon_new_tensor_impl(ctx, a->type, 3, ne, a->data);

    result->op   = DRAGON_OP_RESHAPE;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// dragon_view_1d

struct dragon_tensor * dragon_view_1d(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   ne0,
        size_t                offset) {
    if (a->grad) {
        DRAGON_ASSERT(false); // gradient propagation is not supported
    }

    struct dragon_tensor * result = dragon_new_tensor_impl(ctx, a->type, 1, &ne0, (char *) a->data + offset);

    result->op   = DRAGON_OP_VIEW;
    result->grad = NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the offset here?

    return result;
}

// dragon_view_2d

struct dragon_tensor * dragon_view_2d(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   ne0,
        int                   ne1,
        size_t                nb1,
        size_t                offset) {
    if (a->grad) {
        DRAGON_ASSERT(false); // gradient propagation is not supported
    }

    const int ne[DRAGON_MAX_DIMS] = { ne0, ne1, 1, 1 };

    struct dragon_tensor * result = dragon_new_tensor_impl(ctx, a->type, 2, ne, (char *) a->data + offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    result->op   = DRAGON_OP_VIEW;
    result->grad = NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the offset here?

    return result;
}

// dragon_permute

struct dragon_tensor * dragon_permute(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3) {
    DRAGON_ASSERT(axis0 >= 0 && axis0 < DRAGON_MAX_DIMS);
    DRAGON_ASSERT(axis1 >= 0 && axis1 < DRAGON_MAX_DIMS);
    DRAGON_ASSERT(axis2 >= 0 && axis2 < DRAGON_MAX_DIMS);
    DRAGON_ASSERT(axis3 >= 0 && axis3 < DRAGON_MAX_DIMS);

    DRAGON_ASSERT(axis0 != axis1);
    DRAGON_ASSERT(axis0 != axis2);
    DRAGON_ASSERT(axis0 != axis3);
    DRAGON_ASSERT(axis1 != axis2);
    DRAGON_ASSERT(axis1 != axis3);
    DRAGON_ASSERT(axis2 != axis3);

    bool is_node = false;

    if (a->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct dragon_tensor * result = dragon_view_tensor(ctx, a);

    int ne[DRAGON_MAX_DIMS];
    int nb[DRAGON_MAX_DIMS];

    ne[axis0] = a->ne[0];
    ne[axis1] = a->ne[1];
    ne[axis2] = a->ne[2];
    ne[axis3] = a->ne[3];

    nb[axis0] = a->nb[0];
    nb[axis1] = a->nb[1];
    nb[axis2] = a->nb[2];
    nb[axis3] = a->nb[3];

    result->ne[0] = ne[0];
    result->ne[1] = ne[1];
    result->ne[2] = ne[2];
    result->ne[3] = ne[3];

    result->nb[0] = nb[0];
    result->nb[1] = nb[1];
    result->nb[2] = nb[2];
    result->nb[3] = nb[3];

    result->op   = DRAGON_OP_PERMUTE;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the permutation here?

    return result;
}

// dragon_transpose

struct dragon_tensor * dragon_transpose(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    bool is_node = false;

    if (a->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct dragon_tensor * result = dragon_view_tensor(ctx, a);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op   = DRAGON_OP_TRANSPOSE;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// dragon_get_rows

struct dragon_tensor * dragon_get_rows(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b) {
    DRAGON_ASSERT(dragon_is_matrix(a) && dragon_is_vector(b) && b->type == DATA_TYPE_I32);

    bool is_node = false;

    if (a->grad || b->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: implement non F32 return
    //struct dragon_tensor * result = dragon_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    struct dragon_tensor * result = dragon_new_tensor_2d(ctx, DATA_TYPE_F32, a->ne[0], b->ne[0]);

    result->op   = DRAGON_OP_GET_ROWS;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// dragon_diag_mask_inf

struct dragon_tensor * dragon_diag_mask_inf(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   n_past) {
    bool is_node = false;

    if (a->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);
    struct dragon_tensor * result = dragon_view_tensor(ctx, a);
    struct dragon_tensor * b = dragon_new_i32(ctx, n_past);

    result->op   = DRAGON_OP_DIAG_MASK_INF;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// dragon_soft_max

struct dragon_tensor * dragon_soft_max(
        struct dragon_context * ctx,
        struct dragon_tensor  * a) {
    bool is_node = false;

    if (a->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);
    struct dragon_tensor * result = dragon_view_tensor(ctx, a);

    result->op   = DRAGON_OP_SOFT_MAX;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// dragon_rope

struct dragon_tensor * dragon_rope(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   n_past,
        int                   n_dims,
        int                   mode) {
    DRAGON_ASSERT(n_past >= 0);
    bool is_node = false;

    if (a->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct dragon_tensor * result = inplace ? dragon_view_tensor(ctx, a) : dragon_dup_tensor(ctx, a);
    struct dragon_tensor * result = dragon_view_tensor(ctx, a);

    struct dragon_tensor * b = dragon_new_tensor_1d(ctx, DATA_TYPE_I32, 3);
    ((int32_t *) b->data)[0] = n_past;
    ((int32_t *) b->data)[1] = n_dims;
    ((int32_t *) b->data)[2] = mode;

    result->op   = DRAGON_OP_ROPE;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// dragon_conv_1d_1s

struct dragon_tensor * dragon_conv_1d_1s(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b) {
    DRAGON_ASSERT(dragon_is_matrix(b));
    DRAGON_ASSERT(a->ne[1] == b->ne[1]);
    DRAGON_ASSERT(a->ne[3] == 1);
    bool is_node = false;

    if (a->grad || b->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int ne[4] = { b->ne[0], a->ne[2], 1, 1, };
    struct dragon_tensor * result = dragon_new_tensor(ctx, DATA_TYPE_F32, 2, ne);

    result->op   = DRAGON_OP_CONV_1D_1S;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// dragon_conv_1d_2s

struct dragon_tensor * dragon_conv_1d_2s(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b) {
    DRAGON_ASSERT(dragon_is_matrix(b));
    DRAGON_ASSERT(a->ne[1] == b->ne[1]);
    DRAGON_ASSERT(a->ne[3] == 1);
    bool is_node = false;

    if (a->grad || b->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int ne[4] = { b->ne[0]/2, a->ne[2], 1, 1, };
    struct dragon_tensor * result = dragon_new_tensor(ctx, DATA_TYPE_F32, 2, ne);

    result->op   = DRAGON_OP_CONV_1D_2S;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// dragon_flash_attn

struct dragon_tensor * dragon_flash_attn(
        struct dragon_context * ctx,
        struct dragon_tensor  * q,
        struct dragon_tensor  * k,
        struct dragon_tensor  * v,
        bool                  masked) {
    DRAGON_ASSERT(dragon_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    bool is_node = false;

    if (q->grad || k->grad || v->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    //struct dragon_tensor * result = dragon_dup_tensor(ctx, q);
    struct dragon_tensor * result = dragon_new_tensor(ctx, DATA_TYPE_F32, 4, q->ne);

    result->op   = DRAGON_OP_FLASH_ATTN;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = q;
    result->src1 = k;
    result->opt[0] = v;
    result->opt[1] = dragon_new_i32(ctx, masked ? 1 : 0);

    return result;
}

// dragon_flash_ff

struct dragon_tensor * dragon_flash_ff(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b0,
        struct dragon_tensor  * b1,
        struct dragon_tensor  * c0,
        struct dragon_tensor  * c1) {
    DRAGON_ASSERT(dragon_can_mul_mat(b0, a));
    // TODO: more checks

    bool is_node = false;

    if (a->grad || b0->grad || b1->grad || c0->grad || c1->grad) {
        DRAGON_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    //struct dragon_tensor * result = dragon_dup_tensor(ctx, a);
    struct dragon_tensor * result = dragon_new_tensor(ctx, DATA_TYPE_F32, 4, a->ne);

    result->op   = DRAGON_OP_FLASH_FF;
    result->grad = is_node ? dragon_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b0;
    result->opt[0] = b1;
    result->opt[1] = c0;
    result->opt[2] = c1;

    return result;
}

////////////////////////////////////////////////////////////////////////////////

void dragon_set_param(
        struct dragon_context * ctx,
        struct dragon_tensor * tensor) {
    tensor->is_param = true;

    DRAGON_ASSERT(tensor->grad == NULL);
    tensor->grad = dragon_dup_tensor(ctx, tensor);
}

// dragon_compute_forward_dup

static void dragon_compute_forward_dup_f16(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    DRAGON_ASSERT(params->ith == 0);
    DRAGON_ASSERT(dragon_is_contiguous(dst));
    DRAGON_ASSERT(dragon_nelements(dst) == dragon_nelements(src0));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    if (dragon_is_contiguous(src0) && src0->type == dst->type) {
        memcpy(dst->data, src0->data, dragon_nelements(dst) * DRAGON_TYPE_SIZE[src0->type]);
        return;
    }

    if (src0->nb[0] == sizeof(dragon_fp16_t)) {
        if (dst->type == DATA_TYPE_F16) {
            int id = 0;
            const size_t rs = ne00*nb00;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        char * dst_ptr = (char *) dst->data + id*rs;

                        memcpy(dst_ptr, src0_ptr, rs);

                        id++;
                    }
                }
            }
        } else if (dst->type == DATA_TYPE_F32) {
            int id = 0;
            float * dst_ptr = (float *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const dragon_fp16_t * src0_ptr = (dragon_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = DRAGON_FP16_TO_FP32(*src0_ptr);
                            id++;
                        }
                    }
                }
            }
        } else {
            DRAGON_ASSERT(false); // TODO: implement
        }
    } else {
        //printf("%s: this is not optimal - fix me\n", __func__);

        if (dst->type == DATA_TYPE_F32) {
            int id = 0;
            float * dst_ptr = (float *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const dragon_fp16_t * src0_ptr = (dragon_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = DRAGON_FP16_TO_FP32(*src0_ptr);
                            id++;
                        }
                    }
                }
            }
        } else if (dst->type == DATA_TYPE_F16) {
            int id = 0;
            dragon_fp16_t * dst_ptr = (dragon_fp16_t *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const dragon_fp16_t * src0_ptr = (dragon_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = *src0_ptr;
                            id++;
                        }
                    }
                }
            }
        } else {
            DRAGON_ASSERT(false); // TODO: implement
        }
    }
}

static void dragon_compute_forward_dup_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    DRAGON_ASSERT(params->ith == 0);
    DRAGON_ASSERT(dragon_is_contiguous(dst));
    DRAGON_ASSERT(dragon_nelements(dst) == dragon_nelements(src0));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    if (dragon_is_contiguous(src0) && src0->type == dst->type) {
        memcpy(dst->data, src0->data, dragon_nelements(dst) * DRAGON_TYPE_SIZE[src0->type]);
        return;
    }

    if (src0->nb[0] == sizeof(float)) {
        if (dst->type == DATA_TYPE_F32) {
            int id = 0;
            const size_t rs = ne00*nb00;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        char * dst_ptr = (char *) dst->data + id*rs;

                        memcpy(dst_ptr, src0_ptr, rs);

                        id++;
                    }
                }
            }
        } else if (dst->type == DATA_TYPE_F16) {
            int id = 0;
            dragon_fp16_t * dst_ptr = (dragon_fp16_t *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = DRAGON_FP32_TO_FP16(*src0_ptr);
                            id++;
                        }
                    }
                }
            }
        } else {
            DRAGON_ASSERT(false); // TODO: implement
        }
    } else {
        //printf("%s: this is not optimal - fix me\n", __func__);

        if (dst->type == DATA_TYPE_F32) {
            int id = 0;
            float * dst_ptr = (float *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = *src0_ptr;
                            id++;
                        }
                    }
                }
            }
        } else if (dst->type == DATA_TYPE_F16) {
            int id = 0;
            dragon_fp16_t * dst_ptr = (dragon_fp16_t *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = DRAGON_FP32_TO_FP16(*src0_ptr);
                            id++;
                        }
                    }
                }
            }
        } else {
            DRAGON_ASSERT(false); // TODO: implement
        }
    }
}

static void dragon_compute_forward_dup(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F16:
            {
                dragon_compute_forward_dup_f16(params, src0, dst);
            } break;
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_dup_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_add

static void dragon_compute_forward_add_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    DRAGON_ASSERT(dragon_are_same_shape(src0, src1) && dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb10 = src1->nb[0];
    const size_t nb11 = src1->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    DRAGON_ASSERT( nb0 == sizeof(float));
    DRAGON_ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        const int j0 = (n/nth)*ith;
        const int j1 = ith == nth - 1 ? n : (n/nth)*(ith + 1);

        for (int j = j0; j < j1; j++) {
            dragon_vec_add_f32(nc,
                    (float *) ((char *) dst->data  + j*nb1),
                    (float *) ((char *) src0->data + j*nb01),
                    (float *) ((char *) src1->data + j*nb11));
        }
    } else {
        // src1 is not contiguous
        for (int j = ith; j < n; j += nth) {
            float * dst_ptr  = (float *) ((char *) dst->data  + j*nb1);
            float * src0_ptr = (float *) ((char *) src0->data + j*nb01);
            for (int i = 0; i < nc; i++) {
                float * src1_ptr = (float *) ((char *) src1->data + j*nb11 + i*nb10);

                dst_ptr[i] = src0_ptr[i] + *src1_ptr;
            }
        }
    }
}

static void dragon_compute_forward_add(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_add_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_sub

static void dragon_compute_forward_sub_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, src1) && dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_sub_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void dragon_compute_forward_sub(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_sub_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_mul

static void dragon_compute_forward_mul_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, src1) && dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_mul_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void dragon_compute_forward_mul(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_mul_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_div

static void dragon_compute_forward_div_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, src1) && dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_div_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void dragon_compute_forward_div(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_div_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_sqr

static void dragon_compute_forward_sqr_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n     = dragon_nrows(src0);
    const int nc    = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_sqr_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void dragon_compute_forward_sqr(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_sqr_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_sqrt

static void dragon_compute_forward_sqrt_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_sqrt_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void dragon_compute_forward_sqrt(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_sqrt_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_sum

static void dragon_compute_forward_sum_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_is_scalar(dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    assert(dragon_is_scalar(dst));
    assert(src0->nb[0] == sizeof(float));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    for (int i03 = 0; i03 < ne03; i03++) {
        for (int i02 = 0; i02 < ne02; i02++) {
            for (int i01 = 0; i01 < ne01; i01++) {
                dragon_vec_sum_f32(ne00,
                        (float *) (dst->data),
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
            }
        }
    }
}

static void dragon_compute_forward_sum(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_sum_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_mean

static void dragon_compute_forward_mean_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    const int ne3 = dst->ne[3];

    assert(ne0 == 1);
    assert(ne1 == ne01);
    assert(ne2 == ne02);
    assert(ne3 == ne03);

    UNUSED(ne0);
    UNUSED(ne1);
    UNUSED(ne2);
    UNUSED(ne3);

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    for (int i03 = 0; i03 < ne03; i03++) {
        for (int i02 = 0; i02 < ne02; i02++) {
            for (int i01 = 0; i01 < ne01; i01++) {
                dragon_vec_sum_f32(ne00,
                        (float *) ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (float) ne00;
            }
        }
    }
}

static void dragon_compute_forward_mean(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_mean_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_repeat

static void dragon_compute_forward_repeat_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_can_repeat(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // TODO: implement support for rank > 2 tensors
    assert(src0->ne[2] == 1);
    assert(src0->ne[3] == 1);
    assert( dst->ne[2] == 1);
    assert( dst->ne[3] == 1);

    const int nc  = dst->ne[0];
    const int nr  = dst->ne[1];
    const int nc0 = src0->ne[0];
    const int nr0 = src0->ne[1];
    const int ncr = nc/nc0; // guaranteed to be an integer due to the check in dragon_can_repeat
    const int nrr = nr/nr0; // guaranteed to be an integer due to the check in dragon_can_repeat

    // TODO: support for transposed / permuted tensors
    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    // TODO: maybe this is not optimal?
    for (int i = 0; i < nrr; i++) {
        for (int j = 0; j < ncr; j++) {
            for (int k = 0; k < nr0; k++) {
                dragon_vec_cpy_f32(nc0,
                        (float *) ((char *)  dst->data + (i*nr0 + k)*( dst->nb[1]) + j*nc0*( dst->nb[0])),
                        (float *) ((char *) src0->data + (        k)*(src0->nb[1])));
            }
        }
    }
}

static void dragon_compute_forward_repeat(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_repeat_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_abs

static void dragon_compute_forward_abs_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_abs_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void dragon_compute_forward_abs(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_abs_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_sgn

static void dragon_compute_forward_sgn_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_sgn_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void dragon_compute_forward_sgn(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_sgn_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_neg

static void dragon_compute_forward_neg_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_neg_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void dragon_compute_forward_neg(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_neg_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_step

static void dragon_compute_forward_step_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_step_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void dragon_compute_forward_step(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_step_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_relu

static void dragon_compute_forward_relu_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        dragon_vec_relu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void dragon_compute_forward_relu(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_relu_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_gelu

static void dragon_compute_forward_gelu_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    DRAGON_ASSERT(dragon_is_contiguous(src0));
    DRAGON_ASSERT(dragon_is_contiguous(dst));
    DRAGON_ASSERT(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = dragon_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        dragon_vec_gelu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void dragon_compute_forward_gelu(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_gelu_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }

    //printf("XXXXXXXX gelu\n");
}

// dragon_compute_forward_silu

static void dragon_compute_forward_silu_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    DRAGON_ASSERT(dragon_is_contiguous(src0));
    DRAGON_ASSERT(dragon_is_contiguous(dst));
    DRAGON_ASSERT(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = dragon_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        dragon_vec_silu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void dragon_compute_forward_silu(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_silu_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}


// dragon_compute_forward_norm

static void dragon_compute_forward_norm_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    DRAGON_ASSERT(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    DRAGON_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const dragon_float eps = 1e-5f; // TODO: make this a parameter

    // TODO: optimize
    for (int i03 = 0; i03 < ne03; i03++) {
        for (int i02 = 0; i02 < ne02; i02++) {
            for (int i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                dragon_float mean = 0.0;
                for (int i00 = 0; i00 < ne00; i00++) {
                    mean += x[i00];
                }

                mean /= ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                dragon_float sum2 = 0.0;
                for (int i00 = 0; i00 < ne00; i00++) {
                    dragon_float v = x[i00] - mean;
                    y[i00] = v;
                    sum2 += v*v;
                }

                const float scale = 1.0/sqrt(sum2/ne00 + eps);

                dragon_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void dragon_compute_forward_norm(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_norm_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

static void dragon_compute_forward_rms_norm_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    DRAGON_ASSERT(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    DRAGON_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const dragon_float eps = 1e-5f; // TODO: make this a parameter

    // TODO: optimize
    for (int i03 = 0; i03 < ne03; i03++) {
        for (int i02 = 0; i02 < ne02; i02++) {
            for (int i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                dragon_float mean = 0.0;
                for (int i00 = 0; i00 < ne00; i00++) {
                    mean += x[i00] * x[i00];
                }

                mean /= ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);
                
                memcpy(y, x, ne00 * sizeof(float));
                // for (int i00 = 0; i00 < ne00; i00++) {
                //     y[i00] = x[i00];
                // }

                const float scale = 1.0/sqrt(mean + eps);

                dragon_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void dragon_compute_forward_rms_norm(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_rms_norm_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}


// dragon_compute_forward_mul_mat

#if defined(DRAGON_USE_ACCELERATE) || defined(DRAGON_USE_OPENBLAS)
// helper function to determine if it is better to use BLAS or not
// for large matrices, BLAS is faster
static bool dragon_compute_forward_mul_mat_use_blas(
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    UNUSED(src0);

    const int ne10 = src1->ne[0];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if (dragon_is_contiguous(src0) &&
        dragon_is_contiguous(src1) && ((ne0 >= 32 && ne1 >= 32 && ne10 >= 32))) {
        //printf("BLAS: %d %d %d\n", ne0, ne1, ne10);
        return true;
    }

    return false;
}
#endif

static void dragon_compute_forward_mul_mat_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];
    const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    assert(ne02 == ne12);
    assert(ne03 == ne13);
    assert(ne2  == ne12);
    assert(ne3  == ne13);

    // TODO: we don't support permuted src0
    assert(nb00 == sizeof(float) || nb01 == sizeof(float));

    // dst cannot be transposed or permuted
    assert(nb0 == sizeof(float));
    assert(nb0 <= nb1);
    assert(nb1 <= nb2);
    assert(nb2 <= nb3);

    assert(ne0 == ne01);
    assert(ne1 == ne11);
    assert(ne2 == ne02);
    assert(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
    //
    // nb00 <  nb01 - src0 is transposed
    //   compute by src0 columns

#if defined(DRAGON_USE_ACCELERATE) || defined(DRAGON_USE_OPENBLAS)
    if (dragon_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        DRAGON_ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == DRAGON_TASK_INIT) {
            return;
        }

        if (params->type == DRAGON_TASK_FINALIZE) {
            return;
        }

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                const float * x = (float *) (src0->data);
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // zT = y * xT
                {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ne11, ne01, ne10,
                            1.0f,    y, ne10,
                                     x, ne10,
                            0.0f,    d, ne01);
                }
            }
        }

        //printf("CBLAS F32 = %f ms, %d x %d x %d x %d\n", (dragon_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

        return;
    }
#endif

    if (params->type == DRAGON_TASK_INIT) {
        if (nb01 >= nb00) {
            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);
        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        if (nb01 >= nb00) {
            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        //assert(params->wsize == (dragon_nbytes(dst) + CACHE_LINE_SIZE)*nth);

        float * const wdata = params->wdata;

        // cols per thread
        const int dc = (ne + nth - 1)/nth;

        // col range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, ne);

        dragon_vec_cpy_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + ic0);

        for (int k = 1; k < nth; k++) {
            dragon_vec_acc_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + (ne + CACHE_LINE_SIZE_F32)*k + ic0);
        }

        return;
    }

    if (nb01 >= nb00) {
        // TODO: do not support transposed src1
        assert(nb10 == sizeof(float));

        // parallelize by src0 rows using dragon_vec_dot_f32

        // total rows in src0
        const int nr = ne01*ne02*ne03;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        for (int ir = ir0; ir < ir1; ++ir) {
            // src0 indices
            const int i03 = ir/(ne02*ne01);
            const int i02 = (ir - i03*ne02*ne01)/ne01;
            const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

            for (int ic = 0; ic < ne11; ++ic) {
                // src1 indices
                const int i13 = i03;
                const int i12 = i02;
                const int i11 = ic;

                // dst indices
                const int i0 = i01;
                const int i1 = i11;
                const int i2 = i02;
                const int i3 = i03;

                dragon_vec_dot_f32(ne00,
                        (float *) ((char *)  dst->data + (i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3)),
                        (float *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03)),
                        (float *) ((char *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13)));
            }
        }
    } else {
        // parallelize by src1 columns using dragon_vec_mad_f32
        // each thread has its own work data
        // during FINALIZE we accumulate all work data into dst

        // total columns in src1
        const int nc = ne10;

        // columns per thread
        const int dc = (nc + nth - 1)/nth;

        // column range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, nc);

        // work data for thread
        const int wo = (ne + CACHE_LINE_SIZE_F32)*ith;
        float * const wdata = params->wdata;

        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                for (int i11 = 0; i11 < ne11; ++i11) {
                    for (int ic = ic0; ic < ic1; ++ic) {
                        // src1 indices
                        const int i10 = ic;

                        // src0 indices
                        const int i03 = i13;
                        const int i02 = i12;
                        const int i00 = ic;

                        // dst indices
                        const int i1 = i11;
                        const int i2 = i12;
                        const int i3 = i13;

                        assert(sizeof(float)*(wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + ne01) <= params->wsize);

                        dragon_vec_mad_f32(ne01,
                                (float *) (wdata + wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0),
                                (float *) ((char *) src0->data + (i00*nb00 + i02*nb02 + i03*nb03)),
                               *(float *) ((char *) src1->data + (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13)));
                    }
                }
            }
        }
    }

    //int64_t t1 = dragon_perf_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);
    //    printf("nb10 = %5d, nb11 = %5d, nb12 = %5d, nb13 = %5d\n", nb10, nb11, nb12, nb13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}

static void dragon_compute_forward_mul_mat_f16_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];
    const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    DRAGON_ASSERT(ne02 == ne12);
    DRAGON_ASSERT(ne03 == ne13);
    DRAGON_ASSERT(ne2  == ne12);
    DRAGON_ASSERT(ne3  == ne13);

    // TODO: we don't support permuted src0
    DRAGON_ASSERT(nb00 == sizeof(dragon_fp16_t) || nb01 == sizeof(dragon_fp16_t));

    // dst cannot be transposed or permuted
    DRAGON_ASSERT(nb0 == sizeof(float));
    DRAGON_ASSERT(nb0 <= nb1);
    DRAGON_ASSERT(nb1 <= nb2);
    DRAGON_ASSERT(nb2 <= nb3);

    DRAGON_ASSERT(ne0 == ne01);
    DRAGON_ASSERT(ne1 == ne11);
    DRAGON_ASSERT(ne2 == ne02);
    DRAGON_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
    //
    // nb00 <  nb01 - src0 is transposed
    //   compute by src0 columns

#if defined(DRAGON_USE_ACCELERATE) || defined(DRAGON_USE_OPENBLAS)
    if (dragon_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        DRAGON_ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == DRAGON_TASK_INIT) {
            return;
        }

        if (params->type == DRAGON_TASK_FINALIZE) {
            return;
        }

        float * const wdata = params->wdata;

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                {
                    int id = 0;
                    for (int i01 = 0; i01 < ne01; ++i01) {
                        for (int i00 = 0; i00 < ne00; ++i00) {
                            wdata[id++] = DRAGON_FP16_TO_FP32(*(dragon_fp16_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00));
                        }
                    }
                }

                const float * x = wdata;
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                //      float * z =                          wdata + ne00*ne01;

                // z = x * yT
                //{
                //    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                //            ne01, ne11, ne00,
                //            1.0f, x, ne00,
                //                  y, ne00,
                //            0.0f, z, ne11);
                //}

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // transpose z
                //for (int j = 0; j < ne11; ++j) {
                //    for (int i = 0; i < ne01; ++i) {
                //        d[j*ne01 + i] = z[i*ne11 + j];
                //    }
                //}

                {
#if 1
                    // zT = y * xT
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ne11, ne01, ne10,
                            1.0f,    y, ne00,
                                     x, ne00,
                            0.0f,    d, ne01);
#else
                    // zT = (xT * y)T
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            ne01, ne11, ne10,
                            1.0f,    x, ne00,
                                     y, ne00,
                            0.0f,    d, ne01);
#endif
                }
            }
        }

        /*printf("CBLAS F16 = %f ms, %d x %d x %d x %d\n", (dragon_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);*/

        return;
    }
#endif

    if (params->type == DRAGON_TASK_INIT) {
        if (nb01 >= nb00) {
            dragon_fp16_t * const wdata = params->wdata;

            int id = 0;
            for (int i13 = 0; i13 < ne13; ++i13) {
                for (int i12 = 0; i12 < ne12; ++i12) {
                    for (int i11 = 0; i11 < ne11; ++i11) {
                        for (int i10 = 0; i10 < ne10; ++i10) {
                            wdata[id++] = DRAGON_FP32_TO_FP16(*(float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10));
                        }
                    }
                }
            }

            DRAGON_ASSERT(id*sizeof(dragon_fp16_t) <= params->wsize);

            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);
        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        if (nb01 >= nb00) {
            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        //assert(params->wsize == (dragon_nbytes(dst) + CACHE_LINE_SIZE)*nth);

        dragon_fp16_t * const wdata = params->wdata;

        // cols per thread
        const int dc = (ne + nth - 1)/nth;

        // col range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, ne);

        for (int i = ic0; i < ic1; ++i) {
            ((float *) dst->data)[i] = DRAGON_FP16_TO_FP32(wdata[i]);
        }

        for (int k = 1; k < nth; k++) {
            for (int i = ic0; i < ic1; ++i) {
                ((float *) dst->data)[i] += DRAGON_FP16_TO_FP32(wdata[(ne + CACHE_LINE_SIZE_F32)*k + i]);
            }
        }

        return;
    }

    if (nb01 >= nb00) {
        // fp16 -> half the size, so divide by 2
        // TODO: do not support transposed src1
        assert(nb10/2 == sizeof(dragon_fp16_t));

        // parallelize by src0 rows using dragon_vec_dot_f16

        // total rows in src0
        const int nr = ne01*ne02*ne03;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        dragon_fp16_t * wdata = params->wdata;

        for (int ir = ir0; ir < ir1; ++ir) {
            // src0 indices
            const int i03 = ir/(ne02*ne01);
            const int i02 = (ir - i03*ne02*ne01)/ne01;
            const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int i13 = i03;
            const int i12 = i02;

            const int i0 = i01;
            const int i2 = i02;
            const int i3 = i03;

            dragon_fp16_t * src0_row = (dragon_fp16_t *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            dragon_fp16_t * src1_col =                                wdata + (       0 + i12*ne11 + i13*ne12*ne11)*ne00;

            float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

            assert(ne00 % 32 == 0);

            for (int ic = 0; ic < ne11; ++ic) {
                dragon_vec_dot_f16(ne00, &dst_col[ic*ne0], src0_row, src1_col + ic*ne00);
            }
        }
    } else {
        // parallelize by src1 columns using dragon_vec_mad_f16
        // each thread has its own work data
        // during FINALIZE we accumulate all work data into dst

        // total columns in src1
        const int nc = ne10;

        // columns per thread
        const int dc = (nc + nth - 1)/nth;

        // column range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, nc);

        // work data for thread
        const int wo = (ne + CACHE_LINE_SIZE_F32)*ith;
        dragon_fp16_t * const wdata = params->wdata;

        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                for (int i11 = 0; i11 < ne11; ++i11) {
                    // dst indices
                    const int i1 = i11;
                    const int i2 = i12;
                    const int i3 = i13;

                    dragon_fp16_t * dst_row = wdata + wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0;

                    for (int ic = ic0; ic < ic1; ++ic) {
                        // src1 indices
                        const int i10 = ic;

                        // src0 indices
                        const int i03 = i13;
                        const int i02 = i12;
                        const int i00 = ic;

                        assert(sizeof(dragon_fp16_t)*(wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + ne01) <= params->wsize);

                        dragon_fp16_t * src0_col =  (dragon_fp16_t *) ((char *) src0->data + (i00*nb00 + i02*nb02 + i03*nb03));
                        float         src1_val = *      (float *) ((char *) src1->data + (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13));

                        dragon_vec_mad_f16(ne01, dst_row, src0_col, src1_val);
                    }
                }
            }
        }
    }

    //int64_t t1 = dragon_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}

static void dragon_compute_forward_mul_mat_q4_0_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];
    const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    DRAGON_ASSERT(ne02 == ne12);
    DRAGON_ASSERT(ne03 == ne13);
    DRAGON_ASSERT(ne2  == ne12);
    DRAGON_ASSERT(ne3  == ne13);

    // TODO: we don't support permuted src0
    DRAGON_ASSERT(nb00 == (int) DRAGON_TYPE_SIZE[DATA_TYPE_Q4_0] || nb01 == (int) DRAGON_TYPE_SIZE[DATA_TYPE_Q4_0]);

    // dst cannot be transposed or permuted
    DRAGON_ASSERT(nb0 == sizeof(float));
    DRAGON_ASSERT(nb0 <= nb1);
    DRAGON_ASSERT(nb1 <= nb2);
    DRAGON_ASSERT(nb2 <= nb3);

    DRAGON_ASSERT(ne0 == ne01);
    DRAGON_ASSERT(ne1 == ne11);
    DRAGON_ASSERT(ne2 == ne02);
    DRAGON_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
    //
    // nb00 <  nb01 - src0 is transposed
    //   compute by src0 columns

#if defined(DRAGON_USE_ACCELERATE) || defined(DRAGON_USE_OPENBLAS)
    if (dragon_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        DRAGON_ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == DRAGON_TASK_INIT) {
            return;
        }

        if (params->type == DRAGON_TASK_FINALIZE) {
            return;
        }

        float * const wdata = params->wdata;

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                {
                    int id = 0;
                    for (int i01 = 0; i01 < ne01; ++i01) {
                        //for (int i00 = 0; i00 < ne00; ++i00) {
                        //    wdata[id++] = DRAGON_FP16_TO_FP32(*(dragon_fp16_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00));
                        //}
                        dequantize_row_q4_0((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01, wdata + id, ne00);
                        id += ne00;
                    }
                }

                const float * x = wdata;
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                //      float * z =                          wdata + ne00*ne01;

                // z = x * yT
                //{
                //    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                //            ne01, ne11, ne00,
                //            1.0f, x, ne00,
                //                  y, ne00,
                //            0.0f, z, ne11);
                //}

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // transpose z
                //for (int j = 0; j < ne11; ++j) {
                //    for (int i = 0; i < ne01; ++i) {
                //        d[j*ne01 + i] = z[i*ne11 + j];
                //    }
                //}

                {
#if 1
                    // zT = y * xT
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ne11, ne01, ne10,
                            1.0f,    y, ne00,
                                     x, ne00,
                            0.0f,    d, ne01);
#else
                    // zT = (xT * y)T
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            ne01, ne11, ne10,
                            1.0f,    x, ne00,
                                     y, ne00,
                            0.0f,    d, ne01);
#endif
                }
            }
        }

        /*printf("CBLAS Q4_0 = %f ms, %d x %d x %d x %d\n", (dragon_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);*/

        return;
    }
#endif

    if (params->type == DRAGON_TASK_INIT) {
        //printf("HHHHHHHHH ith = %d, nth = %d\n", ith, nth);
        if (nb01 >= nb00) {
            char * wdata = params->wdata;

            for (int i13 = 0; i13 < ne13; ++i13) {
                for (int i12 = 0; i12 < ne12; ++i12) {
                    for (int i11 = 0; i11 < ne11; ++i11) {
                        //for (int i10 = 0; i10 < ne10; ++i10) {
                        //    wdata[id++] = DRAGON_FP32_TO_FP16(*(float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10));
                        //}
                        quantize_row_q4_0((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), (void *) wdata, ne10);
                        wdata += (ne10*DRAGON_TYPE_SIZE[DATA_TYPE_Q4_0])/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_0];
                    }
                }
            }

            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);
        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        if (nb01 >= nb00) {
            return;
        }

        float * const wdata = params->wdata;

        // cols per thread
        const int dc = (ne + nth - 1)/nth;

        // col range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, ne);

        dragon_vec_cpy_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + ic0);

        for (int k = 1; k < nth; k++) {
            dragon_vec_acc_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + (ne + CACHE_LINE_SIZE_F32)*k + ic0);
        }

        return;
    }

    if (nb01 >= nb00) {
        // TODO: do not support transposed src1

        // parallelize by src0 rows using dragon_vec_dot_q4_0

        // total rows in src0
        const int nr = ne01*ne02*ne03;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        void * wdata = params->wdata;

        for (int ir = ir0; ir < ir1; ++ir) {
            // src0 indices
            const int i03 = ir/(ne02*ne01);
            const int i02 = (ir - i03*ne02*ne01)/ne01;
            const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int i13 = i03;
            const int i12 = i02;

            const int i0 = i01;
            const int i2 = i02;
            const int i3 = i03;

            void * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            char * src1_col =          ((char *)      wdata + (      (0 + i12*ne11 + i13*ne12*ne11)*ne00*DRAGON_TYPE_SIZE[DATA_TYPE_Q4_0])/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_0]);

            float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

            assert(ne00 % 32 == 0);

            for (int ic = 0; ic < ne11; ++ic) {
                dragon_vec_dot_q4_0(ne00, &dst_col[ic*ne0], src0_row, ((void *) (src1_col + (ic*ne00*DRAGON_TYPE_SIZE[DATA_TYPE_Q4_0])/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_0])));
            }
        }
    } else {
        //printf("AAAAA ith = %d, nth = %d\n", ith, nth);
        // parallelize by src1 columns using dragon_vec_mad_q4_0
        // each thread has its own work data
        // during FINALIZE we accumulate all work data into dst

        // total columns in src1
        const int nc = ne10;

        // columns per thread
        const int dc = (nc + nth - 1)/nth;

        // column range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, nc);

        // work data for thread
        const int wo = (ne + CACHE_LINE_SIZE_F32)*ith;
        float * const wdata = params->wdata;

        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                for (int i11 = 0; i11 < ne11; ++i11) {
                    // dst indices
                    const int i1 = i11;
                    const int i2 = i12;
                    const int i3 = i13;

                    float * dst_row = wdata + wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0;

                    for (int ic = ic0; ic < ic1; ++ic) {
                        // src1 indices
                        const int i10 = ic;

                        // src0 indices
                        const int i03 = i13;
                        const int i02 = i12;
                        const int i00 = ic;

                        assert(sizeof(float)*(wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + ne01) <= params->wsize);

                        void * src0_col =   (void *) ((char *) src0->data + (i00*nb00 + i02*nb02 + i03*nb03));
                        float  src1_val = *(float *) ((char *) src1->data + (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13));

                        dragon_vec_mad_q4_0(ne01, dst_row, src0_col, src1_val);
                    }
                }
            }
        }
    }

    //int64_t t1 = dragon_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}

static void dragon_compute_forward_mul_mat_q4_1_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];
    const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    DRAGON_ASSERT(ne02 == ne12);
    DRAGON_ASSERT(ne03 == ne13);
    DRAGON_ASSERT(ne2  == ne12);
    DRAGON_ASSERT(ne3  == ne13);

    // TODO: we don't support permuted src0
    DRAGON_ASSERT(nb00 == (int) DRAGON_TYPE_SIZE[DATA_TYPE_Q4_1] || nb01 == (int) DRAGON_TYPE_SIZE[DATA_TYPE_Q4_1]);

    // dst cannot be transposed or permuted
    DRAGON_ASSERT(nb0 == sizeof(float));
    DRAGON_ASSERT(nb0 <= nb1);
    DRAGON_ASSERT(nb1 <= nb2);
    DRAGON_ASSERT(nb2 <= nb3);

    DRAGON_ASSERT(ne0 == ne01);
    DRAGON_ASSERT(ne1 == ne11);
    DRAGON_ASSERT(ne2 == ne02);
    DRAGON_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
    //
    // nb00 <  nb01 - src0 is transposed
    //   compute by src0 columns

#if defined(DRAGON_USE_ACCELERATE) || defined(DRAGON_USE_OPENBLAS)
    if (dragon_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        DRAGON_ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == DRAGON_TASK_INIT) {
            return;
        }

        if (params->type == DRAGON_TASK_FINALIZE) {
            return;
        }

        float * const wdata = params->wdata;

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                {
                    int id = 0;
                    for (int i01 = 0; i01 < ne01; ++i01) {
                        //for (int i00 = 0; i00 < ne00; ++i00) {
                        //    wdata[id++] = DRAGON_FP16_TO_FP32(*(dragon_fp16_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00));
                        //}
                        dequantize_row_q4_1((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01, wdata + id, ne00);
                        id += ne00;
                    }
                }

                const float * x = wdata;
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                //      float * z =                          wdata + ne00*ne01;

                // z = x * yT
                //{
                //    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                //            ne01, ne11, ne00,
                //            1.0f, x, ne00,
                //                  y, ne00,
                //            0.0f, z, ne11);
                //}

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // transpose z
                //for (int j = 0; j < ne11; ++j) {
                //    for (int i = 0; i < ne01; ++i) {
                //        d[j*ne01 + i] = z[i*ne11 + j];
                //    }
                //}

                {
#if 1
                    // zT = y * xT
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ne11, ne01, ne10,
                            1.0f,    y, ne00,
                                     x, ne00,
                            0.0f,    d, ne01);
#else
                    // zT = (xT * y)T
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            ne01, ne11, ne10,
                            1.0f,    x, ne00,
                                     y, ne00,
                            0.0f,    d, ne01);
#endif
                }
            }
        }

        //printf("CBLAS = %f ms, %d x %d x %d x %d\n", (dragon_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

        return;
    }
#endif

    if (params->type == DRAGON_TASK_INIT) {
        //printf("HHHHHHHHH ith = %d, nth = %d\n", ith, nth);
        if (nb01 >= nb00) {
            char * wdata = params->wdata;

            for (int i13 = 0; i13 < ne13; ++i13) {
                for (int i12 = 0; i12 < ne12; ++i12) {
                    for (int i11 = 0; i11 < ne11; ++i11) {
                        //for (int i10 = 0; i10 < ne10; ++i10) {
                        //    wdata[id++] = DRAGON_FP32_TO_FP16(*(float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10));
                        //}
                        quantize_row_q4_1((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), (void *) wdata, ne10);
                        wdata += (ne10*DRAGON_TYPE_SIZE[DATA_TYPE_Q4_1])/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_1];
                    }
                }
            }

            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);
        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        if (nb01 >= nb00) {
            return;
        }

        float * const wdata = params->wdata;

        // cols per thread
        const int dc = (ne + nth - 1)/nth;

        // col range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, ne);

        dragon_vec_cpy_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + ic0);

        for (int k = 1; k < nth; k++) {
            dragon_vec_acc_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + (ne + CACHE_LINE_SIZE_F32)*k + ic0);
        }

        return;
    }

    if (nb01 >= nb00) {
        // TODO: do not support transposed src1

        // parallelize by src0 rows using dragon_vec_dot_q4_1

        // total rows in src0
        const int nr = ne01*ne02*ne03;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        void * wdata = params->wdata;

        for (int ir = ir0; ir < ir1; ++ir) {
            // src0 indices
            const int i03 = ir/(ne02*ne01);
            const int i02 = (ir - i03*ne02*ne01)/ne01;
            const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int i13 = i03;
            const int i12 = i02;

            const int i0 = i01;
            const int i2 = i02;
            const int i3 = i03;

            void * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            char * src1_col =          ((char *)      wdata + (      (0 + i12*ne11 + i13*ne12*ne11)*ne00*DRAGON_TYPE_SIZE[DATA_TYPE_Q4_1])/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_1]);

            float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

            assert(ne00 % 32 == 0);

            for (int ic = 0; ic < ne11; ++ic) {
                dragon_vec_dot_q4_1(ne00, &dst_col[ic*ne0], src0_row, ((void *) (src1_col + (ic*ne00*DRAGON_TYPE_SIZE[DATA_TYPE_Q4_1])/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_1])));
            }
        }
    } else {
        //printf("AAAAA ith = %d, nth = %d\n", ith, nth);
        // parallelize by src1 columns using dragon_vec_mad_q4_1
        // each thread has its own work data
        // during FINALIZE we accumulate all work data into dst

        // total columns in src1
        const int nc = ne10;

        // columns per thread
        const int dc = (nc + nth - 1)/nth;

        // column range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, nc);

        // work data for thread
        const int wo = (ne + CACHE_LINE_SIZE_F32)*ith;
        float * const wdata = params->wdata;

        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                for (int i11 = 0; i11 < ne11; ++i11) {
                    // dst indices
                    const int i1 = i11;
                    const int i2 = i12;
                    const int i3 = i13;

                    float * dst_row = wdata + wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0;

                    for (int ic = ic0; ic < ic1; ++ic) {
                        // src1 indices
                        const int i10 = ic;

                        // src0 indices
                        const int i03 = i13;
                        const int i02 = i12;
                        const int i00 = ic;

                        assert(sizeof(float)*(wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + ne01) <= params->wsize);

                        void * src0_col =   (void *) ((char *) src0->data + (i00*nb00 + i02*nb02 + i03*nb03));
                        float  src1_val = *(float *) ((char *) src1->data + (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13));

                        dragon_vec_mad_q4_1(ne01, dst_row, src0_col, src1_val);
                    }
                }
            }
        }
    }

    //int64_t t1 = dragon_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}

static void dragon_compute_forward_mul_mat(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_Q4_0:
            {
                dragon_compute_forward_mul_mat_q4_0_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_1:
            {
                dragon_compute_forward_mul_mat_q4_1_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_F16:
            {
                dragon_compute_forward_mul_mat_f16_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_mul_mat_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }

#if 0
    if (src0->type == DRAGON_TYPE_F16 || src0->type == DRAGON_TYPE_Q4_1) {
        static int first = 8;
        printf("src0: ne0 = %5d, ne1 = %5d, ne2 = %5d\n", src0->ne[0], src0->ne[1], src0->ne[2]);
        printf("src1: ne0 = %5d, ne1 = %5d, ne2 = %5d\n", src1->ne[0], src1->ne[1], src1->ne[2]);
        printf("dst:  ne0 = %5d, ne1 = %5d, ne2 = %5d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
        if (first) {
            --first;
        } else {
            for (int k = 0; k < dst->ne[1]; ++k) {
                for (int j = 0; j < dst->ne[0]/16; ++j) {
                    for (int i = 0; i < 16; ++i) {
                        printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
            exit(0);
        }
    } else {
        printf("aaaa src0: ne0 = %5d, ne1 = %5d, ne2 = %5d\n", src0->ne[0], src0->ne[1], src0->ne[2]);
        printf("aaaa src1: ne0 = %5d, ne1 = %5d, ne2 = %5d\n", src1->ne[0], src1->ne[1], src1->ne[2]);
        printf("aaaa dst:  ne0 = %5d, ne1 = %5d, ne2 = %5d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    }
#endif
}

// dragon_compute_forward_scale

static void dragon_compute_forward_scale_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    DRAGON_ASSERT(dragon_is_contiguous(src0));
    DRAGON_ASSERT(dragon_is_contiguous(dst));
    DRAGON_ASSERT(dragon_are_same_shape(src0, dst));
    DRAGON_ASSERT(dragon_is_scalar(src1));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // scale factor
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = dragon_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        dragon_vec_scale_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), v);
    }
}

static void dragon_compute_forward_scale(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_scale_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_cpy

static void dragon_compute_forward_cpy(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    dragon_compute_forward_dup(params, src0, dst);
}

// dragon_compute_forward_reshape

static void dragon_compute_forward_reshape(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
    UNUSED(dst);
}

// dragon_compute_forward_view

static void dragon_compute_forward_view(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
}

// dragon_compute_forward_permute

static void dragon_compute_forward_permute(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
}

// dragon_compute_forward_transpose

static void dragon_compute_forward_transpose(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
}

// dragon_compute_forward_get_rows

static void dragon_compute_forward_get_rows_q4_0(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = dragon_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == DRAGON_TYPE_SIZE[DATA_TYPE_Q4_0]);

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        dequantize_row_q4_0(
                (const void *) ((char *) src0->data + r*src0->nb[1]),
                     (float *) ((char *)  dst->data + i*dst->nb[1]), nc);
    }
}

static void dragon_compute_forward_get_rows_q4_1(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = dragon_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == DRAGON_TYPE_SIZE[DATA_TYPE_Q4_1]);

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        dequantize_row_q4_1(
                (const void *) ((char *) src0->data + r*src0->nb[1]),
                     (float *) ((char *)  dst->data + i*dst->nb[1]), nc);
    }
}

static void dragon_compute_forward_get_rows_f16(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = dragon_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == sizeof(dragon_fp16_t));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        for (int j = 0; j < nc; ++j) {
            dragon_fp16_t v = ((dragon_fp16_t *) ((char *) src0->data + r*src0->nb[1]))[j];
            ((float *) ((char *)  dst->data + i*dst->nb[1]))[j] = DRAGON_FP16_TO_FP32(v);
        }
    }
}

static void dragon_compute_forward_get_rows_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = dragon_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        dragon_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i*dst->nb[1]),
                (float *) ((char *) src0->data + r*src0->nb[1]));
    }
}

static void dragon_compute_forward_get_rows(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_Q4_0:
            {
                dragon_compute_forward_get_rows_q4_0(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_1:
            {
                dragon_compute_forward_get_rows_q4_1(params, src0, src1, dst);
            } break;
        case DATA_TYPE_F16:
            {
                dragon_compute_forward_get_rows_f16(params, src0, src1, dst);
            } break;
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_get_rows_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

// dragon_compute_forward_diag_mask_inf

static void dragon_compute_forward_diag_mask_inf_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(src1->type == DATA_TYPE_I32);
    assert(dragon_nelements(src1) == 1);

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n_past = ((int32_t *) src1->data)[0];

    // TODO: handle transposed/permuted matrices

    const int n  = dragon_nrows(src0);
    const int nc = src0->ne[0];
    const int nr = src0->ne[1];
    const int nz = n/nr;

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < nr; j++) {
            for (int i = n_past; i < nc; i++) {
                if (i > n_past + j) {
                    *(float *)((char *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = -INFINITY;
                }
            }
        }
    }
}

static void dragon_compute_forward_diag_mask_inf(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_diag_mask_inf_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_soft_max

static void dragon_compute_forward_soft_max_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    DRAGON_ASSERT(dragon_is_contiguous(src0));
    DRAGON_ASSERT(dragon_is_contiguous(dst));
    DRAGON_ASSERT(dragon_are_same_shape(src0, dst));

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // TODO: handle transposed/permuted matrices

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = dragon_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float *p = (float *)((char *) dst->data + i1*dst->nb[1]);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(p[i]));
        }
#endif

        float max = -INFINITY;
        dragon_vec_max_f32(nc, &max, p);

        dragon_float sum = 0.0;

        uint16_t scvt;
        for (int i = 0; i < nc; i++) {
            if (p[i] == -INFINITY) {
                p[i] = 0.0f;
            } else {
                //const float val = (p[i] == -INFINITY) ? 0.0 : exp(p[i] - max);
                dragon_fp16_t s = DRAGON_FP32_TO_FP16(p[i] - max);
                memcpy(&scvt, &s, sizeof(scvt));
                const float val = DRAGON_FP16_TO_FP32(table_exp_f16[scvt]);
                sum += val;
                p[i] = val;
            }
        }

        assert(sum > 0.0f);

        sum = 1.0/sum;
        dragon_vec_scale_f32(nc, p, sum);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(p[i]));
            assert(!isinf(p[i]));
        }
#endif
    }
}

static void dragon_compute_forward_soft_max(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_soft_max_f32(params, src0, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_F16:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_rope

static void dragon_compute_forward_rope_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(src1->type == DATA_TYPE_I32);
    assert(dragon_nelements(src1) == 3);

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n_past = ((int32_t *) src1->data)[0];
    const int n_dims = ((int32_t *) src1->data)[1];
    const int mode   = ((int32_t *) src1->data)[2];

    //const int ne0 = src0->ne[0];
    const int ne1 = src0->ne[1];
    const int ne2 = src0->ne[2];
    const int ne3 = src0->ne[3];

    const int nb0 = src0->nb[0];
    const int nb1 = src0->nb[1];
    const int nb2 = src0->nb[2];
    const int nb3 = src0->nb[3];

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    assert(nb0 == sizeof(float));

    // TODO: optimize
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = (mode == 0 ? 0 : n_past); i2 < ne2; i2++) {
            const int p = (mode == 0 ? n_past + i2 : i2);
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const double theta = pow(10000.0, ((double)-i0)/n_dims);

                    const double cos_theta = cos(p*theta);
                    const double sin_theta = sin(p*theta);

                    const float * const src = (float *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                          float * dst_data  = (float *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

                    double x0 = src[0];
                    double x1 = src[1];

                    dst_data[0] = x0*cos_theta - x1*sin_theta;
                    dst_data[1] = x0*sin_theta + x1*cos_theta;
                }
            }
        }
    }
}

static void dragon_compute_forward_rope_f16(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    assert(params->ith == 0);
    assert(src1->type == DATA_TYPE_I32);
    assert(dragon_nelements(src1) == 3);

    if (params->type == DRAGON_TASK_INIT || params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    const int n_past = ((int32_t *) src1->data)[0];
    const int n_dims = ((int32_t *) src1->data)[1];
    const int mode   = ((int32_t *) src1->data)[2];

    //const int ne0 = src0->ne[0];
    const int ne1 = src0->ne[1];
    const int ne2 = src0->ne[2];
    const int ne3 = src0->ne[3];

    const int nb0 = src0->nb[0];
    const int nb1 = src0->nb[1];
    const int nb2 = src0->nb[2];
    const int nb3 = src0->nb[3];

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    assert(nb0 == sizeof(dragon_fp16_t));

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = (mode == 0 ? 0 : n_past); i2 < ne2; i2++) {
            const int p = (mode == 0 ? n_past + i2 : i2);
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const double theta = pow(10000.0, ((double)-i0)/n_dims);

                    const double cos_theta = cos(p*theta);
                    const double sin_theta = sin(p*theta);

                    const dragon_fp16_t * const src = (dragon_fp16_t *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                          dragon_fp16_t * dst_data  = (dragon_fp16_t *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

                    double x0 = dragon_fp16_to_fp32(src[0]);
                    double x1 = dragon_fp16_to_fp32(src[1]);

                    dst_data[0] = dragon_fp32_to_fp16(x0*cos_theta - x1*sin_theta);
                    dst_data[1] = dragon_fp32_to_fp16(x0*sin_theta + x1*cos_theta);
                }
            }
        }
    }
}

static void dragon_compute_forward_rope(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F16:
            {
                dragon_compute_forward_rope_f16(params, src0, src1, dst);
            } break;
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_rope_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_conv_1d_1s

static void dragon_compute_forward_conv_1d_1s_f16_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    DRAGON_ASSERT(src0->type == DATA_TYPE_F16);
    DRAGON_ASSERT(src1->type == DATA_TYPE_F32);
    DRAGON_ASSERT( dst->type == DATA_TYPE_F32);

    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    //const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    //const int ne12 = src1->ne[2];
    //const int ne13 = src1->ne[3];

    //const int ne0  = dst->ne[0];
    //const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];
    //const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = dragon_up32(ne01);

    DRAGON_ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    DRAGON_ASSERT(nb00 == sizeof(dragon_fp16_t));
    DRAGON_ASSERT(nb10 == sizeof(float));

    if (params->type == DRAGON_TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            dragon_fp16_t * const wdata = (dragon_fp16_t *) params->wdata + 0;

            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const dragon_fp16_t * const src = (dragon_fp16_t *)((char *) src0->data + i02*nb02 + i01*nb01);
                    dragon_fp16_t * dst_data = wdata + i02*ew0*ne00;
                    for (int i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            dragon_fp16_t * const wdata = (dragon_fp16_t *) params->wdata + ne02*ew0*ne00;

            for (int i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                dragon_fp16_t * dst_data = wdata;
                for (int i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = DRAGON_FP32_TO_FP16(src[i10]);
                }
            }
        }

        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int i0 = 0; i0 < ne10; ++i0) {
            dst_data[i0] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                dragon_vec_dot_f16(ew0, &v,
                        (dragon_fp16_t *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (dragon_fp16_t *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0] += v;
            }
        }
    }
}

static void dragon_compute_forward_conv_1d_1s_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    DRAGON_ASSERT(src0->type == DATA_TYPE_F32);
    DRAGON_ASSERT(src1->type == DATA_TYPE_F32);
    DRAGON_ASSERT( dst->type == DATA_TYPE_F32);

    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    //const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    //const int ne12 = src1->ne[2];
    //const int ne13 = src1->ne[3];

    //const int ne0  = dst->ne[0];
    //const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];
    //const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = dragon_up32(ne01);

    DRAGON_ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    DRAGON_ASSERT(nb00 == sizeof(float));
    DRAGON_ASSERT(nb10 == sizeof(float));

    if (params->type == DRAGON_TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            float * const wdata = (float *) params->wdata + 0;

            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const float * const src = (float *)((char *) src0->data + i02*nb02 + i01*nb01);
                    float * dst_data = wdata + i02*ew0*ne00;
                    for (int i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            float * const wdata = (float *) params->wdata + ne02*ew0*ne00;

            for (int i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                float * dst_data = wdata;
                for (int i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = src[i10];
                }
            }
        }

        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int i0 = 0; i0 < ne10; ++i0) {
            dst_data[i0] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                dragon_vec_dot_f32(ew0, &v,
                        (float *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (float *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0] += v;
            }
        }
    }
}

static void dragon_compute_forward_conv_1d_1s(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F16:
            {
                dragon_compute_forward_conv_1d_1s_f16_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_conv_1d_1s_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_conv_1d_2s

static void dragon_compute_forward_conv_1d_2s_f16_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    DRAGON_ASSERT(src0->type == DATA_TYPE_F16);
    DRAGON_ASSERT(src1->type == DATA_TYPE_F32);
    DRAGON_ASSERT( dst->type == DATA_TYPE_F32);

    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    //const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    //const int ne12 = src1->ne[2];
    //const int ne13 = src1->ne[3];

    //const int ne0  = dst->ne[0];
    //const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];
    //const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = dragon_up32(ne01);

    DRAGON_ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    DRAGON_ASSERT(nb00 == sizeof(dragon_fp16_t));
    DRAGON_ASSERT(nb10 == sizeof(float));

    if (params->type == DRAGON_TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            dragon_fp16_t * const wdata = (dragon_fp16_t *) params->wdata + 0;

            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const dragon_fp16_t * const src = (dragon_fp16_t *)((char *) src0->data + i02*nb02 + i01*nb01);
                    dragon_fp16_t * dst_data = wdata + i02*ew0*ne00;
                    for (int i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            dragon_fp16_t * const wdata = (dragon_fp16_t *) params->wdata + ne02*ew0*ne00;

            for (int i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                dragon_fp16_t * dst_data = wdata;
                for (int i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = DRAGON_FP32_TO_FP16(src[i10]);
                }
            }
        }

        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int i0 = 0; i0 < ne10; i0 += 2) {
            dst_data[i0/2] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                dragon_vec_dot_f16(ew0, &v,
                        (dragon_fp16_t *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (dragon_fp16_t *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0/2] += v;
            }
        }
    }
}

static void dragon_compute_forward_conv_1d_2s_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
              struct dragon_tensor * dst) {
    DRAGON_ASSERT(src0->type == DATA_TYPE_F32);
    DRAGON_ASSERT(src1->type == DATA_TYPE_F32);
    DRAGON_ASSERT( dst->type == DATA_TYPE_F32);

    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    //const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    //const int ne12 = src1->ne[2];
    //const int ne13 = src1->ne[3];

    //const int ne0  = dst->ne[0];
    //const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];
    //const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = dragon_up32(ne01);

    DRAGON_ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    DRAGON_ASSERT(nb00 == sizeof(float));
    DRAGON_ASSERT(nb10 == sizeof(float));

    if (params->type == DRAGON_TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            float * const wdata = (float *) params->wdata + 0;

            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const float * const src = (float *)((char *) src0->data + i02*nb02 + i01*nb01);
                    float * dst_data = wdata + i02*ew0*ne00;
                    for (int i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            float * const wdata = (float *) params->wdata + ne02*ew0*ne00;

            for (int i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                float * dst_data = wdata;
                for (int i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = src[i10];
                }
            }
        }

        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int i0 = 0; i0 < ne10; i0 += 2) {
            dst_data[i0/2] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                dragon_vec_dot_f32(ew0, &v,
                        (float *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (float *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0/2] += v;
            }
        }
    }
}

static void dragon_compute_forward_conv_1d_2s(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * src0,
        const struct dragon_tensor * src1,
        struct dragon_tensor * dst) {
    switch (src0->type) {
        case DATA_TYPE_F16:
            {
                dragon_compute_forward_conv_1d_2s_f16_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_conv_1d_2s_f32(params, src0, src1, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_flash_attn

static void dragon_compute_forward_flash_attn_f32(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * q,
        const struct dragon_tensor * k,
        const struct dragon_tensor * v,
        const bool masked,
             struct dragon_tensor * dst) {
    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int neq0 = q->ne[0];
    const int neq1 = q->ne[1];
    const int neq2 = q->ne[2];
    const int neq3 = q->ne[3];

    const int nek0 = k->ne[0];
    const int nek1 = k->ne[1];
    //const int nek2 = k->ne[2];
    //const int nek3 = k->ne[3];

    //const int nev0 = v->ne[0];
    const int nev1 = v->ne[1];
    //const int nev2 = v->ne[2];
    //const int nev3 = v->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];

    const int nbk0 = k->nb[0];
    const int nbk1 = k->nb[1];
    const int nbk2 = k->nb[2];
    const int nbk3 = k->nb[3];

    const int nbq0 = q->nb[0];
    const int nbq1 = q->nb[1];
    const int nbq2 = q->nb[2];
    const int nbq3 = q->nb[3];

    const int nbv0 = v->nb[0];
    const int nbv1 = v->nb[1];
    const int nbv2 = v->nb[2];
    const int nbv3 = v->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int D = neq0;
    const int N = neq1;
    const int P = nek1 - N;
    const int M = P + N;

    const int Mup = dragon_up(M, DRAGON_SOFT_MAX_UNROLL);

    DRAGON_ASSERT(ne0 == D);
    DRAGON_ASSERT(ne1 == N);
    DRAGON_ASSERT(P >= 0);

    DRAGON_ASSERT(nbq0 == sizeof(float));
    DRAGON_ASSERT(nbk0 == sizeof(float));
    DRAGON_ASSERT(nbv0 == sizeof(float));

    DRAGON_ASSERT(neq0 == D);
    DRAGON_ASSERT(nek0 == D);
    DRAGON_ASSERT(nev1 == D);

    DRAGON_ASSERT(neq1 == N);
    DRAGON_ASSERT(nek1 == N + P);
    DRAGON_ASSERT(nev1 == D);

    // dst cannot be transposed or permuted
    DRAGON_ASSERT(nb0 == sizeof(float));
    DRAGON_ASSERT(nb0 <= nb1);
    DRAGON_ASSERT(nb1 <= nb2);
    DRAGON_ASSERT(nb2 <= nb3);

    if (params->type == DRAGON_TASK_INIT) {
        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // parallelize by q rows using dragon_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float scale = 1.0/sqrt((double) D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        float * S = (float *) params->wdata + ith*(Mup + CACHE_LINE_SIZE_F32);

        for (int i = M; i < Mup; ++i) {
            S[i] = -INFINITY;
        }

        for (int ic = 0; ic < nek1; ++ic) {
            // k indices
            const int ik3 = iq3;
            const int ik2 = iq2;
            const int ik1 = ic;

            // S indices
            const int i1 = ik1;

            dragon_vec_dot_f32(neq0,
                    S + i1,
                    (float *) ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)),
                    (float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)));
        }

        // scale
        dragon_vec_scale_f32(nek1, S, scale);

        if (masked) {
            for (int i = P; i < M; i++) {
                if (i > P + iq1) {
                    S[i] = -INFINITY;
                }
            }
        }

        // softmax
        {
            float max = -INFINITY;
            dragon_vec_max_f32(M, &max, S);

            float sum = 0.0f;
            {
#ifdef DRAGON_SOFT_MAX_ACCELERATE
                max = -max;
                vDSP_vsadd(S, 1, &max, S, 1, Mup);
                vvexpf(S, S, &Mup);
                dragon_vec_sum_f32(Mup, &sum, S);
#else
                uint16_t   scvt[DRAGON_SOFT_MAX_UNROLL];
                dragon_float sump[DRAGON_SOFT_MAX_UNROLL] = { 0.0 };

                for (int i = 0; i < Mup; i += DRAGON_SOFT_MAX_UNROLL) {
                    float * SS = S + i;

                    for (int j = 0; j < DRAGON_SOFT_MAX_UNROLL; ++j) {
                        if (SS[j] == -INFINITY) {
                            SS[j] = 0.0f;
                        } else {
                            dragon_fp16_t s = DRAGON_FP32_TO_FP16(SS[j] - max);
                            memcpy(&scvt[j], &s, sizeof(uint16_t));
                            const float val = DRAGON_FP16_TO_FP32(table_exp_f16[scvt[j]]);
                            sump[j] += val;
                            SS[j] = val;
                        }
                    }
                }

                for (int i = 0; i < DRAGON_SOFT_MAX_UNROLL; i++) {
                    sum += sump[i];
                }
#endif
            }

            assert(sum > 0.0f);

            sum = 1.0/sum;
            dragon_vec_scale_f32(M, S, sum);

#ifndef NDEBUG
            for (int i = 0; i < M; ++i) {
                assert(!isnan(S[i]));
                assert(!isinf(S[i]));
            }
#endif
        }

        for (int ic = 0; ic < nev1; ++ic) {
            // dst indices
            const int i1 = iq1;
            const int i2 = iq2;
            const int i3 = iq3;

            dragon_vec_dot_f32(nek1,
                    (float *) ((char *) dst->data + (ic*nb0 + i1*nb1  + i2*nb2  + i3*nb3)),
                    (float *) ((char *) v->data   + (         ic*nbv1 + i2*nbv2 + i3*nbv3)),
                    S);
        }
    }
}

static void dragon_compute_forward_flash_attn_f16(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * q,
        const struct dragon_tensor * k,
        const struct dragon_tensor * v,
        const bool masked,
             struct dragon_tensor * dst) {
    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int neq0 = q->ne[0];
    const int neq1 = q->ne[1];
    const int neq2 = q->ne[2];
    const int neq3 = q->ne[3];

    const int nek0 = k->ne[0];
    const int nek1 = k->ne[1];
    //const int nek2 = k->ne[2];
    //const int nek3 = k->ne[3];

    //const int nev0 = v->ne[0];
    const int nev1 = v->ne[1];
    //const int nev2 = v->ne[2];
    //const int nev3 = v->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];

    const int nbk0 = k->nb[0];
    const int nbk1 = k->nb[1];
    const int nbk2 = k->nb[2];
    const int nbk3 = k->nb[3];

    const int nbq0 = q->nb[0];
    const int nbq1 = q->nb[1];
    const int nbq2 = q->nb[2];
    const int nbq3 = q->nb[3];

    const int nbv0 = v->nb[0];
    const int nbv1 = v->nb[1];
    const int nbv2 = v->nb[2];
    const int nbv3 = v->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int D = neq0;
    const int N = neq1;
    const int P = nek1 - N;
    const int M = P + N;

    const int Mup = dragon_up(M, DRAGON_SOFT_MAX_UNROLL);

    DRAGON_ASSERT(ne0 == D);
    DRAGON_ASSERT(ne1 == N);
    DRAGON_ASSERT(P >= 0);

    DRAGON_ASSERT(nbq0 == sizeof(dragon_fp16_t));
    DRAGON_ASSERT(nbk0 == sizeof(dragon_fp16_t));
    DRAGON_ASSERT(nbv0 == sizeof(dragon_fp16_t));

    DRAGON_ASSERT(neq0 == D);
    DRAGON_ASSERT(nek0 == D);
    DRAGON_ASSERT(nev1 == D);

    DRAGON_ASSERT(neq1 == N);
    DRAGON_ASSERT(nek1 == N + P);
    DRAGON_ASSERT(nev1 == D);

    // dst cannot be transposed or permuted
    DRAGON_ASSERT(nb0 == sizeof(float));
    DRAGON_ASSERT(nb0 <= nb1);
    DRAGON_ASSERT(nb1 <= nb2);
    DRAGON_ASSERT(nb2 <= nb3);

    if (params->type == DRAGON_TASK_INIT) {
        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // parallelize by q rows using dragon_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float scale = 1.0/sqrt((double) D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        float * S = (float *) params->wdata + ith*(2*Mup + CACHE_LINE_SIZE_F32);

        for (int i = M; i < Mup; ++i) {
            S[i] = -INFINITY;
        }

        if (DRAGON_VEC_DOT_UNROLL > 2 || nek1 % DRAGON_VEC_DOT_UNROLL != 0) {
            for (int ic = 0; ic < nek1; ++ic) {
                // k indices
                const int ik3 = iq3;
                const int ik2 = iq2;
                const int ik1 = ic;

                // S indices
                const int i1 = ik1;

                dragon_vec_dot_f16(neq0,
                        S + i1,
                        (dragon_fp16_t *) ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)),
                        (dragon_fp16_t *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)));
            }
        } else {
            for (int ic = 0; ic < nek1; ic += DRAGON_VEC_DOT_UNROLL) {
                // k indices
                const int ik3 = iq3;
                const int ik2 = iq2;
                const int ik1 = ic;

                // S indices
                const int i1 = ik1;

                dragon_vec_dot_f16_unroll(neq0, nbk1,
                        S + i1,
                        ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)),
                        (dragon_fp16_t *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)));
            }
        }

        // scale
        dragon_vec_scale_f32(nek1, S, scale);

        if (masked) {
            for (int i = P; i < M; i++) {
                if (i > P + iq1) {
                    S[i] = -INFINITY;
                }
            }
        }

        // softmax
        {
            float max = -INFINITY;
            dragon_vec_max_f32(M, &max, S);

            float sum = 0.0f;
            {
#ifdef DRAGON_SOFT_MAX_ACCELERATE
                max = -max;
                vDSP_vsadd(S, 1, &max, S, 1, Mup);
                vvexpf(S, S, &Mup);
                dragon_vec_sum_f32(Mup, &sum, S);
#else
                uint16_t   scvt[DRAGON_SOFT_MAX_UNROLL];
                dragon_float sump[DRAGON_SOFT_MAX_UNROLL] = { 0.0 };

                for (int i = 0; i < Mup; i += DRAGON_SOFT_MAX_UNROLL) {
                    float * SS = S + i;

                    for (int j = 0; j < DRAGON_SOFT_MAX_UNROLL; ++j) {
                        if (SS[j] == -INFINITY) {
                            SS[j] = 0.0f;
                        } else {
                            dragon_fp16_t s = DRAGON_FP32_TO_FP16(SS[j] - max);
                            memcpy(&scvt[j], &s, sizeof(uint16_t));
                            const float val = DRAGON_FP16_TO_FP32(table_exp_f16[scvt[j]]);
                            sump[j] += val;
                            SS[j] = val;
                        }
                    }
                }

                for (int i = 0; i < DRAGON_SOFT_MAX_UNROLL; i++) {
                    sum += sump[i];
                }
#endif
            }

            assert(sum > 0.0f);

            sum = 1.0/sum;
            dragon_vec_scale_f32(M, S, sum);

#ifndef NDEBUG
            for (int i = 0; i < M; ++i) {
                assert(!isnan(S[i]));
                assert(!isinf(S[i]));
            }
#endif
        }

        dragon_fp16_t * S16 = (dragon_fp16_t *) ((float *) params->wdata + ith*(2*Mup + CACHE_LINE_SIZE_F32) + Mup);

        for (int i = 0; i < M; i++) {
            S16[i] = DRAGON_FP32_TO_FP16(S[i]);
        }

        if (DRAGON_VEC_DOT_UNROLL == 1 || (nev1 % DRAGON_VEC_DOT_UNROLL != 0)) {
            for (int ic = 0; ic < nev1; ++ic) {
                // dst indices
                const int i1 = iq1;
                const int i2 = iq2;
                const int i3 = iq3;

                dragon_vec_dot_f16(nek1,
                        (float *)       ((char *) dst->data + (ic*nb0 + i1*nb1  + i2*nb2  + i3*nb3)),
                        (dragon_fp16_t *) ((char *) v->data   + (         ic*nbv1 + i2*nbv2 + i3*nbv3)),
                        S16);
            }
        } else {
            for (int ic = 0; ic < nev1; ic += DRAGON_VEC_DOT_UNROLL) {
                // dst indices
                const int i1 = iq1;
                const int i2 = iq2;
                const int i3 = iq3;

                dragon_vec_dot_f16_unroll(nek1, nbv1,
                        (float *) ((char *) dst->data + (ic*nb0 + i1*nb1  + i2*nb2  + i3*nb3)),
                        ((char *) v->data   + (         ic*nbv1 + i2*nbv2 + i3*nbv3)),
                        S16);
            }
        }
    }
}

static void dragon_compute_forward_flash_attn(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * q,
        const struct dragon_tensor * k,
        const struct dragon_tensor * v,
        const bool masked,
        struct dragon_tensor * dst) {
    switch (q->type) {
        case DATA_TYPE_F16:
            {
                dragon_compute_forward_flash_attn_f16(params, q, k, v, masked, dst);
            } break;
        case DATA_TYPE_F32:
            {
                dragon_compute_forward_flash_attn_f32(params, q, k, v, masked, dst);
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

// dragon_compute_forward_flash_ff

static void dragon_compute_forward_flash_ff_f16(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * a,  // F16
        const struct dragon_tensor * b0, // F16 fc_w
        const struct dragon_tensor * b1, // F32 fc_b
        const struct dragon_tensor * c0, // F16 proj_w
        const struct dragon_tensor * c1, // F32 proj_b
        struct dragon_tensor * dst) {
    int64_t t0 = dragon_perf_time_us();
    UNUSED(t0);

    const int nea0 = a->ne[0];
    const int nea1 = a->ne[1];
    const int nea2 = a->ne[2];
    const int nea3 = a->ne[3];

    const int neb00 = b0->ne[0];
    const int neb01 = b0->ne[1];
    //const int neb02 = b0->ne[2];
    //const int neb03 = b0->ne[3];

    const int neb10 = b1->ne[0];
    const int neb11 = b1->ne[1];
    //const int neb12 = b1->ne[2];
    //const int neb13 = b1->ne[3];

    const int nec00 = c0->ne[0];
    const int nec01 = c0->ne[1];
    //const int nec02 = c0->ne[2];
    //const int nec03 = c0->ne[3];

    const int nec10 = c1->ne[0];
    const int nec11 = c1->ne[1];
    //const int nec12 = c1->ne[2];
    //const int nec13 = c1->ne[3];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    //const int ne3 = dst->ne[3];

    const int nba0 = a->nb[0];
    const int nba1 = a->nb[1];
    const int nba2 = a->nb[2];
    const int nba3 = a->nb[3];

    const int nbb00 = b0->nb[0];
    const int nbb01 = b0->nb[1];
    const int nbb02 = b0->nb[2];
    const int nbb03 = b0->nb[3];

    const int nbb10 = b1->nb[0];
    //const int nbb11 = b1->nb[1];
    //const int nbb12 = b1->nb[2];
    //const int nbb13 = b1->nb[3];

    const int nbc00 = c0->nb[0];
    const int nbc01 = c0->nb[1];
    const int nbc02 = c0->nb[2];
    const int nbc03 = c0->nb[3];

    const int nbc10 = c1->nb[0];
    //const int nbc11 = c1->nb[1];
    //const int nbc12 = c1->nb[2];
    //const int nbc13 = c1->nb[3];

    const int nb0 = dst->nb[0];
    const int nb1 = dst->nb[1];
    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int D = nea0;
    //const int N = nea1;
    const int M = neb01;

    DRAGON_ASSERT(ne0 == nea0);
    DRAGON_ASSERT(ne1 == nea1);
    DRAGON_ASSERT(ne2 == nea2);

    DRAGON_ASSERT(nba0  == sizeof(dragon_fp16_t));
    DRAGON_ASSERT(nbb00 == sizeof(dragon_fp16_t));
    DRAGON_ASSERT(nbb10 == sizeof(float));
    DRAGON_ASSERT(nbc00 == sizeof(dragon_fp16_t));
    DRAGON_ASSERT(nbc10 == sizeof(float));

    DRAGON_ASSERT(neb00 == D);
    DRAGON_ASSERT(neb01 == M);
    DRAGON_ASSERT(neb10 == M);
    DRAGON_ASSERT(neb11 == 1);

    DRAGON_ASSERT(nec00 == M);
    DRAGON_ASSERT(nec01 == D);
    DRAGON_ASSERT(nec10 == D);
    DRAGON_ASSERT(nec11 == 1);

    // dst cannot be transposed or permuted
    DRAGON_ASSERT(nb0 == sizeof(float));
    DRAGON_ASSERT(nb0 <= nb1);
    DRAGON_ASSERT(nb1 <= nb2);
    DRAGON_ASSERT(nb2 <= nb3);

    if (params->type == DRAGON_TASK_INIT) {
        return;
    }

    if (params->type == DRAGON_TASK_FINALIZE) {
        return;
    }

    // parallelize by a rows using dragon_vec_dot_f32

    // total rows in a
    const int nr = nea1*nea2*nea3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // a indices
        const int ia3 = ir/(nea2*nea1);
        const int ia2 = (ir - ia3*nea2*nea1)/nea1;
        const int ia1 = (ir - ia3*nea2*nea1 - ia2*nea1);

        float * S = (float *) params->wdata + ith*(2*M + CACHE_LINE_SIZE_F32);

        for (int ic = 0; ic < neb01; ++ic) {
            // b0 indices
            const int ib03 = ia3;
            const int ib02 = ia2;
            const int ib01 = ic;

            // S indices
            const int i1 = ib01;

            dragon_vec_dot_f16(nea0,
                    S + i1,
                    (dragon_fp16_t *) ((char *) b0->data + (ib01*nbb01 + ib02*nbb02 + ib03*nbb03)),
                    (dragon_fp16_t *) ((char *)  a->data + ( ia1*nba1  +  ia2*nba2  +  ia3*nba3)));
        }

        dragon_vec_add_f32(neb01, S, S, (float *) b1->data);
        //dragon_vec_gelu_f32(neb01, S, S);

        dragon_fp16_t * S16 = (dragon_fp16_t *) ((float *) params->wdata + ith*(2*M + CACHE_LINE_SIZE_F32) + M);

        for (int i = 0; i < M; i++) {
            S16[i] = DRAGON_FP32_TO_FP16(S[i]);
        }

        dragon_vec_gelu_f16(neb01, S16, S16);

        {
            // dst indices
            const int i1 = ia1;
            const int i2 = ia2;
            const int i3 = ia3;

            for (int ic = 0; ic < nec01; ++ic) {

                dragon_vec_dot_f16(neb01,
                        (float *)       ((char *) dst->data + (ic*nb0 + i1*nb1   + i2*nb2   + i3*nb3)),
                        (dragon_fp16_t *) ((char *) c0->data  + (         ic*nbc01 + i2*nbc02 + i3*nbc03)),
                        S16);
            }

            dragon_vec_add_f32(nec01,
                    (float *) ((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3)),
                    (float *) ((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3)),
                    (float *) c1->data);
        }
    }
}

static void dragon_compute_forward_flash_ff(
        const struct dragon_compute_params * params,
        const struct dragon_tensor * a,
        const struct dragon_tensor * b0,
        const struct dragon_tensor * b1,
        const struct dragon_tensor * c0,
        const struct dragon_tensor * c1,
        struct dragon_tensor * dst) {
    switch (b0->type) {
        case DATA_TYPE_F16:
            {
                dragon_compute_forward_flash_ff_f16(params, a, b0, b1, c0, c1, dst);
            } break;
        case DATA_TYPE_F32:
            {
                DRAGON_ASSERT(false); // TODO
            } break;
        case DATA_TYPE_Q4_0:
        case DATA_TYPE_Q4_1:
        case DATA_TYPE_I8:
        case DATA_TYPE_I16:
        case DATA_TYPE_I32:
        case DATA_TYPE_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

/////////////////////////////////

static void dragon_compute_forward(struct dragon_compute_params * params, struct dragon_tensor * tensor) {
    DRAGON_ASSERT(params);

    switch (tensor->op) {
        case DRAGON_OP_DUP:
            {
                dragon_compute_forward_dup(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_ADD:
            {
                dragon_compute_forward_add(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_SUB:
            {
                dragon_compute_forward_sub(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_MUL:
            {
                dragon_compute_forward_mul(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_DIV:
            {
                dragon_compute_forward_div(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_SQR:
            {
                dragon_compute_forward_sqr(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_SQRT:
            {
                dragon_compute_forward_sqrt(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_SUM:
            {
                dragon_compute_forward_sum(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_MEAN:
            {
                dragon_compute_forward_mean(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_REPEAT:
            {
                dragon_compute_forward_repeat(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_ABS:
            {
                dragon_compute_forward_abs(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_SGN:
            {
                dragon_compute_forward_sgn(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_NEG:
            {
                dragon_compute_forward_neg(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_STEP:
            {
                dragon_compute_forward_step(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_RELU:
            {
                dragon_compute_forward_relu(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_GELU:
            {
                dragon_compute_forward_gelu(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_SILU:
            {
                dragon_compute_forward_silu(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_NORM:
            {
                dragon_compute_forward_norm(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_RMS_NORM:
            {
                dragon_compute_forward_rms_norm(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_MUL_MAT:
            {
                dragon_compute_forward_mul_mat(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_SCALE:
            {
                dragon_compute_forward_scale(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_CPY:
            {
                dragon_compute_forward_cpy(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_RESHAPE:
            {
                dragon_compute_forward_reshape(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_VIEW:
            {
                dragon_compute_forward_view(params, tensor->src0);
            } break;
        case DRAGON_OP_PERMUTE:
            {
                dragon_compute_forward_permute(params, tensor->src0);
            } break;
        case DRAGON_OP_TRANSPOSE:
            {
                dragon_compute_forward_transpose(params, tensor->src0);
            } break;
        case DRAGON_OP_GET_ROWS:
            {
                dragon_compute_forward_get_rows(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_DIAG_MASK_INF:
            {
                dragon_compute_forward_diag_mask_inf(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_SOFT_MAX:
            {
                dragon_compute_forward_soft_max(params, tensor->src0, tensor);
            } break;
        case DRAGON_OP_ROPE:
            {
                dragon_compute_forward_rope(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_CONV_1D_1S:
            {
                dragon_compute_forward_conv_1d_1s(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_CONV_1D_2S:
            {
                dragon_compute_forward_conv_1d_2s(params, tensor->src0, tensor->src1, tensor);
            } break;
        case DRAGON_OP_FLASH_ATTN:
            {
                int32_t t = dragon_get_i32_1d(tensor->opt[1], 0);
                DRAGON_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                dragon_compute_forward_flash_attn(params, tensor->src0, tensor->src1, tensor->opt[0], masked, tensor);
            } break;
        case DRAGON_OP_FLASH_FF:
            {
                dragon_compute_forward_flash_ff(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor->opt[2], tensor);
            } break;
        case DRAGON_OP_NONE:
            {
                // nop
            } break;
        case DRAGON_OP_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

////////////////////////////////////////////////////////////////////////////////

static void dragon_compute_backward(struct dragon_context * ctx, struct dragon_tensor * tensor, bool inplace) {
    struct dragon_tensor * src0 = tensor->src0;
    struct dragon_tensor * src1 = tensor->src1;

    switch (tensor->op) {
        case DRAGON_OP_DUP:
            {
                if (src0->grad) {
                    src0->grad = dragon_add_impl(ctx, src0->grad, tensor->grad, inplace);
                }
            } break;
        case DRAGON_OP_ADD:
            {
                if (src0->grad) {
                    src0->grad = dragon_add_impl(ctx, src0->grad, tensor->grad, inplace);
                }
                if (src1->grad) {
                    src1->grad = dragon_add_impl(ctx, src1->grad, tensor->grad, inplace);
                }
            } break;
        case DRAGON_OP_SUB:
            {
                if (src0->grad) {
                    src0->grad = dragon_add_impl(ctx, src0->grad, tensor->grad, inplace);
                }
                if (src1->grad) {
                    src1->grad = dragon_sub_impl(ctx, src1->grad, tensor->grad, inplace);
                }
            } break;
        case DRAGON_OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        dragon_add_impl(ctx,
                                src0->grad,
                                dragon_mul(ctx, src1, tensor->grad),
                                inplace);
                }
                if (src1->grad) {
                    src1->grad =
                        dragon_add_impl(ctx,
                                src1->grad,
                                dragon_mul(ctx, src0, tensor->grad),
                                inplace);
                }
            } break;
        case DRAGON_OP_DIV:
            {
                if (src0->grad) {
                    src0->grad =
                        dragon_add_impl(ctx,
                                src0->grad,
                                dragon_div(ctx, tensor->grad, src1),
                                inplace);
                }
                if (src1->grad) {
                    src1->grad =
                        dragon_sub_impl(ctx,
                                src1->grad,
                                dragon_mul(ctx,
                                    tensor->grad,
                                    dragon_div(ctx, tensor, src1)),
                                inplace);
                }
            } break;
        case DRAGON_OP_SQR:
            {
                if (src0->grad) {
                    src0->grad =
                        dragon_add_impl(ctx,
                                src0->grad,
                                dragon_mul(ctx,
                                    dragon_mul(ctx, src0, tensor->grad),
                                    dragon_repeat(ctx, dragon_new_f32(ctx, 2.0f), src0)),
                                inplace);
                }
            } break;
        case DRAGON_OP_SQRT:
            {
                if (src0->grad) {
                    src0->grad =
                        dragon_add_impl(ctx,
                                src0->grad,
                                dragon_div(ctx,
                                    dragon_repeat(ctx, dragon_new_f32(ctx, 0.5f), tensor),
                                    tensor),
                                inplace);
                }
            } break;
        case DRAGON_OP_SUM:
            {
                if (src0->grad) {
                    src0->grad =
                        dragon_add_impl(ctx,
                                src0->grad,
                                dragon_repeat(ctx, tensor->grad, src0->grad),
                                inplace);
                }
            } break;
        case DRAGON_OP_MEAN:
            {
                DRAGON_ASSERT(false); // TODO: implement
            } break;
        case DRAGON_OP_REPEAT:
            {
                if (src0->grad) {
                    src0->grad =
                        dragon_add_impl(ctx,
                                src0->grad,
                                dragon_sum(ctx, tensor->grad),
                                inplace);
                }
            } break;
        case DRAGON_OP_ABS:
            {
                if (src0->grad) {
                    src0->grad =
                        dragon_add_impl(ctx,
                                src0->grad,
                                dragon_mul(ctx,
                                    dragon_sgn(ctx, src0),
                                    tensor->grad),
                                inplace);
                }
            } break;
        case DRAGON_OP_SGN:
            {
                if (src0->grad) {
                    // noop
                }
            } break;
        case DRAGON_OP_NEG:
            {
                if (src0->grad) {
                    src0->grad = dragon_sub_impl(ctx, src0->grad, tensor->grad, inplace);
                }
            } break;
        case DRAGON_OP_STEP:
            {
                if (src0->grad) {
                    // noop
                }
            } break;
        case DRAGON_OP_RELU:
            {
                if (src0->grad) {
                    src0->grad = dragon_sub_impl(ctx,
                            src0->grad,
                            dragon_mul(ctx,
                                dragon_step(ctx, src0),
                                tensor->grad),
                            inplace);
                }
            } break;
        case DRAGON_OP_GELU:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_SILU:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_NORM:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_RMS_NORM:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_MUL_MAT:
            {
                if (src0->grad) {
                    // TODO: this requires outer product - dragon_out_prod(ctx, src1, tensor->grad);
                    DRAGON_ASSERT(false);
                }
                if (src1->grad) {
                    src1->grad =
                        dragon_add_impl(ctx,
                                src1->grad,
                                // TODO: fix transpose, the node will break the graph connections
                                dragon_mul_mat(ctx, dragon_transpose(ctx, src0), tensor->grad),
                                inplace);
                }
            } break;
        case DRAGON_OP_SCALE:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_CPY:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_RESHAPE:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_VIEW:
            {
                DRAGON_ASSERT(false); // not supported
            } break;
        case DRAGON_OP_PERMUTE:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_TRANSPOSE:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_GET_ROWS:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_DIAG_MASK_INF:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_SOFT_MAX:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_ROPE:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_CONV_1D_1S:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_CONV_1D_2S:
            {
                DRAGON_ASSERT(false); // TODO: not implemented
            } break;
        case DRAGON_OP_FLASH_ATTN:
            {
                DRAGON_ASSERT(false); // not supported
            } break;
        case DRAGON_OP_FLASH_FF:
            {
                DRAGON_ASSERT(false); // not supported
            } break;
        case DRAGON_OP_NONE:
            {
                // nop
            } break;
        case DRAGON_OP_COUNT:
            {
                DRAGON_ASSERT(false);
            } break;
    }
}

static void dragon_visit_parents(struct dragon_cgraph * cgraph, struct dragon_tensor * node) {
    if (node->grad == NULL) {
        // this usually happens when we generate intermediate nodes from constants in the backward pass
        // it can also happen during forward pass, if the user performs computations with constants
        if (node->op != DRAGON_OP_NONE) {
            //DRAGON_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
        }
    }

    // check if already visited
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i] == node) {
            return;
        }
    }

    for (int i = 0; i < cgraph->n_leafs; i++) {
        if (cgraph->leafs[i] == node) {
            return;
        }
    }

    if (node->src0) {
        dragon_visit_parents(cgraph, node->src0);
    }

    if (node->src1) {
        dragon_visit_parents(cgraph, node->src1);
    }

    for (int i = 0; i < DRAGON_MAX_OPT; ++i) {
        if (node->opt[i]) {
            dragon_visit_parents(cgraph, node->opt[i]);
        }
    }

    if (node->op == DRAGON_OP_NONE && node->grad == NULL) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        DRAGON_ASSERT(cgraph->n_leafs < DRAGON_MAX_NODES);

        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {
        DRAGON_ASSERT(cgraph->n_nodes < DRAGON_MAX_NODES);

        cgraph->nodes[cgraph->n_nodes] = node;
        cgraph->grads[cgraph->n_nodes] = node->grad;
        cgraph->n_nodes++;
    }
}

static void dragon_build_forward_impl(struct dragon_cgraph * cgraph, struct dragon_tensor * tensor, bool expand) {
    if (!expand) {
        cgraph->n_nodes = 0;
        cgraph->n_leafs = 0;
    }

    const int n0 = cgraph->n_nodes;
    UNUSED(n0);

    dragon_visit_parents(cgraph, tensor);

    const int n_new = cgraph->n_nodes - n0;
    DRAGON_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
        DRAGON_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

void dragon_build_forward_expand(struct dragon_cgraph * cgraph, struct dragon_tensor * tensor) {
    dragon_build_forward_impl(cgraph, tensor, true);
}

struct dragon_cgraph dragon_build_forward(struct dragon_tensor * tensor) {
    struct dragon_cgraph result = {
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.n_threads    =*/ 0,
        /*.work_size    =*/ 0,
        /*.work         =*/ NULL,
        /*.nodes        =*/ { NULL },
        /*.grads        =*/ { NULL },
        /*.leafs        =*/ { NULL },
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
    };

    dragon_build_forward_impl(&result, tensor, false);

    return result;
}

struct dragon_cgraph dragon_build_backward(struct dragon_context * ctx, struct dragon_cgraph * gf, bool keep) {
    struct dragon_cgraph result = *gf;

    DRAGON_ASSERT(gf->n_nodes > 0);

    // if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
    if (keep) {
        for (int i = 0; i < gf->n_nodes; i++) {
            struct dragon_tensor * node = gf->nodes[i];

            if (node->grad) {
                node->grad = dragon_dup_tensor(ctx, node);
                gf->grads[i] = node->grad;
            }
        }
    }

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct dragon_tensor * node = gf->nodes[i];

        // because we detached the grad nodes from the original graph, we can afford inplace operations
        if (node->grad) {
            dragon_compute_backward(ctx, node, keep);
        }
    }

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct dragon_tensor * node = gf->nodes[i];

        if (node->is_param) {
            DRAGON_PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
            dragon_build_forward_impl(&result, node->grad, true);
        }
    }

    return result;
}

//
// thread data
//
// synchronization is done via busy loops
// I tried using spin locks, but not sure how to use them correctly - the things I tried were slower than busy loops
//

#ifdef __APPLE__

//#include <os/lock.h>
//
//typedef os_unfair_lock dragon_lock_t;
//
//#define dragon_lock_init(x)    UNUSED(x)
//#define dragon_lock_destroy(x) UNUSED(x)
//#define dragon_lock_lock       os_unfair_lock_lock
//#define dragon_lock_unlock     os_unfair_lock_unlock
//
//#define DRAGON_LOCK_INITIALIZER OS_UNFAIR_LOCK_INIT

typedef int dragon_lock_t;

#define dragon_lock_init(x)    UNUSED(x)
#define dragon_lock_destroy(x) UNUSED(x)
#define dragon_lock_lock(x)    UNUSED(x)
#define dragon_lock_unlock(x)  UNUSED(x)

#define DRAGON_LOCK_INITIALIZER 0

typedef pthread_t dragon_thread_t;

#define dragon_thread_create pthread_create
#define dragon_thread_join   pthread_join

#else

//typedef pthread_spinlock_t dragon_lock_t;

//#define dragon_lock_init(x) pthread_spin_init(x, PTHREAD_PROCESS_PRIVATE)
//#define dragon_lock_destroy pthread_spin_destroy
//#define dragon_lock_lock    pthread_spin_lock
//#define dragon_lock_unlock  pthread_spin_unlock

typedef int dragon_lock_t;

#define dragon_lock_init(x)    UNUSED(x)
#define dragon_lock_destroy(x) UNUSED(x)
#define dragon_lock_lock(x)    UNUSED(x)
#define dragon_lock_unlock(x)  UNUSED(x)

#define DRAGON_LOCK_INITIALIZER 0

typedef pthread_t dragon_thread_t;

#define dragon_thread_create pthread_create
#define dragon_thread_join   pthread_join

#endif

struct dragon_compute_state_shared {
    dragon_lock_t spin;

    int n_threads;

    // synchronization primitives
    atomic_int  n_ready;
    atomic_bool has_work;
    atomic_bool stop; // stop all threads
};

struct dragon_compute_state {
    dragon_thread_t thrd;

    struct dragon_compute_params params;
    struct dragon_tensor * node;

    struct dragon_compute_state_shared * shared;
};

static thread_ret_t dragon_graph_compute_thread(void * data) {
    struct dragon_compute_state * state = (struct dragon_compute_state *) data;

    const int n_threads = state->shared->n_threads;

    while (true) {
        // 同步屏障：所有线程在这里等待
        if (atomic_fetch_add(&state->shared->n_ready, 1) == n_threads) {
            // 最后一个线程发出通知：所有线程已经到达，可以执行下一步操作
            atomic_store(&state->shared->has_work, false);
        } else {
            while (!atomic_load(&state->shared->has_work)) {
                if (atomic_load(&state->shared->stop)) {
                    return 0;
                }
            }
        }

        atomic_fetch_sub(&state->shared->n_ready, 1);

        // 等待分配工作
        while (atomic_load(&state->shared->has_work)) {
            if (atomic_load(&state->shared->stop)) {
                return 0;
            }
        }

        // check if we should stop
        if (atomic_load(&state->shared->stop)) {
            break;
        }

        if (state->node) {
            if (state->params.ith < state->params.nth) {
                dragon_compute_forward(&state->params, state->node);
            }

            state->node = NULL;
        } else {
            break;
        }
    }

    return 0;
}

void dragon_graph_compute(struct dragon_context * ctx, struct dragon_cgraph * cgraph) {
    const int n_threads = cgraph->n_threads;

    struct dragon_compute_state_shared state_shared = {
        /*.spin      =*/ DRAGON_LOCK_INITIALIZER,
        /*.n_threads =*/ n_threads,
        /*.n_ready   =*/ 0,
        /*.has_work  =*/ false,
        /*.stop      =*/ false,
    };
    struct dragon_compute_state * workers = n_threads > 1 ? alloca(sizeof(struct dragon_compute_state)*(n_threads - 1)) : NULL;

    // create thread pool
    if (n_threads > 1) {
        dragon_lock_init(&state_shared.spin);

        atomic_store(&state_shared.has_work, true);

        for (int j = 0; j < n_threads - 1; j++) {
            workers[j] = (struct dragon_compute_state) {
                .thrd   = 0,
                .params = {
                    .type  = DRAGON_TASK_COMPUTE,
                    .ith   = j + 1,
                    .nth   = n_threads,
                    .wsize = cgraph->work ? dragon_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                },
                .node   = NULL,
                .shared = &state_shared,
            };

            int rc = dragon_thread_create(&workers[j].thrd, NULL, dragon_graph_compute_thread, &workers[j]);
            DRAGON_ASSERT(rc == 0);
            UNUSED(rc);
        }
    }

    // initialize tasks + work buffer
    {
        size_t work_size = 0;

        // thread scheduling for the different operations
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct dragon_tensor * node = cgraph->nodes[i];

            switch (node->op) {
                case DRAGON_OP_DUP:
                    {
                        node->n_tasks = 1;
                    } break;
                case DRAGON_OP_ADD:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case DRAGON_OP_SUB:
                case DRAGON_OP_MUL:
                case DRAGON_OP_DIV:
                case DRAGON_OP_SQR:
                case DRAGON_OP_SQRT:
                case DRAGON_OP_SUM:
                case DRAGON_OP_MEAN:
                case DRAGON_OP_REPEAT:
                case DRAGON_OP_ABS:
                case DRAGON_OP_SGN:
                case DRAGON_OP_NEG:
                case DRAGON_OP_STEP:
                case DRAGON_OP_RELU:
                    {
                        node->n_tasks = 1;
                    } break;
                case DRAGON_OP_GELU:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case DRAGON_OP_SILU:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case DRAGON_OP_NORM:
                case DRAGON_OP_RMS_NORM:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case DRAGON_OP_MUL_MAT:
                    {
                        node->n_tasks = n_threads;

                        // TODO: use different scheduling for different matrix sizes
                        //const int nr0 = dragon_nrows(node->src0);
                        //const int nr1 = dragon_nrows(node->src1);

                        //node->n_tasks = MIN(n_threads, MAX(1, nr0/128));
                        //printf("nr0 = %8d, nr1 = %8d, nr0*nr1 = %8d, n_tasks = %d\n", nr0, nr1, nr0*nr1, node->n_tasks);

                        size_t cur = 0;

                        // TODO: better way to determine if the matrix is transposed
                        if (node->src0->nb[1] < node->src0->nb[0]) {
                            cur = dragon_nbytes(node)*node->n_tasks; // TODO: this can become (n_tasks-1)
                                                                   // TODO: overestimated by factor of x2 for FP16
                        } else {
                            if (node->src0->type == DATA_TYPE_F16 &&
                                node->src1->type == DATA_TYPE_F32) {
#if defined(DRAGON_USE_ACCELERATE) || defined(DRAGON_USE_OPENBLAS)
                                if (dragon_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                    node->n_tasks = 1; // TODO: this actually is doing nothing
                                                       //       the threads are still spinning
                                    cur = DRAGON_TYPE_SIZE[DATA_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                                    //printf("src0: ne0 = %d, ne1 = %d, ne = %d\n", node->src0->ne[0], node->src0->ne[1], node->src0->ne[0]*node->src0->ne[1]);
                                    //printf("src1: ne0 = %d, ne1 = %d, ne = %d\n", node->src1->ne[0], node->src1->ne[1], node->src1->ne[0]*node->src1->ne[1]);
                                    //printf("cur = %zu\n", cur);
                                } else {
                                    cur = DRAGON_TYPE_SIZE[DATA_TYPE_F16]*dragon_nelements(node->src1);
                                }
#else
                                cur = DRAGON_TYPE_SIZE[DATA_TYPE_F16]*dragon_nelements(node->src1);
#endif
                            } else if (node->src0->type == DATA_TYPE_F32 &&
                                       node->src1->type == DATA_TYPE_F32) {
                                cur = 0;
                            } else if (node->src0->type == DATA_TYPE_Q4_0 &&
                                       node->src1->type == DATA_TYPE_F32) {
#if defined(DRAGON_USE_ACCELERATE) || defined(DRAGON_USE_OPENBLAS)
                                if (dragon_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                    node->n_tasks = 1;
                                    cur = DRAGON_TYPE_SIZE[DATA_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                                } else {
                                    cur = (DRAGON_TYPE_SIZE[DATA_TYPE_Q4_0]*dragon_nelements(node->src1))/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_0];
                                }
#else
                                cur = (DRAGON_TYPE_SIZE[DATA_TYPE_Q4_0]*dragon_nelements(node->src1))/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_0];
#endif
                            } else if (node->src0->type == DATA_TYPE_Q4_1 &&
                                       node->src1->type == DATA_TYPE_F32) {
#if defined(DRAGON_USE_ACCELERATE) || defined(DRAGON_USE_OPENBLAS)
                                if (dragon_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                    node->n_tasks = 1;
                                    cur = DRAGON_TYPE_SIZE[DATA_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                                } else {
                                    cur = (DRAGON_TYPE_SIZE[DATA_TYPE_Q4_1]*dragon_nelements(node->src1))/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_1];
                                }
#else
                                cur = (DRAGON_TYPE_SIZE[DATA_TYPE_Q4_1]*dragon_nelements(node->src1))/DRAGON_BLCK_SIZE[DATA_TYPE_Q4_1];
#endif
                            } else {
                                DRAGON_ASSERT(false);
                            }
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case DRAGON_OP_SCALE:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case DRAGON_OP_CPY:
                case DRAGON_OP_RESHAPE:
                case DRAGON_OP_VIEW:
                case DRAGON_OP_PERMUTE:
                case DRAGON_OP_TRANSPOSE:
                case DRAGON_OP_GET_ROWS:
                case DRAGON_OP_DIAG_MASK_INF:
                    {
                        node->n_tasks = 1;
                    } break;
                case DRAGON_OP_SOFT_MAX:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case DRAGON_OP_ROPE:
                    {
                        node->n_tasks = 1;
                    } break;
                case DRAGON_OP_CONV_1D_1S:
                case DRAGON_OP_CONV_1D_2S:
                    {
                        node->n_tasks = n_threads;

                        DRAGON_ASSERT(node->src0->ne[3] == 1);
                        DRAGON_ASSERT(node->src1->ne[2] == 1);
                        DRAGON_ASSERT(node->src1->ne[3] == 1);

                        size_t cur = 0;
                        const int nk = node->src0->ne[0];

                        if (node->src0->type == DATA_TYPE_F16 &&
                            node->src1->type == DATA_TYPE_F32) {
                            cur = sizeof(dragon_fp16_t)*(
                                    nk*dragon_up32(node->src0->ne[1])*node->src0->ne[2] +
                                    ( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]
                                    );
                        } else if (node->src0->type == DATA_TYPE_F32 &&
                                   node->src1->type == DATA_TYPE_F32) {
                            cur = sizeof(float)*(
                                    nk*dragon_up32(node->src0->ne[1])*node->src0->ne[2] +
                                    ( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]
                                    );
                        } else {
                            DRAGON_ASSERT(false);
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case DRAGON_OP_FLASH_ATTN:
                    {
                        node->n_tasks = n_threads;

                        size_t cur = 0;

                        const int ne11 = dragon_up(node->src1->ne[1], DRAGON_SOFT_MAX_UNROLL);

                        if (node->src1->type == DATA_TYPE_F32) {
                            cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
                        }

                        if (node->src1->type == DATA_TYPE_F16) {
                            cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case DRAGON_OP_FLASH_FF:
                    {
                        node->n_tasks = n_threads;

                        size_t cur = 0;

                        if (node->src1->type == DATA_TYPE_F32) {
                            cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
                        }

                        if (node->src1->type == DATA_TYPE_F16) {
                            cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case DRAGON_OP_NONE:
                    {
                        node->n_tasks = 1;
                    } break;
                case DRAGON_OP_COUNT:
                    {
                        DRAGON_ASSERT(false);
                    } break;
            }
        }

        if (cgraph->work != NULL && work_size > cgraph->work_size) {
            DRAGON_ASSERT(false); // TODO: better handling
        }

        if (work_size > 0 && cgraph->work == NULL) {
            cgraph->work_size = work_size + CACHE_LINE_SIZE*(n_threads - 1);

            DRAGON_PRINT_DEBUG("%s: allocating work buffer for graph (%zu bytes)\n", __func__, cgraph->work_size);
            cgraph->work = dragon_new_tensor_1d(ctx, DATA_TYPE_I8, cgraph->work_size);
        }
    }

    const int64_t perf_start_cycles  = dragon_perf_cycles();
    const int64_t perf_start_time_us = dragon_perf_time_us();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        DRAGON_PRINT_DEBUG_5("%s: %d/%d\n", __func__, i, cgraph->n_nodes);

        struct dragon_tensor * node = cgraph->nodes[i];

        // TODO: this could be used to avoid unnecessary computations, but it needs to be improved
        //if (node->grad == NULL && node->perf_runs > 0) {
        //    continue;
        //}

        const int64_t perf_node_start_cycles  = dragon_perf_cycles();
        const int64_t perf_node_start_time_us = dragon_perf_time_us();

        // 每一个node计算分为三个阶段，分别是：
        // 1. INIT 阶段：负责node的初始化操作，只需要主线程工作
        // 2. COMPUTE 阶段：执行计算，主线程和其它线程一起工作
        // 3. FINALIZE 阶段：整合计算结果，主线程和其它线程一起工作
        // 在 COMPUTE 和 FINALIZE 阶段，主线程通过设置其它线程的 params 来分配任务。
        // 各进程使用 state_shared 交互信息。你需要认真思考清楚 state_shared.n_ready 和 state_shared.has_work的功能。

        // INIT 阶段：只需要主线程工作
        struct dragon_compute_params params = {
            /*.type  =*/ DRAGON_TASK_INIT,
            /*.ith   =*/ 0,
            /*.nth   =*/ node->n_tasks,
            /*.wsize =*/ cgraph->work ? dragon_nbytes(cgraph->work) : 0,
            /*.wdata =*/ cgraph->work ? cgraph->work->data : NULL,
        };

        dragon_compute_forward(&params, node);

        // COMPUTE 阶段：主线程和其它线程一起工作
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads) {
                atomic_store(&state_shared.has_work, false);
            }

            while (!atomic_load(&state_shared.has_work)) {}

            // launch thread pool
            for (int j = 0; j < n_threads - 1; j++) {
                workers[j].params = (struct dragon_compute_params) {
                    .type  = DRAGON_TASK_COMPUTE,
                    .ith   = j + 1,
                    .nth   = node->n_tasks,
                    .wsize = cgraph->work ? dragon_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                };
                workers[j].node = node;
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) > 0) {}

            atomic_store(&state_shared.has_work, true);
        }

        params.type = DRAGON_TASK_COMPUTE;
        dragon_compute_forward(&params, node);

        // wait for thread pool
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {}

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) != 0) {}
        }

        // FINALIZE 阶段：主线程和其它线程一起工作
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {}

            // launch thread pool
            for (int j = 0; j < n_threads - 1; j++) {
                workers[j].params = (struct dragon_compute_params) {
                    .type  = DRAGON_TASK_FINALIZE,
                    .ith   = j + 1,
                    .nth   = node->n_tasks,
                    .wsize = cgraph->work ? dragon_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                };
                workers[j].node = node;
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) > 0) {}

            atomic_store(&state_shared.has_work, true);
        }

        params.type = DRAGON_TASK_FINALIZE;
        dragon_compute_forward(&params, node);

        // wait for thread pool
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {}

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) != 0) {}
        }

        // performance stats (node)
        {
            int64_t perf_cycles_cur  = dragon_perf_cycles()  - perf_node_start_cycles;
            int64_t perf_time_us_cur = dragon_perf_time_us() - perf_node_start_time_us;

            node->perf_runs++;
            node->perf_cycles  += perf_cycles_cur;
            node->perf_time_us += perf_time_us_cur;
        }
    }

    // join thread pool
    if (n_threads > 1) {
        atomic_store(&state_shared.stop, true);
        atomic_store(&state_shared.has_work, true);

        for (int j = 0; j < n_threads - 1; j++) {
            int rc = dragon_thread_join(workers[j].thrd, NULL);
            DRAGON_ASSERT(rc == 0);
            UNUSED(rc);
        }

        dragon_lock_destroy(&state_shared.spin);
    }

    // performance stats (graph)
    {
        int64_t perf_cycles_cur  = dragon_perf_cycles()  - perf_start_cycles;
        int64_t perf_time_us_cur = dragon_perf_time_us() - perf_start_time_us;

        cgraph->perf_runs++;
        cgraph->perf_cycles  += perf_cycles_cur;
        cgraph->perf_time_us += perf_time_us_cur;

        DRAGON_PRINT_DEBUG("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n",
                __func__, cgraph->perf_runs,
                (double) perf_cycles_cur      / (double) dragon_cycles_per_ms(),
                (double) cgraph->perf_cycles  / (double) dragon_cycles_per_ms() / (double) cgraph->perf_runs,
                (double) perf_time_us_cur     / 1000.0,
                (double) cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
    }
}

void dragon_graph_reset(struct dragon_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct dragon_tensor * grad = cgraph->grads[i];

        if (grad) {
            dragon_set_zero(grad);
        }
    }
}

void dragon_graph_print(const struct dragon_cgraph * cgraph) {
    int64_t perf_total_per_op_us[DRAGON_OP_COUNT] = {0};

    DRAGON_PRINT("=== GRAPH ===\n");

    DRAGON_PRINT_DEBUG("n_threads       = %d\n",       cgraph->n_threads);
    DRAGON_PRINT_DEBUG("total work size = %zu bytes\n",cgraph->work_size);

    DRAGON_PRINT("n_nodes = %d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct dragon_tensor * node = cgraph->nodes[i];

        perf_total_per_op_us[node->op] += node->perf_time_us;

        DRAGON_PRINT(" - %3d: [ %6d, %6d, %6d] %16s %s (%3d) cpu = %7.3f / %7.3f ms, wall = %7.3f / %7.3f ms\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                DRAGON_OP_LABEL[node->op], node->is_param ? "x" : node->grad ? "g" : " ", node->perf_runs,
                (double) node->perf_cycles  / (double) dragon_cycles_per_ms(),
                (double) node->perf_cycles  / (double) dragon_cycles_per_ms() / (double) node->perf_runs,
                (double) node->perf_time_us / 1000.0,
                (double) node->perf_time_us / 1000.0 / node->perf_runs);
    }

    DRAGON_PRINT("n_leafs = %d\n", cgraph->n_leafs);
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct dragon_tensor * node = cgraph->leafs[i];

        DRAGON_PRINT(" - %3d: [ %6d, %6d] %8s\n",
                i,
                node->ne[0], node->ne[1],
                DRAGON_OP_LABEL[node->op]);
    }

    for (int i = 0; i < DRAGON_OP_COUNT; i++) {
        DRAGON_PRINT("perf_total_per_op_us[%16s] = %7.3f ms\n", DRAGON_OP_LABEL[i], (double) perf_total_per_op_us[i] / 1000.0);
    }

    DRAGON_PRINT("========================================\n");
}

// check if node is part of the graph
static bool dragon_graph_find(const struct dragon_cgraph * cgraph, const struct dragon_tensor * node) {
    if (cgraph == NULL) {
        return true;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i] == node) {
            return true;
        }
    }

    return false;
}

static struct dragon_tensor * dragon_graph_get_parent(const struct dragon_cgraph * cgraph, const struct dragon_tensor * node) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct dragon_tensor * parent = cgraph->nodes[i];

        if (parent->grad == node) {
            return parent;
        }
    }

    return NULL;
}

void dragon_graph_dump_dot(const struct dragon_cgraph * gb, const struct dragon_cgraph * gf, const char * filename) {
    char color[16];

    FILE * fp = fopen(filename, "w");
    DRAGON_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = LR;\n");

    for (int i = 0; i < gb->n_nodes; i++) {
        struct dragon_tensor * node = gb->nodes[i];

        if (dragon_graph_get_parent(gb, node) != NULL) {
            continue;
        }

        if (node->is_param) {
            snprintf(color, sizeof(color), "yellow");
        } else if (node->grad) {
            if (dragon_graph_find(gf, node)) {
                snprintf(color, sizeof(color), "green");
            } else {
                snprintf(color, sizeof(color), "lightblue");
            }
        } else {
            snprintf(color, sizeof(color), "white");
        }

        fprintf(fp, "  \"%p\" [ \
style = filled; fillcolor = %s; shape = record; \
label=\"%d [%d, %d] | <x>%s",
                (void *) node, color,
                i, node->ne[0], node->ne[1],
                DRAGON_OP_SYMBOL[node->op]);

        if (node->grad) {
            fprintf(fp, " | <g>%s\"; ]\n", DRAGON_OP_SYMBOL[node->grad->op]);
        } else {
            fprintf(fp, "\"; ]\n");
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct dragon_tensor * node = gb->leafs[i];

        snprintf(color, sizeof(color), "pink");

        if (dragon_nelements(node) == 1) {
            fprintf(fp, "  \"%p\" [ \
style = filled; fillcolor = %s; shape = record; \
label=\"<x>%.1e\"; ]\n",
                    (void *) node, color, dragon_get_f32_1d(node, 0));
        } else {
            fprintf(fp, "  \"%p\" [ \
style = filled; fillcolor = %s; shape = record; \
label=\"<x>CONST %d [%d, %d]\"; ]\n",
                    (void *) node, color,
                    i, node->ne[0], node->ne[1]);
        }
    }

    for (int i = 0; i < gb->n_nodes; i++) {
        struct dragon_tensor * node = gb->nodes[i];

        struct dragon_tensor * parent = dragon_graph_get_parent(gb, node);

        if (node->src0) {
            struct dragon_tensor * parent0 = dragon_graph_get_parent(gb, node->src0);

            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"x\"; ]\n",
                    parent0 ? (void *) parent0 : (void *) node->src0,
                    parent0 ? "g" : "x",
                    parent ? (void *) parent : (void *) node,
                    parent ? "g" : "x",
                    parent ? "empty" : "vee",
                    parent ? "dashed" : "solid");
        }

        if (node->src1) {
            struct dragon_tensor * parent1 = dragon_graph_get_parent(gb, node->src1);

            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"y\"; ]\n",
                    parent1 ? (void *) parent1 : (void *) node->src1,
                    parent1 ? "g" : "x",
                    parent ? (void *) parent : (void *) node,
                    parent ? "g" : "x",
                    parent ? "empty" : "vee",
                    parent ? "dashed" : "solid");
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct dragon_tensor * node = gb->leafs[i];

        if (node->src0) {
            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"x\"; ]\n",
                    (void *) node->src0, "x",
                    (void *) node, "x");
        }

        if (node->src1) {
            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"y\"; ]\n",
                    (void *) node->src1, "x",
                    (void *) node, "x");
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    DRAGON_PRINT("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

static void dragon_opt_set_params(int np, struct dragon_tensor * const ps[], const float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int ne = dragon_nelements(ps[p]) ;
        // TODO: add function to set tensor from array
        for (int j = 0; j < ne; ++j) {
            dragon_set_f32_1d(ps[p], j, x[i++]);
        }
    }
}

static void dragon_opt_get_params(int np, struct dragon_tensor * const ps[], float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int ne = dragon_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int j = 0; j < ne; ++j) {
            x[i++] = dragon_get_f32_1d(ps[p], j);
        }
    }
}

static void dragon_opt_get_grad(int np, struct dragon_tensor * const ps[], float * g) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int ne = dragon_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int j = 0; j < ne; ++j) {
            g[i++] = dragon_get_f32_1d(ps[p]->grad, j);
        }
    }
}

//
// ADAM
//
//   ref: https://arxiv.org/pdf/1412.6980.pdf
//

static enum dragon_opt_result dragon_opt_adam(
        struct dragon_context * ctx,
        struct dragon_opt_params params,
        struct dragon_tensor * f,
        struct dragon_cgraph * gf,
        struct dragon_cgraph * gb) {
    DRAGON_ASSERT(dragon_is_scalar(f));

    gf->n_threads = params.n_threads;
    gb->n_threads = params.n_threads;

    // these will store the parameters we want to optimize
    struct dragon_tensor * ps[DRAGON_MAX_PARAMS];

    int np = 0;
    int nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->is_param) {
            DRAGON_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            DRAGON_ASSERT(np < DRAGON_MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += dragon_nelements(gf->nodes[i]);
        }
    }

    // constants
    const float alpha = params.adam.alpha;
    const float beta1 = params.adam.beta1;
    const float beta2 = params.adam.beta2;
    const float eps   = params.adam.eps;

    float * x  = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // view of the parameters
    float * g1 = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // gradient
    float * g2 = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // gradient squared
    float * m  = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // first moment
    float * v  = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // second moment
    float * mh = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // first moment hat
    float * vh = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // second moment hat

    float * pf = params.past > 0 ? dragon_new_tensor_1d(ctx, DATA_TYPE_F32, params.past)->data : NULL; // past function values

    // initialize
    dragon_vec_set_f32(nx, m, 0.0f);
    dragon_vec_set_f32(nx, v, 0.0f);

    // update view
    dragon_opt_get_params(np, ps, x);

    // compute the function value
    dragon_graph_reset  (gf);
    dragon_set_f32      (f->grad, 1.0f);
    dragon_graph_compute(ctx, gb);

    float fx_prev = dragon_get_f32_1d(f, 0);
    if (pf) {
        pf[0] = fx_prev;
    }

    int n_no_improvement = 0;
    float fx_best = fx_prev;

    // run the optimizer
    for (int t = 0; t < params.adam.n_iter; ++t) {
        DRAGON_PRINT_DEBUG  ("=== iter %d ===\n", t);

        DRAGON_PRINT_DEBUG  ("f      = %10.6f\n", dragon_get_f32_1d(f, 0));
        DRAGON_PRINT_DEBUG_5("df/dx0 = %10.6f\n", dragon_get_f32_1d(ps[0]->grad, 0));
        DRAGON_PRINT_DEBUG_5("df/dx1 = %10.6f\n", dragon_get_f32_1d(ps[1]->grad, 0));

        for (int i = 0; i < np; ++i) {
            DRAGON_PRINT_DEBUG("param %d: %10.6f, g = %10.6f\n", i,
                    dragon_get_f32_1d(ps[i], 0), dragon_get_f32_1d(ps[i]->grad, 0));
        }

        const int64_t t_start_wall = dragon_time_us();
        const int64_t t_start_cpu = dragon_cycles();
        UNUSED(t_start_wall);
        UNUSED(t_start_cpu);

        {
            // update the gradient
            dragon_opt_get_grad(np, ps, g1);

            // m_t = beta1*m_t-1 + (1 - beta1)*g_t
            dragon_vec_scale_f32(nx, m, beta1);
            dragon_vec_mad_f32  (nx, m, g1, 1.0f - beta1);

            // g2 = g1^2
            dragon_vec_sqr_f32  (nx, g2, g1);

            // v_t = beta2*v_t-1 + (1 - beta2)*g_t^2
            dragon_vec_scale_f32(nx, v, beta2);
            dragon_vec_mad_f32  (nx, v, g2, 1.0f - beta2);

            // m^hat = m_t / (1 - beta1^t)
            // v^hat = v_t / (1 - beta2^t)
            // x_t = x_t-1 - alpha*m^hat/(sqrt(v^hat) + eps)
            dragon_vec_cpy_f32  (nx, mh, m);
            dragon_vec_cpy_f32  (nx, vh, v);

            dragon_vec_scale_f32(nx, mh, alpha/(1.0f - powf(beta1, t + 1)));
            dragon_vec_scale_f32(nx, vh,  1.0f/(1.0f - powf(beta2, t + 1)));

            dragon_vec_sqrt_f32 (nx, vh, vh);
            dragon_vec_acc1_f32 (nx, vh, eps);

            dragon_vec_div_f32  (nx, mh, mh, vh);
            dragon_vec_sub_f32  (nx, x,  x,  mh);

            // update the parameters
            dragon_opt_set_params(np, ps, x);
        }

        dragon_graph_reset  (gf);
        dragon_set_f32      (f->grad, 1.0f);
        dragon_graph_compute(ctx, gb);

        const float fx = dragon_get_f32_1d(f, 0);

        // check convergence
        if (fabsf(fx - fx_prev)/fx < params.adam.eps_f) {
            DRAGON_PRINT_DEBUG("converged\n");

            return DRAGON_OPT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= t) {
                const float rate = (pf[t%params.past] - fx)/fx;

                if (fabs(rate) < params.delta) {
                    return DRAGON_OPT_OK;
                }
            }

            pf[t%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx_best > fx) {
                fx_best = fx;
                n_no_improvement = 0;
            } else {
                ++n_no_improvement;

                if (n_no_improvement >= params.max_no_improvement) {
                    return DRAGON_OPT_OK;
                }
            }
        }

        fx_prev = fx;

        {
            const int64_t t_end_cpu = dragon_cycles();
            DRAGON_PRINT_DEBUG("time iter:      %5.3f s\n", ((float)(t_end_cpu - t_start_cpu))/CLOCKS_PER_SEC);
            UNUSED(t_end_cpu);

            const int64_t t_end_wall = dragon_time_us();
            DRAGON_PRINT_DEBUG("wall time iter: %5.3f s\n", (t_end_wall - t_start_wall)/1e6);
            UNUSED(t_end_wall);
        }
    }

    return DRAGON_OPT_DID_NOT_CONVERGE;
}

//
// L-BFGS
//
// the L-BFGS implementation below is based on the following implementation:
//
//   https://github.com/chokkan/liblbfgs
//

struct dragon_lbfgs_iteration_data {
    float alpha;
    float ys;
    float * s;
    float * y;
};

static enum dragon_opt_result linesearch_backtracking(
        struct dragon_context * ctx,
        const struct dragon_opt_params * params,
        int nx,
        float * x,
        float * fx,
        float * g,
        float * d,
        float * step,
        const float * xp,
        struct dragon_tensor * f,
        struct dragon_cgraph * gf,
        struct dragon_cgraph * gb,
        const int np,
        struct dragon_tensor * ps[]) {
    int count = 0;

    float width  = 0.0f;
    float dg     = 0.0f;
    float finit  = 0.0f;
    float dginit = 0.0f;
    float dgtest = 0.0f;

    const float dec = 0.5f;
    const float inc = 2.1f;

    if (*step <= 0.) {
        return DRAGON_LINESEARCH_INVALID_PARAMETERS;
    }

    // compute the initial gradient in the search direction
    dragon_vec_dot_f32(nx, &dginit, g, d);

    // make sure that d points to a descent direction
    if (0 < dginit) {
        return DRAGON_LINESEARCH_FAIL;
    }

    // initialize local variables
    finit = *fx;
    dgtest = params->lbfgs.ftol*dginit;

    while (true) {
        dragon_vec_cpy_f32(nx, x, xp);
        dragon_vec_mad_f32(nx, x, d, *step);

        // evaluate the function and gradient values
        {
            dragon_opt_set_params(np, ps, x);

            dragon_graph_reset  (gf);
            dragon_set_f32      (f->grad, 1.0f);
            dragon_graph_compute(ctx, gb);

            dragon_opt_get_grad(np, ps, g);

            *fx = dragon_get_f32_1d(f, 0);
        }

        ++count;

        if (*fx > finit + (*step)*dgtest) {
            width = dec;
        } else {
            // Armijo condition is satisfied
            if (params->lbfgs.linesearch == DRAGON_LINESEARCH_BACKTRACKING_ARMIJO) {
                return count;
            }

            dragon_vec_dot_f32(nx, &dg, g, d);

            // check the Wolfe condition
            if (dg < params->lbfgs.wolfe * dginit) {
                width = inc;
            } else {
                if(params->lbfgs.linesearch == DRAGON_LINESEARCH_BACKTRACKING_WOLFE) {
                    // regular Wolfe conditions
                    return count;
                }

                if(dg > -params->lbfgs.wolfe*dginit) {
                    width = dec;
                } else {
                    // strong Wolfe condition (DRAGON_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
                    return count;
                }
                return count;
            }
        }

        if (*step < params->lbfgs.min_step) {
            return DRAGON_LINESEARCH_MINIMUM_STEP;
        }
        if (*step > params->lbfgs.max_step) {
            return DRAGON_LINESEARCH_MAXIMUM_STEP;
        }
        if (params->lbfgs.max_linesearch <= count) {
            return DRAGON_LINESEARCH_MAXIMUM_ITERATIONS;
        }

        (*step) *= width;
    }

    return DRAGON_LINESEARCH_FAIL;
}

static enum dragon_opt_result dragon_opt_lbfgs(
        struct dragon_context * ctx,
        struct dragon_opt_params params,
        struct dragon_tensor * f,
        struct dragon_cgraph * gf,
        struct dragon_cgraph * gb) {
    if (params.lbfgs.linesearch == DRAGON_LINESEARCH_BACKTRACKING_WOLFE ||
        params.lbfgs.linesearch == DRAGON_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
        if (params.lbfgs.wolfe <= params.lbfgs.ftol || 1. <= params.lbfgs.wolfe) {
            return DRAGON_OPT_INVALID_WOLFE;
        }
    }

    gf->n_threads = params.n_threads;
    gb->n_threads = params.n_threads;

    const int m = params.lbfgs.m;

    // these will store the parameters we want to optimize
    struct dragon_tensor * ps[DRAGON_MAX_PARAMS];

    int np = 0;
    int nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->is_param) {
            DRAGON_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            DRAGON_ASSERT(np < DRAGON_MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += dragon_nelements(gf->nodes[i]);
        }
    }

    float * x  = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // current parameters
    float * xp = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // previous parameters
    float * g  = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // current gradient
    float * gp = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // previous gradient
    float * d  = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data; // search direction

    float * pf = params.past > 0 ? dragon_new_tensor_1d(ctx, DATA_TYPE_F32, params.past)->data : NULL; // past function values

    float fx    = 0.0f; // cost function value
    float xnorm = 0.0f; // ||x||
    float gnorm = 0.0f; // ||g||
    float step  = 0.0f;

    // initialize x from the graph nodes
    dragon_opt_get_params(np, ps, x);

    // the L-BFGS memory
    struct dragon_lbfgs_iteration_data * lm = alloca(sizeof(struct dragon_lbfgs_iteration_data)*m);

    for (int i = 0; i < m; ++i) {
        lm[i].alpha = 0.0f;
        lm[i].ys    = 0.0f;
        lm[i].s     = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data;
        lm[i].y     = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, nx)->data;
    }

    // evaluate the function value and its gradient
    {
        dragon_opt_set_params(np, ps, x);

        dragon_graph_reset  (gf);
        dragon_set_f32      (f->grad, 1.0f);
        dragon_graph_compute(ctx, gb);

        dragon_opt_get_grad(np, ps, g);

        fx = dragon_get_f32_1d(f, 0);
    }

    if (pf) {
        pf[0] = fx;
    }

    float fx_best = fx;

    // search direction = -gradient
    dragon_vec_neg_f32(nx, d, g);

    // ||x||, ||g||
    dragon_vec_norm_f32(nx, &xnorm, x);
    dragon_vec_norm_f32(nx, &gnorm, g);

    if (xnorm < 1.0f) {
        xnorm = 1.0f;
    }

    // already optimized
    if (gnorm/xnorm <= params.lbfgs.eps) {
        return DRAGON_OPT_OK;
    }

    // initial step
    dragon_vec_norm_inv_f32(nx, &step, d);

    int j                = 0;
    int k                = 1;
    int ls               = 0;
    int end              = 0;
    int bound            = 0;
    int n_no_improvement = 0;

    float ys   = 0.0f;
    float yy   = 0.0f;
    float beta = 0.0f;

    while (true) {
        // store the current position and gradient vectors
        dragon_vec_cpy_f32(nx, xp, x);
        dragon_vec_cpy_f32(nx, gp, g);

        ls = linesearch_backtracking(ctx, &params, nx, x, &fx, g, d, &step, xp, f, gf, gb, np, ps);

        if (ls < 0) {
            // linesearch failed - go back to the previous point and return
            dragon_vec_cpy_f32(nx, x, xp);
            dragon_vec_cpy_f32(nx, g, gp);

            return ls;
        }

        dragon_vec_norm_f32(nx, &xnorm, x);
        dragon_vec_norm_f32(nx, &gnorm, g);

        DRAGON_PRINT_DEBUG("f = %10.6f\n", dragon_get_f32_1d(f, 0));

        if (xnorm < 1.0) {
            xnorm = 1.0;
        }
        if (gnorm/xnorm <= params.lbfgs.eps) {
            // converged
            return DRAGON_OPT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= k) {
                const float rate = (pf[k%params.past] - fx)/fx;

                if (fabs(rate) < params.delta) {
                    return DRAGON_OPT_OK;
                }
            }

            pf[k%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx < fx_best) {
                fx_best = fx;
                n_no_improvement = 0;
            } else {
                n_no_improvement++;

                if (n_no_improvement >= params.max_no_improvement) {
                    return DRAGON_OPT_OK;
                }
            }
        }

        if (params.lbfgs.n_iter != 0 && params.lbfgs.n_iter < k + 1) {
            // reached the maximum number of iterations
            return DRAGON_OPT_DID_NOT_CONVERGE;
        }

        // update vectors s and y:
        //   s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
        //   y_{k+1} = g_{k+1} - g_{k}.
        //
        dragon_vec_sub_f32(nx, lm[end].s, x, xp);
        dragon_vec_sub_f32(nx, lm[end].y, g, gp);

        // compute scalars ys and yy:
        //     ys = y^t \cdot s    -> 1 / \rho.
        //     yy = y^t \cdot y.
        //
        dragon_vec_dot_f32(nx, &ys, lm[end].y, lm[end].s);
        dragon_vec_dot_f32(nx, &yy, lm[end].y, lm[end].y);

        lm[end].ys = ys;

        // find new search direction
        //   ref: https://en.wikipedia.org/wiki/Limited-memory_BFGS

        bound = (m <= k) ? m : k;
        k++;
        end = (end + 1)%m;

        // initialize search direction with -g
        dragon_vec_neg_f32(nx, d, g);

        j = end;
        for (int i = 0; i < bound; ++i) {
            j = (j + m - 1) % m;
            // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
            dragon_vec_dot_f32(nx, &lm[j].alpha, lm[j].s, d);
            lm[j].alpha /= lm[j].ys;
            // q_{i} = q_{i+1} - \alpha_{i} y_{i}
            dragon_vec_mad_f32(nx, d, lm[j].y, -lm[j].alpha);
        }

        dragon_vec_scale_f32(nx, d, ys/yy);

        for (int i = 0; i < bound; ++i) {
            // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
            dragon_vec_dot_f32(nx, &beta, lm[j].y, d);
            beta /= lm[j].ys;
            // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
            dragon_vec_mad_f32(nx, d, lm[j].s, lm[j].alpha - beta);
            j = (j + 1)%m;
        }

        step = 1.0;
    }

    return DRAGON_OPT_DID_NOT_CONVERGE;
}

struct dragon_opt_params dragon_opt_default_params(enum dragon_opt_type type) {
    struct dragon_opt_params result;

    switch (type) {
        case DRAGON_OPT_ADAM:
            {
                result = (struct dragon_opt_params) {
                    .type      = DRAGON_OPT_ADAM,
                    .n_threads = 1,
                    .past      = 0,
                    .delta     = 1e-5f,

                    .max_no_improvement = 100,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .adam = {
                        .n_iter = 10000,
                        .alpha  = 0.001f,
                        .beta1  = 0.9f,
                        .beta2  = 0.999f,
                        .eps    = 1e-8f,
                        .eps_f  = 1e-5f,
                        .eps_g  = 1e-3f,
                    },
                };
            } break;
        case DRAGON_OPT_LBFGS:
            {
                result = (struct dragon_opt_params) {
                    .type      = DRAGON_OPT_LBFGS,
                    .n_threads = 1,
                    .past      = 0,
                    .delta     = 1e-5f,

                    .max_no_improvement = 0,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .lbfgs = {
                        .m              = 6,
                        .n_iter         = 100,
                        .max_linesearch = 20,

                        .eps      = 1e-5f,
                        .ftol     = 1e-4f,
                        .wolfe    = 0.9f,
                        .min_step = 1e-20f,
                        .max_step = 1e+20f,

                        .linesearch = DRAGON_LINESEARCH_DEFAULT,
                    },
                };
            } break;
    }

    return result;
}

enum dragon_opt_result dragon_opt(
        struct dragon_context * ctx,
        struct dragon_opt_params params,
        struct dragon_tensor * f) {
    bool free_ctx = false;
    if (ctx == NULL) {
        struct dragon_init_params params_ctx = {
            .mem_size   = 16*1024*1024,
            .mem_buffer = NULL,
        };

        ctx = dragon_init(params_ctx);
        if (ctx == NULL) {
            return DRAGON_OPT_NO_CONTEXT;
        }

        free_ctx = true;
    }

    enum dragon_opt_result result = DRAGON_OPT_OK;

    // build forward + backward compute graphs
    struct dragon_cgraph gf = dragon_build_forward (f);
    struct dragon_cgraph gb = dragon_build_backward(ctx, &gf, false);

    switch (params.type) {
        case DRAGON_OPT_ADAM:
            {
                result = dragon_opt_adam(ctx, params, f, &gf, &gb);
            } break;
        case DRAGON_OPT_LBFGS:
            {
                result = dragon_opt_lbfgs(ctx, params, f, &gf, &gb);
            } break;
    }

    if (params.print_forward_graph) {
        dragon_graph_print   (&gf);
        dragon_graph_dump_dot(&gf, NULL, "opt-forward.dot");
    }

    if (params.print_backward_graph) {
        dragon_graph_print   (&gb);
        dragon_graph_dump_dot(&gb, &gf, "opt-backward.dot");
    }

    if (free_ctx) {
        dragon_free(ctx);
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////

int dragon_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_neon(void) {
#if defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_blas(void) {
#if defined(DRAGON_USE_ACCELERATE) || defined(DRAGON_USE_OPENBLAS)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int dragon_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////
