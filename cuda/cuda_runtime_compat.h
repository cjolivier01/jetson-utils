/*
 * CUDA/HIP runtime compatibility shim
 *
 * When JETSON_USE_HIP or JUT_INCLUDE_HIP_HEADERS is defined, this header maps
 * common CUDA runtime APIs and types used in this project to their HIP
 * equivalents so the same sources can compile for AMD GPUs under ROCm/HIP.
 * Otherwise it includes CUDA headers.
 */

#pragma once

#if defined(JETSON_USE_HIP) || defined(JUT_INCLUDE_HIP_HEADERS)

// If building with HIP but the target platform macro isn't set yet, default to AMD.
// For device compilation (hipcc), include the full HIP runtime headers.
#if defined(__HIPCC__)
    #if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
    #define __HIP_PLATFORM_AMD__ 1
    #endif
    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>
#else
    // Host-only compilation path (e.g., GCC/Clang, not hipcc): avoid including
    // HIP's vector type headers that require clang extensions. Provide minimal
    // forward declarations and type aliases sufficient for host code.

    #ifdef JUT_INCLUDE_HIP_HEADERS
        // Some host translation units opt-in to including HIP headers even when not using hipcc,
        // in order to share the same vector types and function signatures as device code.
        #if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
        #define __HIP_PLATFORM_AMD__ 1
        #endif
        #include <hip/hip_runtime.h>
        #include <hip/hip_runtime_api.h>
        #ifdef __noinline__
        #undef __noinline__
        #endif
    #else

    // Basic opaque handle types
    #include <cstddef>
    #ifndef __host__
    #define __host__
    #endif
    #ifndef __device__
    #define __device__
    #endif
    #ifndef __forceinline__
    #define __forceinline__ inline
    #endif
    typedef int                         hipError_t;
    typedef struct ihipStream_t*        hipStream_t;
    typedef struct ihipEvent_t*         hipEvent_t;
    typedef struct hipDeviceProp_t      hipDeviceProp_t;

    // Memory kinds & flags used
    typedef enum hipMemcpyKind {
        hipMemcpyHostToDevice = 1,
        hipMemcpyDeviceToHost = 2,
        hipMemcpyDeviceToDevice = 3,
    } hipMemcpyKind;

    // Host allocation flags
    #ifndef hipHostMallocDefault
    #define hipHostMallocDefault 0x00
    #endif
    #ifndef hipHostMallocMapped
    #define hipHostMallocMapped  0x02
    #endif

    // Stream flags
    #ifndef hipMemAttachGlobal
    #define hipMemAttachGlobal 0
    #endif

    // Error codes (subset)
    #ifndef hipSuccess
    #define hipSuccess                    0
    #define hipErrorInvalidValue          1
    #define hipErrorInvalidDevicePointer  17
    #define hipErrorNotSupported          801
    #define hipErrorUnknown               999
    #define hipErrorMemoryAllocation      2
    #define hipErrorInvalidMemcpyDirection 21
    #define hipErrorInvalidSymbol         200
    #endif

    extern "C" {
    // Minimal function prototypes referenced by host code
    const char* hipGetErrorString(hipError_t);
    hipError_t hipGetLastError(void);
    hipError_t hipPeekAtLastError(void);
    hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
    hipError_t hipHostFree(void* ptr);
    hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
    hipError_t hipMalloc(void** devPtr, size_t size);
    hipError_t hipFree(void* devPtr);
    hipError_t hipMemset(void* dst, int value, size_t count);
    hipError_t hipMemsetAsync(void* dst, int value, size_t count, hipStream_t stream);
    hipError_t hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind);
    hipError_t hipMemcpyAsync(void* dst, const void* src, size_t count, hipMemcpyKind kind, hipStream_t stream);
    hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
    hipError_t hipDeviceSynchronize(void);
    hipError_t hipStreamCreate(hipStream_t* stream);
    hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);
    hipError_t hipStreamDestroy(hipStream_t stream);
    hipError_t hipStreamSynchronize(hipStream_t stream);
    hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
    hipError_t hipEventCreate(hipEvent_t* event);
    hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags);
    hipError_t hipEventDestroy(hipEvent_t event);
    hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
    hipError_t hipEventQuery(hipEvent_t event);
    hipError_t hipEventSynchronize(hipEvent_t event);
    hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
    }

    // Minimal vector types and constructors used by headers in host-only builds
    #if !defined(__VECTOR_TYPES_H__)
    typedef struct { float x, y; }          float2;
    typedef struct { float x, y, z; }       float3;
    typedef struct { float x, y, z, w; }    float4;
    typedef struct { int x, y; }            int2;
    typedef struct { int x, y, z; }         int3;
    typedef struct { int x, y, z, w; }      int4;
    typedef struct { unsigned int x, y; }   uint2;
    typedef struct { unsigned int x, y, z; } uint3;
    typedef struct { unsigned int x, y, z, w; } uint4;
    typedef struct { unsigned char x, y; } uchar2;
    typedef struct { unsigned char x, y, z; } uchar3;
    typedef struct { unsigned char x, y, z, w; } uchar4;

    // Constructors
    inline __host__ __device__ float2 make_float2(float x, float y) { return float2{x, y}; }
    inline __host__ __device__ float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
    inline __host__ __device__ float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }
    inline __host__ __device__ int2   make_int2(int x, int y) { return int2{x, y}; }
    inline __host__ __device__ int3   make_int3(int x, int y, int z) { return int3{x, y, z}; }
    inline __host__ __device__ int4   make_int4(int x, int y, int z, int w) { return int4{x, y, z, w}; }
    inline __host__ __device__ uint2  make_uint2(unsigned int x, unsigned int y) { return uint2{x, y}; }
    inline __host__ __device__ uint3  make_uint3(unsigned int x, unsigned int y, unsigned int z) { return uint3{x, y, z}; }
    inline __host__ __device__ uint4  make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return uint4{x, y, z, w}; }
    inline __host__ __device__ uchar2 make_uchar2(unsigned char x, unsigned char y) { return uchar2{x, y}; }
    inline __host__ __device__ uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) { return uchar3{x, y, z}; }
    inline __host__ __device__ uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { return uchar4{x, y, z, w}; }
    #endif
    #endif // JUT_INCLUDE_HIP_HEADERS
#endif

// Basic type aliases
#define cudaError_t                 hipError_t
#define cudaSuccess                 hipSuccess
#define cudaErrorInvalidValue       hipErrorInvalidValue
#define cudaErrorInvalidDevicePointer hipErrorInvalidDevicePointer
#define cudaErrorNotSupported       hipErrorNotSupported
#define cudaErrorUnknown            hipErrorUnknown
#define cudaErrorMemoryAllocation    hipErrorMemoryAllocation
#define cudaErrorInvalidMemcpyDirection hipErrorInvalidMemcpyDirection
#define cudaErrorInvalidSymbol       hipErrorInvalidSymbol
#define cudaErrorNotYetImplemented    hipErrorNotSupported

#define cudaStream_t                hipStream_t
#define cudaEvent_t                 hipEvent_t

#define cudaDeviceProp              hipDeviceProp_t

#define cudaMemcpyKind              hipMemcpyKind
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice

// Runtime API mappings
#define cudaGetDeviceCount          hipGetDeviceCount
#define cudaGetDevice               hipGetDevice
#define cudaSetDevice               hipSetDevice
#define cudaGetDeviceProperties     hipGetDeviceProperties
#define cudaDeviceReset             hipDeviceReset

#define cudaDeviceSynchronize       hipDeviceSynchronize
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize
#define cudaStreamWaitEvent         hipStreamWaitEvent

#define cudaEventCreate             hipEventCreate
#define cudaEventCreateWithFlags    hipEventCreateWithFlags
#define cudaEventDestroy            hipEventDestroy
#define cudaEventRecord             hipEventRecord
#define cudaEventQuery              hipEventQuery
#define cudaEventSynchronize        hipEventSynchronize
#define cudaEventElapsedTime        hipEventElapsedTime

#define cudaGetLastError            hipGetLastError
#define cudaPeekAtLastError         hipPeekAtLastError
#define cudaGetErrorString          hipGetErrorString

#if defined(__HIP_DEVICE_COMPILE__)
    #define cudaMalloc                  hipMalloc
    #define cudaMallocPitch             hipMallocPitch
    #define cudaMallocArray             hipMallocArray
    #define cudaFree                    hipFree

    #define cudaHostAlloc               hipHostMalloc
    #define cudaFreeHost                hipHostFree
    #define cudaMallocHost              hipHostMalloc
    #define cudaHostAllocMapped         hipHostMallocMapped
    #define cudaHostGetDevicePointer    hipHostGetDevicePointer
#else
    // Host-only inline wrappers to satisfy C++ pointer conversions
    template<typename T>
    static inline hipError_t cudaMalloc(T** devPtr, size_t size) {
        return hipMalloc(reinterpret_cast<void**>(devPtr), size);
    }
    static inline hipError_t cudaFree(void* devPtr) { return hipFree(devPtr); }
    static inline hipError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
        return hipHostMalloc(ptr, size, flags);
    }
    static inline hipError_t cudaMallocHost(void** ptr, size_t size) {
        return hipHostMalloc(ptr, size, hipHostMallocDefault);
    }
    static inline hipError_t cudaFreeHost(void* ptr) { return hipHostFree(ptr); }
    template<typename T>
    static inline hipError_t cudaHostGetDevicePointer(T** devPtr, T* hstPtr, unsigned int flags) {
        return hipHostGetDevicePointer(reinterpret_cast<void**>(devPtr), reinterpret_cast<void*>(hstPtr), flags);
    }
#endif

// Constant/flag mappings (apply to both host/device builds)
#ifndef hipStreamDefault
#define hipStreamDefault 0x0
#endif
#ifndef hipStreamNonBlocking
#define hipStreamNonBlocking 0x1
#endif
#define cudaStreamDefault            hipStreamDefault
#define cudaStreamNonBlocking        hipStreamNonBlocking

#define cudaHostAllocMapped         hipHostMallocMapped

#define cudaMemset                  hipMemset
#define cudaMemsetAsync             hipMemsetAsync
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemcpy2D                hipMemcpy2D
#define cudaMemcpy2DAsync           hipMemcpy2DAsync
#define cudaMemcpy2DFromArray       hipMemcpy2DFromArray
#define cudaMemcpy2DFromArrayAsync  hipMemcpy2DFromArrayAsync

// Error code mappings for special cases used in code
#ifndef hipErrorNotReady
#define hipErrorNotReady 600
#endif
#define cudaErrorNotReady           hipErrorNotReady

// Channel/array/extent mappings
#define cudaChannelFormatDesc       hipChannelFormatDesc
#define cudaExtent                  hipExtent
#define cudaArray_t                 hipArray_t
#define cudaArrayGetInfo            hipArrayGetInfo

// Occupancy helpers
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
    hipOccupancyMaxActiveBlocksPerMultiprocessor

// Graphics (OpenGL) interop — implemented in cuda_gl_interop.h shim

// CUDA graphics flags used by this project
enum cudaGraphicsRegisterFlags {
    cudaGraphicsRegisterFlagsNone          = 0,
    cudaGraphicsRegisterFlagsReadOnly      = 1,
    cudaGraphicsRegisterFlagsWriteDiscard  = 2,
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,   // reserved, not used
    cudaGraphicsRegisterFlagsTextureGather = 8       // reserved, not used
};

enum cudaGraphicsMapFlags {
    cudaGraphicsMapFlagsNone      = 0,
    cudaGraphicsMapFlagsReadOnly  = 1,
    cudaGraphicsMapFlagsWriteDiscard = 2,
};

#else  // JETSON_USE_HIP || JUT_INCLUDE_HIP_HEADERS

#include <cuda_runtime.h>

#endif  // JETSON_USE_HIP || JUT_INCLUDE_HIP_HEADERS
