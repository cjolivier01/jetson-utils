/*
 * CUDA/HIP runtime compatibility shim
 *
 * When JETSON_USE_HIP is defined, this header maps common CUDA runtime APIs
 * and types used in this project to their HIP equivalents so the same sources
 * can compile for AMD GPUs under ROCm/HIP. Otherwise it includes CUDA headers.
 */

#pragma once

#ifdef JETSON_USE_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// Basic type aliases
#define cudaError_t                 hipError_t
#define cudaSuccess                 hipSuccess
#define cudaErrorInvalidValue       hipErrorInvalidValue
#define cudaErrorNotSupported       hipErrorNotSupported

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
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize

#define cudaEventCreate             hipEventCreate
#define cudaEventCreateWithFlags    hipEventCreateWithFlags
#define cudaEventDestroy            hipEventDestroy
#define cudaEventRecord             hipEventRecord
#define cudaEventSynchronize        hipEventSynchronize
#define cudaEventElapsedTime        hipEventElapsedTime

#define cudaGetLastError            hipGetLastError
#define cudaPeekAtLastError         hipPeekAtLastError
#define cudaGetErrorString          hipGetErrorString

#define cudaMalloc                  hipMalloc
#define cudaMallocPitch             hipMallocPitch
#define cudaMallocArray             hipMallocArray
#define cudaFree                    hipFree

#define cudaHostAlloc               hipHostMalloc
#define cudaFreeHost                hipHostFree

#define cudaMemset                  hipMemset
#define cudaMemsetAsync             hipMemsetAsync
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemcpy2D                hipMemcpy2D
#define cudaMemcpy2DAsync           hipMemcpy2DAsync
#define cudaMemcpy2DFromArray       hipMemcpy2DFromArray
#define cudaMemcpy2DFromArrayAsync  hipMemcpy2DFromArrayAsync

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

#else  // JETSON_USE_HIP

#include <cuda_runtime.h>

#endif  // JETSON_USE_HIP

