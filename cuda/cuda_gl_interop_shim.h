/*
 * CUDA-OpenGL interop shim with HIP support
 *
 * When building with HIP (JETSON_USE_HIP), this header maps the CUDA GL interop
 * functions used by this project to the corresponding HIP GL interop calls.
 * It intentionally uses wrapper functions to avoid depending on the HIP
 * graphics resource concrete type from translation units that expect CUDA types.
 */

#pragma once

#include "cuda_runtime_compat.h"

#ifdef JETSON_USE_HIP
#include <GL/gl.h>

#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_gl_interop.h>
#else
// Host-only compilation. If HIP headers are present (via JUT_INCLUDE_HIP_HEADERS),
// rely on their declarations. Otherwise, forward-declare minimal prototypes.
#if !defined(JUT_INCLUDE_HIP_HEADERS)
extern "C" {
struct hipGraphicsResource;
hipError_t hipGraphicsGLRegisterBuffer(struct hipGraphicsResource** resource, GLuint buffer, unsigned int flags);
hipError_t hipGraphicsGLRegisterImage(struct hipGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags);
hipError_t hipGraphicsUnregisterResource(struct hipGraphicsResource* resource);
hipError_t hipGraphicsResourceSetMapFlags(struct hipGraphicsResource* resource, unsigned int flags);
hipError_t hipGraphicsMapResources(int count, struct hipGraphicsResource** resources, hipStream_t stream);
hipError_t hipGraphicsUnmapResources(int count, struct hipGraphicsResource** resources, hipStream_t stream);
hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, struct hipGraphicsResource* resource);
}
#endif
#endif

// Provide a forward-declared CUDA graphics resource type to preserve source
// compatibility. It's only used by pointer in this project.
struct cudaGraphicsResource;

// Map CUDA interop API to HIP via thin wrappers with safe casting
static inline cudaError_t cudaGraphicsGLRegisterBuffer(
    cudaGraphicsResource** resource,
    GLuint buffer,
    unsigned int flags)
{
    return hipGraphicsGLRegisterBuffer(reinterpret_cast<hipGraphicsResource**>(resource), buffer, flags);
}

static inline cudaError_t cudaGraphicsGLRegisterImage(
    cudaGraphicsResource** resource,
    GLuint image,
    GLenum target,
    unsigned int flags)
{
    return hipGraphicsGLRegisterImage(reinterpret_cast<hipGraphicsResource**>(resource), image, target, flags);
}

static inline cudaError_t cudaGraphicsUnregisterResource(
    cudaGraphicsResource* resource)
{
    return hipGraphicsUnregisterResource(reinterpret_cast<hipGraphicsResource*>(resource));
}

static inline cudaError_t cudaGraphicsResourceSetMapFlags(
    cudaGraphicsResource* resource,
    unsigned int flags)
{
    #if defined(__HIP_DEVICE_COMPILE__)
    return hipGraphicsResourceSetMapFlags(reinterpret_cast<hipGraphicsResource*>(resource), flags);
    #else
    // Some HIP distributions do not expose hipGraphicsResourceSetMapFlags at host compile time.
    // Treat as a no-op and report success, matching CUDA behavior when flags are unused by caller.
    (void)resource; (void)flags;
    return hipSuccess;
    #endif
}

static inline cudaError_t cudaGraphicsMapResources(
    int count,
    cudaGraphicsResource** resources,
    cudaStream_t stream = 0)
{
    return hipGraphicsMapResources(count, reinterpret_cast<hipGraphicsResource**>(resources), stream);
}

static inline cudaError_t cudaGraphicsUnmapResources(
    int count,
    cudaGraphicsResource** resources,
    cudaStream_t stream = 0)
{
    return hipGraphicsUnmapResources(count, reinterpret_cast<hipGraphicsResource**>(resources), stream);
}

static inline cudaError_t cudaGraphicsResourceGetMappedPointer(
    void** devPtr,
    size_t* size,
    cudaGraphicsResource* resource)
{
    return hipGraphicsResourceGetMappedPointer(devPtr, size, reinterpret_cast<hipGraphicsResource*>(resource));
}

#else  // JETSON_USE_HIP

// Default case: include CUDA's official header
#include <cuda_gl_interop.h>

#endif  // JETSON_USE_HIP
