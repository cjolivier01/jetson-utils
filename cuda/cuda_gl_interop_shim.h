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
#include <hip/hip_gl_interop.h>
#include <GL/gl.h>

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
    return hipGraphicsResourceSetMapFlags(reinterpret_cast<hipGraphicsResource*>(resource), flags);
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

