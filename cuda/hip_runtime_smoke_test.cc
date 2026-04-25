#include "cudaNormalize.h"
#include "cudaResize.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool almost_equal(float a, float b) {
	return std::fabs(a - b) < 1e-5f;
}

int failf(const char* message, size_t index, float expected, float actual) {
	std::fprintf(stderr, "%s at index %zu: expected %.6f, got %.6f\n", message, index, expected, actual);
	return 1;
}

template<typename T>
void release_device(T*& ptr) {
	if(ptr != nullptr) {
		const cudaError_t status = cudaFree(ptr);
		(void)status;
	}

	ptr = nullptr;
}

} // namespace

int main() {
	float* input = nullptr;
	float* normalized = nullptr;
	float* resized = nullptr;

	const std::vector<float> host_input = {
		0.0f, 1.0f,
		2.0f, 3.0f,
	};
	std::vector<float> host_normalized(4, -1.0f);
	std::vector<float> host_resized(16, -1.0f);
	const float expected_normalized[] = {
		0.0f, 10.0f,
		20.0f, 30.0f,
	};
	const float expected_resized[] = {
		0.0f, 0.0f, 10.0f, 10.0f,
		0.0f, 0.0f, 10.0f, 10.0f,
		20.0f, 20.0f, 30.0f, 30.0f,
		20.0f, 20.0f, 30.0f, 30.0f,
	};

	int rc = 1;

	if(CUDA_FAILED(cudaMalloc(&input, host_input.size() * sizeof(float))))
		goto done;

	if(CUDA_FAILED(cudaMalloc(&normalized, host_normalized.size() * sizeof(float))))
		goto done;

	if(CUDA_FAILED(cudaMalloc(&resized, host_resized.size() * sizeof(float))))
		goto done;

	if(CUDA_FAILED(cudaMemcpy(input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice)))
		goto done;

	if(CUDA_FAILED(cudaNormalize(input, make_float2(0.0f, 3.0f), normalized, make_float2(0.0f, 30.0f), 2, 2)))
		goto done;

	if(CUDA_FAILED(cudaDeviceSynchronize()))
		goto done;

	if(CUDA_FAILED(cudaMemcpy(host_normalized.data(), normalized, host_normalized.size() * sizeof(float), cudaMemcpyDeviceToHost)))
		goto done;

	for(size_t i = 0; i < host_normalized.size(); ++i) {
		if(!almost_equal(host_normalized[i], expected_normalized[i])) {
			rc = failf("cudaNormalize() mismatch", i, expected_normalized[i], host_normalized[i]);
			goto done;
		}
	}

	if(CUDA_FAILED(cudaResize(normalized, 2, 2, resized, 4, 4, FILTER_POINT)))
		goto done;

	if(CUDA_FAILED(cudaDeviceSynchronize()))
		goto done;

	if(CUDA_FAILED(cudaMemcpy(host_resized.data(), resized, host_resized.size() * sizeof(float), cudaMemcpyDeviceToHost)))
		goto done;

	for(size_t i = 0; i < host_resized.size(); ++i) {
		if(!almost_equal(host_resized[i], expected_resized[i])) {
			rc = failf("cudaResize() mismatch", i, expected_resized[i], host_resized[i]);
			goto done;
		}
	}

	rc = 0;

done:
	release_device(resized);
	release_device(normalized);
	release_device(input);
	return rc;
}
