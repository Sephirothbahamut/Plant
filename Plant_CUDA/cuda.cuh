#pragma once

#include <span>
#include <vector>
#include <optional>
#include <stdexcept>

#include <utils/math/vec2.h>
#include <utils/math/vec3.h>
#include <utils/compilation/CUDA.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __intellisense__
void __syncthreads() {}
#endif

namespace utils::cuda
	{
	namespace kernel::coordinates
		{
		namespace local
			{
			__device__ inline unsigned int x() noexcept { return threadIdx.x; }
			__device__ inline unsigned int y() noexcept { return threadIdx.y; }
			__device__ inline unsigned int z() noexcept { return threadIdx.z; }

			__device__ inline utils::math::vec3u vec2() noexcept { return {x(), y()}; }
			__device__ inline utils::math::vec3u vec3() noexcept { return {x(), y(), z()}; }
			}
		namespace total
			{
			__device__ inline unsigned int x() noexcept { return blockIdx.x * blockDim.x + threadIdx.x; }
			__device__ inline unsigned int y() noexcept { return blockIdx.y * blockDim.y + threadIdx.y; }
			__device__ inline unsigned int z() noexcept { return blockIdx.z * blockDim.z + threadIdx.z; }

			__device__ inline utils::math::vec3u vec2() noexcept { return {x(), y()}; }
			__device__ inline utils::math::vec3u vec3() noexcept { return {x(), y(), z()}; }
			}
		}
	
	struct dim3
		{
		dim3(unsigned int x                                ) noexcept : x{    x}, y{    1}, z{    1} {}
		dim3(unsigned int x, unsigned int y                ) noexcept : x{    x}, y{    y}, z{    1} {}
		dim3(unsigned int x, unsigned int y, unsigned int z) noexcept : x{    x}, y{    y}, z{    z} {}
		dim3(size_t       x                                ) noexcept : x{static_cast<unsigned int>(x)}, y{static_cast<unsigned int>(1)}, z{static_cast<unsigned int>(1)} {}
		dim3(size_t       x, size_t       y                ) noexcept : x{static_cast<unsigned int>(x)}, y{static_cast<unsigned int>(y)}, z{static_cast<unsigned int>(1)} {}
		dim3(size_t       x, size_t       y, size_t       z) noexcept : x{static_cast<unsigned int>(x)}, y{static_cast<unsigned int>(y)}, z{static_cast<unsigned int>(z)} {}
		dim3(::dim3             xyz                        ) noexcept : x{xyz.x}, y{xyz.y}, z{xyz.z} {}
		dim3(utils::math::vec2u xy                         ) noexcept : x{xy .x}, y{xy .y}, z{    1} {}
		dim3(utils::math::vec3u xyz                        ) noexcept : x{xyz.x}, y{xyz.y}, z{xyz.z} {}
		dim3(utils::math::vec2s xy                         ) noexcept : x{static_cast<unsigned int>(xy .x)}, y{static_cast<unsigned int>(xy .y)}, z{static_cast<unsigned int>(    1)} {}
		dim3(utils::math::vec3s xyz                        ) noexcept : x{static_cast<unsigned int>(xyz.x)}, y{static_cast<unsigned int>(xyz.y)}, z{static_cast<unsigned int>(xyz.z)} {}
		unsigned int x{1};
		unsigned int y{1};
		unsigned int z{1};
	
		operator utils::math::vec2u() const noexcept { return {x, y   }; }
		operator utils::math::vec3u() const noexcept { return {x, y, z}; }
		operator utils::math::vec2s() const noexcept { return {x, y   }; }
		operator utils::math::vec3s() const noexcept { return {x, y, z}; }
		operator ::dim3            () const noexcept { return {x, y, z}; }
		};
	
	inline bool check(const cudaError_t status) noexcept
		{
		if (status != cudaSuccess)
			{
			std::cout << cudaGetErrorString(status) << std::endl;
			return false;
			}
		return true;
		}
	inline void check_throwing(const cudaError_t status)
		{
		if (status != cudaSuccess)
			{
			throw std::runtime_error{cudaGetErrorString(status)};
			}
		}

	namespace device
		{
		struct params_t
			{
			struct threads_t
				{
				/// <summary> How many threads to allocate per each block. Aka block size. </summary>
				dim3 per_block{1u};
				/// <summary> How many blocks to allocate. Aka grid size. </summary>
				dim3 blocks{1u};
				/// <summary> Deduce blocks count/grid size from threads per block and required threads total. </summary>
				inline static threads_t deduce(const dim3& per_block, const dim3& total_required) noexcept
					{
					return
						{
						.per_block{per_block},
						.blocks
							{
							static_cast<unsigned int>(std::ceil(static_cast<float>(total_required.x) / static_cast<float>(per_block.x))),
							static_cast<unsigned int>(std::ceil(static_cast<float>(total_required.y) / static_cast<float>(per_block.y))),
							static_cast<unsigned int>(std::ceil(static_cast<float>(total_required.z) / static_cast<float>(per_block.z)))
							}
						};
					}
				};
			threads_t threads;
			size_t shared_memory_bytes{0};
			};
		
		template <typename kernel_t, typename ...Args>
		void call(const kernel_t kernel, const params_t& params, Args&&... args) noexcept
			{
			#ifndef __INTELLISENSE__
			kernel<<<params.threads.blocks, params.threads.per_block, params.shared_memory_bytes>>>(std::forward<Args>(args)...);
			#endif
			}
		}
	
	
	//struct graphics_resource
	//	{
	//	graphics_resource(cudaGraphicsResource* cuda_resource) : cuda_resource{cuda_resource}
	//		{
	//		cudaGraphicsMapResources(1, &cuda_resource);
	//		// map OpenGL buffer object for writing from CUDA
	//		cudaGraphicsResourceGetMappedPointer((void**)&data_ptr, &data_size, cuda_resource);
	//		}
	//
	//	cudaGraphicsResource* cuda_resource;
	//	float4* data_ptr;
	//	size_t data_size;
	//
	//	~graphics_resource()
	//		{
	//		cudaGraphicsUnmapResources(1, &cuda_resource);
	//		}
	//	};
	}
