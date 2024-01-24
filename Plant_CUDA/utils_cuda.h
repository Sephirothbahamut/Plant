#pragma once
#include <iostream>
#include <stdexcept>

#include "cuda_runtime.h"

namespace utils::cuda
	{
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
	}