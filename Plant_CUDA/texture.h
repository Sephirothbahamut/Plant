#pragma once

#include <memory>
#include <filesystem>

#include <utils/math/ranged.h>
#include <utils/graphics/colour.h>
#include <utils/matrix_interface.h>
#include <utils/containers/matrix_dyn.h>

#include <thrust/device_vector.h>
#include "image.h"

namespace utils::CUDA
	{
	template <typename pixel_t = utils::graphics::colour::rgba_f>
	class texture
		{
		public:
			texture(const utils::math::vec2s& size);
			texture(const utils::graphics::image<pixel_t>& image);
			~texture();
			//TODO copy and move constructor must make sure the span in the matrix points to the right container

			utils::matrix_wrapper<std::span<pixel_t>> get_kernel_side() noexcept;

		private:
			utils::math::vec2s sizes;
			thrust::device_vector<pixel_t> data;
		};
	}