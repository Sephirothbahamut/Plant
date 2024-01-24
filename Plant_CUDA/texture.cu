#pragma once
#include "texture.h"

namespace utils::cuda
	{
	template <typename pixel_t>
	texture<pixel_t>::texture(const utils::math::vec2s& sizes) :
		sizes{sizes},
		data{sizes.x * sizes.y}
		{
		}

	template <typename pixel_t>
	texture<pixel_t>::texture(const utils::graphics::image<pixel_t>& image) :
		sizes{image.sizes()},
		data{image.begin(), image.end()}
		{}

	template <typename pixel_t>
	texture<pixel_t>::~texture() = default;

	template <typename pixel_t>
	utils::cuda::kernel::texture<pixel_t> texture<pixel_t>::get_kernel_side() noexcept 
		{
		std::span<pixel_t> device_span{thrust::raw_pointer_cast(data.data()), data.size()};
		return {sizes, device_span};
		}
	}

template class utils::cuda::texture<utils::graphics::colour::rgba_f>;
template class utils::cuda::texture<utils::graphics::colour::rgba_d>;
template class utils::cuda::texture<utils::graphics::colour::rgba_u>;
template class utils::cuda::texture<utils::graphics::colour::rgb_f >;
template class utils::cuda::texture<utils::graphics::colour::rgb_d >;
template class utils::cuda::texture<utils::graphics::colour::rgb_u >;