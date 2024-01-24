#include "renderer.h"


#include "cuda.cuh"

__global__ void draw(utils::cuda::kernel::texture<utils::graphics::colour::rgba_u> render_target, utils::cuda::kernel::texture<utils::graphics::colour::rgba_f> tileset_texture)
	{
	const auto coords{utils::cuda::kernel::coordinates::total::vec2()};

	render_target[coords].r = static_cast<uint8_t>(coords.x);
	render_target[coords].g = static_cast<uint8_t>(coords.y);
	render_target[coords].b = static_cast<uint8_t>(255);
	render_target[coords].a = static_cast<uint8_t>(255);

	return;
	if (!render_target.validate_coords(coords) || !tileset_texture.validate_coords(coords)) { return; }

	render_target[coords].r = static_cast<uint8_t>(tileset_texture[coords].r * 255.f);
	render_target[coords].g = static_cast<uint8_t>(tileset_texture[coords].g * 255.f);
	render_target[coords].b = static_cast<uint8_t>(tileset_texture[coords].b * 255.f);
	render_target[coords].a = static_cast<uint8_t>(tileset_texture[coords].a * 255.f);
	}


namespace renderer
	{
	void renderer::draw(utils::cuda::kernel::texture<utils::graphics::colour::rgba_u> render_target)
		{
		utils::cuda::device::params_t threads
			{
			.threads
				{
				utils::cuda::device::params_t::threads_t::deduce
					(
					utils::cuda::dim3{16u, 16u},
					utils::cuda::dim3
						{
						std::min(render_target.width (), kernel_tileset_texture.width ()),
						std::min(render_target.height(), kernel_tileset_texture.height()),
						}
					)
				}
			};

		utils::cuda::device::call(&::draw, threads, render_target, kernel_tileset_texture);

		cudaDeviceSynchronize();
		}
	}