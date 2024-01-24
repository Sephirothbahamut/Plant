#include "renderer.h"


#include "cuda.cuh"

__global__ void draw(utils::cuda::render_target render_target, utils::cuda::kernel::texture<utils::graphics::colour::rgba_f> tileset_texture)
	{
	const auto coords{utils::cuda::kernel::coordinates::total::vec2()};
	if (!render_target.validate_coords(coords)) { return; }

	uchar4 pixel_data;

	if(tileset_texture.validate_coords(coords))
		{
		pixel_data = 
			{
			static_cast<uint8_t>(tileset_texture[coords].r * 255.f),//static_cast<uint8_t>((coords.x / (render_target.sizes.x / 2.f)) * 255.f),
			static_cast<uint8_t>(tileset_texture[coords].g * 255.f),//static_cast<uint8_t>((coords.y / (render_target.sizes.y / 2.f)) * 255.f),
			static_cast<uint8_t>(tileset_texture[coords].b * 255.f),//static_cast<uint8_t>(255.f),
			static_cast<uint8_t>(tileset_texture[coords].a * 255.f) //static_cast<uint8_t>(255.f)
			};
		}
	else
		{
		pixel_data =
			{
			static_cast<uint8_t>((coords.x / (render_target.sizes.x / 2.f)) * 255.f),
			static_cast<uint8_t>((coords.y / (render_target.sizes.y / 2.f)) * 255.f),
			static_cast<uint8_t>(255.f),
			static_cast<uint8_t>(255.f)
			};
		}
	surf2Dwrite(pixel_data, render_target.surface, coords.x * sizeof(uchar4), coords.y);
	}


namespace renderer
	{
	void renderer::draw(utils::cuda::render_target render_target)
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
						std::min(render_target.sizes.x, kernel_tileset_texture.width ()),
						std::min(render_target.sizes.y, kernel_tileset_texture.height()),
						}
					)
				}
			};

		utils::cuda::device::call(&::draw, threads, render_target, kernel_tileset_texture);

		cudaDeviceSynchronize();
		}
	}