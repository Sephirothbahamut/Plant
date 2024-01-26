#include "game.h"

#include <thrust/device_vector.h>

#include "cuda.cuh"

__global__ void step(utils::matrix_wrapper<std::span<game::tile>> grid, float time)
	{
	const auto coords{utils::cuda::kernel::coordinates::total::vec2()};
	if (!grid.validate_coords(coords)) { return; }

	auto& tile   {grid[coords]};
	auto& terrain{tile.terrain};
	auto& plant  {tile.plant  };

	terrain.step(time);
	plant  .step(terrain, time);
	}

__global__ void build_on_tile(utils::matrix_wrapper<std::span<game::tile>> grid, utils::math::vec2s coords, float absorption)
	{
	grid[coords].plant.humidity      = .5f;
	grid[coords].plant.humidity_next = .5f;
	grid[coords].plant.absorption    = absorption;
	}

namespace game
	{
	game::game(const ::game::data_cpu& data_cpu) : data_cpu{data_cpu}, data_gpu{data_cpu} {}

	data_gpu::data_gpu(const data_cpu& data_cpu) :
		grid{data_cpu.grid.begin(), data_cpu.grid.end()},
		grid_kernel_side{[this, &data_cpu]()
			{
			std::span<tile> device_span{thrust::raw_pointer_cast(grid.data()), grid.size()};
			utils::matrix_wrapper<std::span<tile>> grid_kernel_side{data_cpu.grid.sizes(), device_span};
			return grid_kernel_side;
			}()}
		{}
	data_gpu::~data_gpu() = default;

	void game::step(float delta_time) noexcept
		{
		data_cpu.metadata.time += delta_time;
		data_gpu.time = data_cpu.metadata.time;

		utils::cuda::device::params_t kernel_params
			{
			.threads{utils::cuda::device::params_t::threads_t::deduce({16u, 16u}, {data_cpu.grid.sizes()})},
			.shared_memory_bytes{0}
			};

		utils::cuda::device::call(&::step, kernel_params, data_gpu.grid_kernel_side, data_cpu.metadata.time);
		}

	void game::build(float absorption) noexcept
		{
		utils::cuda::device::call(&::build_on_tile, {.threads{.per_block{1u}, .blocks{1u}}}, data_gpu.grid_kernel_side, data_cpu.metadata.mouse_tile, absorption);
		}

	void game::cpu_to_gpu() noexcept
		{
		data_gpu = ::game::data_gpu{data_cpu};
		}
	void game::gpu_to_cpu() noexcept
		{
		thrust::copy(data_gpu.grid.begin(), data_gpu.grid.end(), data_cpu.grid.begin());
		}
	}