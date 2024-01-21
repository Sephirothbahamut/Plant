#include "game.h"
#include "game_gpu_impl.cuh"

#include <thrust/device_vector.h>


namespace game
	{
	game_gpu::game_gpu() noexcept {}
	game_gpu::~game_gpu() = default;
	void game_gpu::step(float delta_time) noexcept
		{}

	
	game_gpu::impl generate_data_gpu(const utils::containers::matrix_dyn<tile>& cpu_grid)
		{
		thrust::device_vector<tile> device_vector{cpu_grid.get_vector()};
		std::span<tile> device_span{thrust::raw_pointer_cast(device_vector.data()), device_vector.size()};
		utils::matrix_observer<std::span<tile>> grid_kernel_side{cpu_grid.sizes(), device_span};
		return
			{
			.grid            {cpu_grid.get_vector()},
			.grid_kernel_side{grid_kernel_side}
			};
		}

	void game::cpu_to_gpu() noexcept
		{
		//data_gpu = generate_data_gpu(data_cpu.grid);
		}
	void game::gpu_to_cpu() noexcept
		{
		//thrust::copy(data_gpu.grid.begin(), data_gpu.grid.end(), data_cpu.grid.begin());
		}
	}