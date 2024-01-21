#pragma once

#include "game.h"

#include <thrust/device_vector.h>

namespace game
	{
	struct game_gpu::impl
		{
		thrust::device_vector<tile> grid;
		utils::matrix_observer<std::span<tile>> grid_kernel_side;
		};
	}