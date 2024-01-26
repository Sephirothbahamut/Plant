#pragma once

#include <memory>
#include <algorithm>
#include <filesystem>

#include <utils/math/ranged.h>
#include <utils/matrix_interface.h>

#include <thrust/device_vector.h>

namespace game
	{
	using float_ranged = utils::math::ranged<float  ,  0.f, 1.f>;
	using int8_ranged  = utils::math::ranged<int8_t, -1  , 1  >;
	using vec2i_ranged = utils::math::vec2  <int8_ranged>;


	inline utils_gpu_available float_ranged falloff(float_ranged starting_value, float_ranged falloff_intensity, float time)
		{
		// Desmos expression: y=\left(1-\frac{t_{ime}}{t_{ime}+\frac{1}{f_{alloffintensity}}}\right)\cdot s_{tartingvalue}
		return (1.f - (time / (time + (1.f / falloff_intensity)))) * starting_value;
		}


	namespace tiles
		{
		class terrain
			{
			public:
				float wall_health               { 0.0f};
				float humidity_starting         { 0.0f};
				float humidity_falloff_intensity{ 0.1f};
				float sunlight_starting         { 0.0f};

				utils_gpu_available float_ranged get_humidity(float time) const noexcept { return falloff(humidity_starting, humidity_falloff_intensity, time); }

				utils_gpu_available inline void step(float time) noexcept
					{
					//sunlight_starting = get_humidity(time);
					}
			private:

			};

		class plant
			{
			public:
				float humidity                            {0.00000f};
				float humidity_next                       {humidity};//for interpolation
				float humidity_distributed_per_each_target{0.00000f};//temporary buffer for humidity distribution
				float humidity_to_distribute              {0.00000f};//temporary buffer for humidity distribution
				float absorption                          {0.50000f};
				float absorbed_light                      {0.00000f};//temporary buffer for score

				utils_gpu_available inline float get_distribution() const noexcept { return 1.f - absorption; }

				utils_gpu_available inline void step(const terrain& terrain, float time) noexcept
					{
					humidity = humidity_next;
					if (humidity <= 0) { absorbed_light = 0.f; return; }

					absorbed_light = humidity * terrain.sunlight_starting;

					const float humidity_absorbed{absorption * terrain.get_humidity(time)};
					humidity_to_distribute = get_distribution() * humidity;
					const float humidity_left{humidity - humidity_to_distribute};

					humidity_distributed_per_each_target = humidity_to_distribute / 8.f;
					humidity_next = utils::math::clamp(humidity_left + humidity_absorbed, 0.f, 1.f);
					}

				utils_gpu_available inline void step_distribution(plant& target) noexcept
					{
					if (humidity <= 0.f || humidity_distributed_per_each_target <= 0.f || target.humidity <= 0.f || target.humidity >= 1.f) { return; }
					const float distributed_to_target{utils::math::min((1.f - target.humidity), humidity_distributed_per_each_target)};
					target.humidity_next   += distributed_to_target;
					humidity_to_distribute -= distributed_to_target;
					}

				utils_gpu_available inline void step_recover_undistributed() noexcept
					{
					if (humidity <= 0.f || humidity_to_distribute <= 0.f) { return; }

					humidity_next += humidity_to_distribute;
					}

			private:

			};
		}

	class tile
		{
		public:
			tiles::terrain terrain;
			tiles::plant   plant;

		private:

		};


	struct data_cpu
		{
		struct wind
			{
			vec2i_ranged direction;
			float change_time;
			} wind;

		float time{0.f};
		float next_time{0.f};
		float build_points{0.f};

		utils::math::vec2s camera_transform{0, 0};
		utils::math::vec2s mouse_tile{0, 0};

		utils::matrix<tile> grid{{32, 32}};
		//utils::matrix<uint8_t> occupied_mask{{32, 32}};
		};

	struct data_gpu
		{
		data_gpu(const data_cpu& data_cpu);
		~data_gpu();

		thrust::device_vector<tile> grid;
		utils::matrix_wrapper<std::span<tile>> grid_kernel_side;
		};

	struct kernel_data_for_draw
		{
		float time;
		float next_time{0.f};
		utils::math::vec2s camera_transform;
		utils::math::vec2s mouse_tile;
		utils::matrix_wrapper<std::span<tile>> grid;
		};
	
	namespace coords
		{
		struct evaluated_t
			{
			utils::math::vec2s in_world;
			utils::math::vec2s of_tile;
			utils::math::vec2s in_tile;
			utils::math::vec2f in_tile_normalized;
			};
		utils_gpu_available inline evaluated_t evaluate(const utils::math::vec2s& window_coords, const utils::math::vec2s& world_size, const utils::math::vec2f& camera_transform)
			{
			const utils::math::vec2s world_pixel       {window_coords + camera_transform };
			const utils::math::vec2s in_world          {world_pixel % (world_size * size_t{64})};
			const utils::math::vec2s of_tile           {in_world / 64};
			const utils::math::vec2u in_tile           {world_pixel % size_t{64}};
			const utils::math::vec2f in_tile_normalized{utils::math::vec2f{in_tile} / 64.f};

			return evaluated_t
				{
				.in_world{in_world},
				.of_tile {of_tile },
				.in_tile {in_tile },
				.in_tile_normalized{in_tile_normalized}
				};
			}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="source"></param>
		/// <param name="world_size"></param>
		/// <param name="delta">Assumed to be [-1, 0, 1] </param>
		/// <returns></returns>
		utils_gpu_available inline utils::math::vec2s tile_neighbour(const utils::math::vec2s& source, const utils::math::vec2s& world_size, const utils::math::vec2u& delta)
			{
			utils::math::vec2s moved{source};

			if (delta.x == -1 && source.x == 0)
				{
				moved.x = world_size.x - 1;
				}
			else
				{
				moved.x += delta.x;
				moved.x %= world_size.x;
				}
			if (delta.y == -1 && source.y == 0)
				{
				moved.y = world_size.y - 1;
				}
			else
				{
				moved.y += delta.y;
				moved.y %= world_size.y;
				}

			return moved;
			}
		}


	class game
		{
		public:
			game(const data_cpu& data_cpu);
			//~game();

			data_cpu data_cpu;
			data_gpu data_gpu;

			kernel_data_for_draw kernel_data_for_draw() const noexcept
				{
				return
					{
					.time            {data_cpu.time},
					.next_time       {data_cpu.next_time},
					.camera_transform{data_cpu.camera_transform},
					.mouse_tile      {data_cpu.mouse_tile},
					.grid            {data_gpu.grid_kernel_side}
					};
				}

			static game load_map (const std::filesystem::path& path);
			static game load_save(const std::filesystem::path& path);

			void step(float delta_time) noexcept;

			void update_mouse_position(const utils::math::vec2s mouse_position_on_window) noexcept
				{
				data_cpu.mouse_tile = coords::evaluate(mouse_position_on_window, data_cpu.grid.sizes(), data_cpu.camera_transform).of_tile;
				}

			void attempt_build(float absorption) noexcept
				{
				if (data_cpu.build_points < 100.f) { return; }
				//if (data_cpu.occupied_mask[coords::tile_neighbour(data_cpu.mouse_tile, data_cpu.grid.sizes(), {-1, -1})] ||
				//	data_cpu.occupied_mask[coords::tile_neighbour(data_cpu.mouse_tile, data_cpu.grid.sizes(), { 0, -1})] ||
				//	data_cpu.occupied_mask[coords::tile_neighbour(data_cpu.mouse_tile, data_cpu.grid.sizes(), { 1, -1})] ||
				//	data_cpu.occupied_mask[coords::tile_neighbour(data_cpu.mouse_tile, data_cpu.grid.sizes(), {-1,  0})] ||
				//	data_cpu.occupied_mask[coords::tile_neighbour(data_cpu.mouse_tile, data_cpu.grid.sizes(), { 1,  0})] ||
				//	data_cpu.occupied_mask[coords::tile_neighbour(data_cpu.mouse_tile, data_cpu.grid.sizes(), {-1,  1})] ||
				//	data_cpu.occupied_mask[coords::tile_neighbour(data_cpu.mouse_tile, data_cpu.grid.sizes(), { 0,  1})] ||
				//	data_cpu.occupied_mask[coords::tile_neighbour(data_cpu.mouse_tile, data_cpu.grid.sizes(), { 1,  1})]
				//	)
					{
					data_cpu.build_points -= 100.f;
					build(absorption);
					}
				}
		
		private:
			void cpu_to_gpu() noexcept;
			void gpu_to_cpu() noexcept;
			void build(float absorption) noexcept;
		};
	}