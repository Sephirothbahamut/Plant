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
				float humidity_falloff_intensity{ 0.2f};
				float sunlight_starting         { 0.0f};

				utils_gpu_available float_ranged get_humidity(float time) const noexcept { return falloff(humidity_starting, humidity_falloff_intensity, time); }

				utils_gpu_available inline void step(float time) noexcept
					{
					sunlight_starting = get_humidity(time);
					}
			private:

			};

		class plant
			{
			public:
				float humidity     {0.00000f};
				float humidity_next{humidity};
				float absorption   {0.50000f};
				utils_gpu_available inline float get_distribution() const noexcept { return 1.f - absorption; }

				utils_gpu_available inline void step(const terrain& terrain, float time) noexcept
					{
					humidity = humidity_next;

					const float absorbed_humidity{absorption * terrain.get_humidity(time)};
					humidity_next = std::min(1.f, humidity + absorbed_humidity);
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
		struct metadata_t
			{
			struct wind
				{
				vec2i_ranged direction;
				float change_time;
				} wind;

			float time{0.f};
			float build_points{0.f};

			utils::math::vec2s camera_transform{0, 0};
			utils::math::vec2s mouse_tile{0, 0};
			};

		metadata_t metadata;
		utils::matrix<tile> grid{{32, 32}};
		};
	struct data_gpu
		{
		data_gpu(const data_cpu& data_cpu);
		~data_gpu();

		float time;
		thrust::device_vector<tile> grid;
		utils::matrix_wrapper<std::span<tile>> grid_kernel_side;
		};

	struct data_kernel
		{
		utils::matrix_wrapper<std::span<tile>> grid;
		vec2i_ranged wind_direction;
		utils::math::vec2s camera_transform;
		utils::math::vec2s mouse_tile;
		float time;
		};

	class game
		{
		public:
			game(const data_cpu& data_cpu);
			//~game();

			data_cpu data_cpu;
			data_gpu data_gpu;

			data_kernel kernel_game_state() const noexcept
				{
				return
					{
					.grid            {data_gpu.grid_kernel_side},
					.wind_direction  {data_cpu.metadata.wind.direction},
					.camera_transform{data_cpu.metadata.camera_transform},
					.mouse_tile      {data_cpu.metadata.mouse_tile},
					.time            {data_gpu.time}
					};
				}

			static game load_map (const std::filesystem::path& path);
			static game load_save(const std::filesystem::path& path);

			void step(float delta_time) noexcept;
			void attempt_build(float absorption) noexcept
				{
				if (data_cpu.metadata.build_points >= 100.f) 
					{
					data_cpu.metadata.build_points -= 100.f; 
					build(absorption);
					}
				}
		
		private:
			void cpu_to_gpu() noexcept;
			void gpu_to_cpu() noexcept;
			void build(float absorption) noexcept;
		};
	}