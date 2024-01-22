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
				float humidity     {0.00001f};
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

	struct metadata
		{
		struct wind
			{
			vec2i_ranged direction;
			float change_time;
			} wind;

		float time        {0.f};
		float build_points{0.f};
		};


	struct data_cpu
		{
		metadata metadata;
		utils::matrix<tile> grid{{32, 32}};
		};
	struct data_gpu
		{
		data_gpu(const data_cpu& data_cpu);
		~data_gpu();

		thrust::device_vector<tile> grid;
		utils::matrix_wrapper<std::span<tile>> grid_kernel_side;
		};

	class game
		{
		public:
			game(const data_cpu& data_cpu);
			//~game();

			data_cpu data_cpu;
			data_gpu data_gpu;

			static game load_map (const std::filesystem::path& path);
			static game load_save(const std::filesystem::path& path);

			void step(float delta_time) noexcept;
		
		private:
			void cpu_to_gpu() noexcept;
			void gpu_to_cpu() noexcept;
		};
	}