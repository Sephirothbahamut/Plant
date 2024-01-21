#pragma once

#include <memory>
#include <filesystem>

#include <utils/math/ranged.h>
#include <utils/matrix_interface.h>
#include <utils/containers/matrix_dyn.h>


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
			private:

			};

		class plant
			{
			public:
				float humidity     {0.00001f};
				float humidity_next{humidity};

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
		utils::containers::matrix_dyn<tile> grid{32, 32};
		};

	struct game_gpu
		{
		struct impl;
	
		std::unique_ptr<impl> impl_ptr;
	
		//these will be defined in the .cu file
		game_gpu() noexcept;
		~game_gpu();
		void step(float delta_time) noexcept;
		};

	class game
		{
		public:

			data_cpu data_cpu;
			game_gpu game_gpu;

			void load_map (const std::filesystem::path& path);
			void load_save(const std::filesystem::path& path);

			void step(float delta_time);
		
		private:
			void cpu_to_gpu() noexcept;
			void gpu_to_cpu() noexcept;
		};
	}