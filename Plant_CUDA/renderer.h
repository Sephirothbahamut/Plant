#pragma once

#include <memory>
#include <filesystem>

#include <utils/math/ranged.h>
#include <utils/matrix_interface.h>
#include <utils/containers/matrix_dyn.h>


namespace renderer
	{
	struct renderer_gpu
		{
		struct impl;
	
		std::unique_ptr<impl> impl_ptr;
	
		//these will be defined in the .cu file
		game_gpu() noexcept;
		~game_gpu();
		void step(float delta_time) noexcept;
		};

	class renderer
		{
		public:
			struct data;
			std::unique_ptr<data> data_ptr;

			void load_map (const std::filesystem::path& path);
			void load_save(const std::filesystem::path& path);

			void step(float delta_time);
		
		private:
			void cpu_to_gpu() noexcept;
			void gpu_to_cpu() noexcept;
		};
	}