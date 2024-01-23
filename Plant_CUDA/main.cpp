#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>
#include <iostream>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <execution>
#include <unordered_map>

#include <utils/memory.h>
#include <utils/math/math.h>
#include <utils/index_range.h>
#include <utils/console/initializer.h>
#include <utils/containers/matrix_dyn.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "IIGE/loop.h"
#include "IIGE/window.h"
#include "IIGE/systems_manager.h"
#include "IIGE/window_loop_interop.h"

#include "sfglobals.h"

#include "game.h"
#include "texture.h"
#include "renderer.h"


class cuda_gl_texture
	{
	public:
		class cuda_resource_mapper
			{
			friend class cuda_gl_texture;
			public:
				~cuda_resource_mapper() { cudaGraphicsUnmapResources(1, &cuda_gl_texture.cuda_pbo_dest_resource, 0); }

				utils::matrix_wrapper<std::span<utils::graphics::colour::rgba_u>> get_kernel_side()
					{
					auto ptr{get_mapped_pointer()};
					utils::math::vec2s sizes{cuda_gl_texture.texture.getSize().x, cuda_gl_texture.texture.getSize().y};
					std::span<utils::graphics::colour::rgba_u> span{ptr, sizes.x * sizes.y};
					return utils::matrix_wrapper<std::span<utils::graphics::colour::rgba_u>>{sizes, span};
					}

			private:
				cuda_resource_mapper(cuda_gl_texture& cuda_gl_texture) : cuda_gl_texture{cuda_gl_texture} { cudaGraphicsMapResources(1, &cuda_gl_texture.cuda_pbo_dest_resource, 0); }
				cuda_gl_texture& cuda_gl_texture;

				utils::graphics::colour::rgba_u* get_mapped_pointer()
					{
					utils::graphics::colour::rgba_u* ptr;
					size_t num_bytes;
					cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, cuda_gl_texture.cuda_pbo_dest_resource);
					return ptr;
					}
			};

		cuda_gl_texture(const utils::math::vec2s& sizes)
			{
			texture.create(sizes.x, sizes.y);
			init_pbo();
			}

		cuda_resource_mapper map_to_cuda() noexcept { return cuda_resource_mapper{*this}; }

		sf::Texture texture;

		void init_pbo()
			{
			//unsigned int num_texels{texture.getSize().x * texture.getSize().y};
			//unsigned int num_values{num_texels * 4};
			//unsigned int size_tex_data{sizeof(GLubyte) * num_values};
			//void* data{malloc(size_tex_data)};
			//
			//// create buffer object
			//glGenBuffers(1, &pbo_dest);
			//glBindBuffer(GL_ARRAY_BUFFER, pbo_dest);
			//glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
			//free(data);
			//
			//glBindBuffer(GL_ARRAY_BUFFER, 0);

			// register this buffer object with CUDA
			pbo_dest = texture.getNativeHandle();
			cudaGraphicsGLRegisterBuffer(&cuda_pbo_dest_resource, pbo_dest, cudaGraphicsMapFlagsNone);
			}

	private:
		cudaGraphicsResource* cuda_pbo_dest_resource;
		GLuint pbo_dest;
	};

int main()
	{
	utils::console::initializer initializer_console;

	sf_globals::font.loadFromFile("cour.ttf");

	const float steps_per_second{10.f};
	const float seconds_per_step{1.f / steps_per_second};


	iige::window window{iige::window::create_info{.title{"Plant"}, .size{800, 600}}};

	utils::graphics::image image{"./data/textures/sample_tileset.png"};
	utils::CUDA::texture texture{image};

	cuda_gl_texture cuda_gl_texture{{32, 32}};


	game::game game{game::game::load_map("./data/maps/sample_map.json")};

	iige::systems_manager systems_manager;

	systems_manager.step.emplace([&game](float delta_time)
		{
		game.step(delta_time);
		});
	systems_manager.draw.emplace([&window, &cuda_gl_texture](float delta_time, float interpolation)
		{


		sf::Sprite sprite{cuda_gl_texture.texture};
		window.draw(sprite);
		window.display();
		});

	iige::window_loop_interop window_loop_interop{window, systems_manager};
	window_loop_interop.events_handler = [&](const sf::Event& event) 
		{
		switch (event.type)
			{
			case sf::Event::Closed: window.close(); break;
			case sf::Event::Resized: 
				{
				sf::FloatRect visibleArea(0, 0, static_cast<float>(event.size.width), static_cast<float>(event.size.height));
				window.setView(sf::View(visibleArea));

				//renderer.on_resize({event.size.width, event.size.height});
				break;
				}
			}

		return true; 
		};

	window.display();
	//iige::loop::fixed_game_speed_variable_framerate loop{window_loop_interop, steps_per_second};
	iige::loop::variable_fps_and_game_speed loop{window_loop_interop};
	loop.run();

	return 0;
	}