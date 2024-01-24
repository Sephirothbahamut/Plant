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
#include "image.h"
#include "texture.h"
#include "renderer.h"


void true_main()
	{
	utils::console::initializer initializer_console;

	sf_globals::font.loadFromFile("cour.ttf");

	const float steps_per_second{10.f};
	const float seconds_per_step{1.f / steps_per_second};

	iige::window window{iige::window::create_info{.title{"Plant"}, .size{800, 600}}};
	window.setActive();
	glewInit(); //glewInit MUST be called after initializing a context, wether real or unused. Otherwise opengl functions won't be available


	utils::graphics::image tilset_image{"./data/textures/sample_tileset.png"};
	utils::cuda::texture tilset_texture{tilset_image};

	renderer::render_target cuda_render_target{{window.getSize().x, window.getSize().y}};
	renderer::renderer renderer{tilset_texture.get_kernel_side()};


	game::game game{game::game::load_map("./data/maps/sample_map.json")};

	iige::systems_manager systems_manager;

	systems_manager.step.emplace([&game](float delta_time)
		{
		game.step(delta_time);
		});
	systems_manager.draw.emplace([&](float delta_time, float interpolation)
		{
		try
			{

			if (true)
				{
				//CUDA rendering
				auto mapper{cuda_render_target.gl_texture.map_to_cuda()};
				renderer.draw(mapper.get_kernel_side());
				}

			cuda_render_target.draw(window);
			}
		catch (const std::exception& e)
			{
			std::cout << e.what() << std::endl;
			}
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

				cuda_render_target.resize({event.size.width, event.size.height});
				break;
				}
			}

		return true; 
		};

	window.display();
	//iige::loop::fixed_game_speed_variable_framerate loop{window_loop_interop, steps_per_second};
	iige::loop::variable_fps_and_game_speed loop{window_loop_interop};
	loop.run();
	}