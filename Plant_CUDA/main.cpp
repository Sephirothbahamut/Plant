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

#include "game.h"
#include "sfglobals.h"

int main()
	{
	utils::console::initializer initializer_console;

	sf_globals::font.loadFromFile("cour.ttf");

	const float steps_per_second{10.f};
	const float seconds_per_step{1.f / steps_per_second};


	iige::window window{iige::window::create_info{.title{"Plant"}, .size{800, 600}}};
	game::game game{game::game::load_map("./data/maps/sample_map_large.json")};

	iige::systems_manager systems_manager;

	systems_manager.step.emplace([&game](float delta_time)
		{
		game.step(delta_time);
		});
	systems_manager.draw.emplace([&window](float delta_time, float interpolation)
		{
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