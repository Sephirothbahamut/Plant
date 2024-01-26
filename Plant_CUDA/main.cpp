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


void entry()
	{
	utils::console::initializer initializer_console;

	sf_globals::font.loadFromFile("cour.ttf");

	const float steps_per_second{10.f};
	const float seconds_per_step{1.f / steps_per_second};

	iige::window window{iige::window::create_info{.title{"Plant"}, .size{800, 600}}};
	window.setActive();
	glewInit(); //glewInit MUST be called after initializing a context, wether real or unused. Otherwise opengl functions won't be available


	utils::graphics::image tilset_image{"./data/textures/plant_tileset.png"};
	utils::cuda::texture tilset_texture{tilset_image};

	renderer::render_target cuda_render_target{{window.getSize().x, window.getSize().y}};
	renderer::renderer renderer{tilset_texture.get_kernel_side()};


	game::game game{game::game::load_map("./data/maps/sample_map.json")};
	renderer.game_ptr = &game;

	iige::systems_manager systems_manager;

	sf::Text ui_overlay_text{"", sf_globals::font, 12u};
	ui_overlay_text.setPosition(4, 4);
	ui_overlay_text.setColor({200, 200, 200, 255});

	sf::RectangleShape ui_overlay_rect{{180, 60}};
	ui_overlay_rect.setPosition(0, 0);
	ui_overlay_rect.setOutlineThickness(1);
	ui_overlay_rect.setOutlineColor(sf::Color::Black);
	ui_overlay_rect.setFillColor(sf::Color{20, 20, 20, 200});



	bool pressed_ll{false};
	bool pressed_rr{false};
	bool pressed_up{false};
	bool pressed_dw{false};
	float camera_speed_multiplier{1.f};
	float build_absorption{.5f};
	constexpr float build_absorption_scroll_delta{.1f};

	auto update_camera{[&](float delta_time)
		{
		utils::math::vec2f camera_delta
			{
			static_cast<float>(pressed_rr) - static_cast<float>(pressed_ll),
			static_cast<float>(pressed_dw) - static_cast<float>(pressed_up)
			};
		camera_delta *= camera_speed_multiplier * delta_time;

		utils::math::vec2f previous_camera_tranform{game.data_cpu.camera_transform};
		utils::math::vec2f new_camera{previous_camera_tranform + camera_delta};
		if (new_camera.x < 0.f) { new_camera.x = (game.data_cpu.grid.width () * 64.f) + new_camera.x; }
		if (new_camera.y < 0.f) { new_camera.y = (game.data_cpu.grid.height() * 64.f) + new_camera.y; }
		if (new_camera.x >= game.data_cpu.grid.width ()) { new_camera.x = new_camera.x - (game.data_cpu.grid.width () * 64.f); }
		if (new_camera.y >= game.data_cpu.grid.height()) { new_camera.y = new_camera.y - (game.data_cpu.grid.height() * 64.f); }

		game.data_cpu.camera_transform = new_camera;
		}};

	systems_manager.step.emplace([&](float delta_time)
		{
		game.step(delta_time);
		});
	systems_manager.draw.emplace([&](float delta_time, float interpolation)
		{
		update_camera(delta_time);
		window.clear();
		window.resetGLStates();
		try
			{
			if (true)
				{
				//CUDA rendering
				auto mapper{cuda_render_target.gl_texture.map_to_cuda()};
				renderer.draw(mapper.get_cuda_render_target(), interpolation);
				}

			cuda_render_target.draw(window);
			}
		catch (const std::exception& e)
			{
			std::cout << e.what() << std::endl;
			}

		ui_overlay_text.setString
			(
			"Time: "                  + std::to_string(static_cast<size_t>(game.data_cpu.time        )) + 
			"\nBuild points: "        + std::to_string(static_cast<size_t>(game.data_cpu.build_points)) + 
			"\nBuilding absorption: " + std::to_string(build_absorption          )
			);
		window.draw(ui_overlay_rect);
		window.draw(ui_overlay_text);
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
			case sf::Event::MouseMoved:
				{
				game.update_mouse_position({event.mouseMove.x, event.mouseMove.y});
				}
			case sf::Event::MouseWheelScrolled:
				{
				build_absorption += event.mouseWheelScroll.delta * build_absorption_scroll_delta;
				build_absorption = utils::math::clamp(build_absorption, 0.f, 1.f);
				break;
				}
			case sf::Event::MouseButtonPressed:
				{
				if (event.mouseButton.button == sf::Mouse::Button::Left)
					{
					game.attempt_build(build_absorption);
					}
				break;
				}
			case sf::Event::KeyPressed:
				{
				switch (event.key.code)
					{
					case sf::Keyboard::Left : pressed_ll = true; break;
					case sf::Keyboard::Right: pressed_rr = true; break;
					case sf::Keyboard::Up   : pressed_up = true; break;
					case sf::Keyboard::Down : pressed_dw = true; break;
					}
				break;
				}
			case sf::Event::KeyReleased:
				{
				switch (event.key.code)
					{
					case sf::Keyboard::Left : pressed_ll = false; break;
					case sf::Keyboard::Right: pressed_rr = false; break;
					case sf::Keyboard::Up   : pressed_up = false; break;
					case sf::Keyboard::Down : pressed_dw = false; break;
					}
				break;
				}
			}

		return true; 
		};

	window.display();
	iige::loop::fixed_game_speed_variable_framerate loop{window_loop_interop, steps_per_second};
	//iige::loop::variable_fps_and_game_speed loop{window_loop_interop};
	loop.run();
	}

int main()
	{
	auto function{&entry};

	/*
	function();
	/*/
	try { function(); }
	catch (const std::exception& e)
		{
		std::cout << e.what() << std::endl;
		}
	/**/
	return 0;
	}