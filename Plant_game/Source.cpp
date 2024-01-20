
#include <cmath>
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
#include <utils/containers/matrix_dyn.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "IIGE/loop.h"
#include "IIGE/systems_manager.h"
#include "IIGE/window_loop_interop.h"

#include "IIGE/window.h"

#include "sfglobals.h"

#include <chrono>


std::chrono::time_point program_start{std::chrono::high_resolution_clock::now()};

float time_since_game_start()
	{
	auto now{std::chrono::high_resolution_clock::now()};
	std::chrono::duration<float> df{now - program_start};
	return df.count();
	}

namespace game
	{
	class terrain
		{
		public:
			struct create_info { utils::math::vec2s grid_size; };
			terrain(const create_info& create_info) noexcept : grid_data{create_info.grid_size}, grid_data_draw{create_info.grid_size}, data_humidity_backbuffer{create_info.grid_size} {}

			struct tile
				{
				float humidity{0.f};
				float sunlight{0.f};
				};

			utils::containers::matrix_dyn<tile> grid_data;
			utils::containers::matrix_dyn<tile> grid_data_draw;
			utils::containers::matrix_dyn<float> data_humidity_backbuffer;

			void step(float delta_time)
				{
				backbuffer_to_front();
				zero_backbuffer    ();
				update_backbuffer  ();

				//auto indices{utils::indices(grid_data)};
				//std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
				//	{
				//	utils::math::vec2s coords{grid_data.get_coords(index)};
				//	grid_data[coords].humidity = ((std::sin(time_since_game_start()) + 1.f) / 2.f);
				//	});
				}

			void draw(float delta_time, float interpolation)
				{

				auto indices{utils::indices(grid_data)};

				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this, interpolation](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					grid_data_draw[coords].humidity = std::lerp(grid_data[coords].humidity, data_humidity_backbuffer[coords], interpolation);
					});
				}

		private:
			void update_backbuffer_tile(const tile& tile_data, float& target_data) noexcept
				{
				target_data += tile_data.humidity / 2.f;
				}

			void update_backbuffer_pass(const utils::math::vec2s& delta) noexcept
				{
				auto indices{utils::indices(grid_data)};

				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this, &delta](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					utils::math::vec2s target_coords{coords + delta};
					if (grid_data.validate_coords(target_coords))
						{
						update_backbuffer_tile(grid_data[coords], data_humidity_backbuffer[target_coords]);
						}
					});
				}

			void zero_backbuffer() noexcept
				{
				std::for_each(std::execution::par_unseq, data_humidity_backbuffer.begin(), data_humidity_backbuffer.end(), [](float& tile_data)
					{
					tile_data = 0;
					});
				}
			void update_backbuffer() noexcept
				{
				auto indices{utils::indices(grid_data)};

				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					if (coords.x > 0) { update_backbuffer_tile(grid_data[coords], data_humidity_backbuffer[coords.x - 1, coords.y]); }
					});
				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					if (coords.y > 0) { update_backbuffer_tile(grid_data[coords], data_humidity_backbuffer[coords.x, coords.y - 1]); }
					});
				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					if (coords.x > 0 && coords.y > 0) { update_backbuffer_tile(grid_data[coords], data_humidity_backbuffer[coords.x - 1, coords.y - 1]); }
					});

				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					if (coords.x < (grid_data.width() - 1)) { update_backbuffer_tile(grid_data[coords], data_humidity_backbuffer[coords.x + 1, coords.y]); }
					});
				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					if (coords.y < (grid_data.height() - 1)) { update_backbuffer_tile(grid_data[coords], data_humidity_backbuffer[coords.x, coords.y + 1]); }
					});
				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					if (coords.x < (grid_data.width() - 1) && coords.y < (grid_data.height() - 1)) { update_backbuffer_tile(grid_data[coords], data_humidity_backbuffer[coords.x + 1, coords.y + 1]); }
					});

				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					if (coords.x > 0 && coords.y < (grid_data.height() - 1)) { update_backbuffer_tile(grid_data[coords], data_humidity_backbuffer[coords.x - 1, coords.y + 1]); }
					});
				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					if (coords.x < (grid_data.width() - 1) && coords.y > 0) { update_backbuffer_tile(grid_data[coords], data_humidity_backbuffer[coords.x + 1, coords.y - 1]); }
					});

				//update_backbuffer_pass({ 0,  0});
				//update_backbuffer_pass({-1,  0});
				//update_backbuffer_pass({-1, -1});
				//update_backbuffer_pass({ 0, -1});
				//update_backbuffer_pass({ 1, -1});
				//update_backbuffer_pass({ 1,  0});
				//update_backbuffer_pass({ 1,  1});
				//update_backbuffer_pass({ 0,  1});
				//update_backbuffer_pass({-1,  1});
				}

			void backbuffer_to_front() noexcept
				{
				auto indices{utils::indices(grid_data)};

				std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this](const size_t& index)
					{
					utils::math::vec2s coords{grid_data.get_coords(index)};
					auto& tile_data{grid_data[coords]};
					const auto& data_humidity_backbuffer_tile{data_humidity_backbuffer[coords]};
					tile_data.humidity = data_humidity_backbuffer_tile;
					});
				}
		};
	}

int main()
	{
	sf_globals::font.loadFromFile("cour.ttf");

	iige::window window{iige::window::create_info{.title{"Plant"}, .size{800, 600}}};

	game::terrain terrain{game::terrain::create_info{.grid_size{200, 200}}};


	iige::systems_manager systems_manager;
	systems_manager.step.emplace([&terrain](float delta_time)
		{
		terrain.step(delta_time);
		});
	systems_manager.draw.emplace([&terrain](float delta_time, float interpolation)
		{
		terrain.draw(delta_time, interpolation);
		});

	iige::window_loop_interop                       window_loop_interop{window, systems_manager};
	window_loop_interop.events_handler = [](const sf::Event&) { return true; };

	window.display();
	iige::loop::fixed_game_speed_variable_framerate loop{window_loop_interop, 10.f};

	loop.run();
	}