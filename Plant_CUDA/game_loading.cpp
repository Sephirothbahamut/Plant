#include "game.h"

#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <execution>

#include <utils/index_range.h>
#include <utils/graphics/colour.h>

#include <SFML/Graphics.hpp>

#include "nlohmann_json.h"

namespace game
	{
	namespace loading
		{
		namespace map
			{
			tiles::terrain tile_from_rgba(const utils::graphics::colour::rgba_f& colour) noexcept
				{
				return
					{
					.wall_health               {1.f - colour.a},
					.humidity_starting         {      colour.b},
					.humidity_falloff_intensity{      colour.g},
					.sunlight_starting         {      colour.r}
					};
				}

			namespace grid
				{
				utils::containers::matrix_dyn<tile> from_image(const sf::Image& sf_image)
					{
					utils::containers::matrix_dyn<tile> grid_data{sf_image.getSize().x, sf_image.getSize().y};
					auto indices{utils::indices(grid_data)};

					std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&grid_data, &sf_image](const size_t& index)
						{
						utils::math::vec2s coords{grid_data.get_coords(index)};

						auto& tile_data{grid_data[index]};
						const auto& sf_rgba{sf_image.getPixel(static_cast<unsigned int>(coords.x), static_cast<unsigned int>(coords.y))};
						const utils::graphics::colour::rgba_f rgba
							{
							static_cast<float>(sf_rgba.r) / 255.f,
							static_cast<float>(sf_rgba.g) / 255.f,
							static_cast<float>(sf_rgba.b) / 255.f,
							static_cast<float>(sf_rgba.a) / 255.f,
							};
						tile_data.terrain = tile_from_rgba(rgba);
						});

					return grid_data;
					}

				utils::containers::matrix_dyn<tile> from_file(const std::filesystem::path& path)
					{
					sf::Image sf_image;
					if (!sf_image.loadFromFile(path.string())) { throw std::runtime_error{"Failed to load image from path: \"" + path.string() + "\"."}; }

					return from_image(sf_image);
					}
				}
			
			data_cpu from_file(const std::filesystem::path& path)
				{
				std::ifstream file;
				file.exceptions(std::ifstream::badbit | std::ifstream::failbit);
				file.open(path);

				const auto json{nlohmann::json::parse(file)};

				const auto wind_key{json["wind"]};

				const std::string grid_data_str{json["grid_data"]};
				const std::filesystem::path grid_data_path{grid_data_str};

				return
					{
					.metadata
						{
						.wind
							{
							.direction
								{//cast to uint16 to force nlohmann read a number rather than a char.
								static_cast<uint8_t>(uint16_t{wind_key["x"]}),
								static_cast<uint8_t>(uint16_t{wind_key["y"]})
								},
							.change_time{wind_key["direction_change_time"]}
							},
						.build_points{json["starting_points"]}
						},
					.grid{grid::from_file(grid_data_path)}
					};
				}
			}
		namespace save
			{
			struct saved_tile
				{
				tiles::plant plant;
				float_ranged wall_health;
				};

			saved_tile tile_from_rgba(const utils::graphics::colour::rgba_f& colour) noexcept
				{
				return
					{
					.plant
						{
						.humidity     {colour.b},
						.humidity_next{colour.b}
						},
					.wall_health {1.f - colour.a}
					};
				}

			namespace grid
				{
				void from_image(const sf::Image& sf_image, utils::containers::matrix_dyn<tile>& grid_data)
					{
					if (sf_image.getSize().x != grid_data.width() || sf_image.getSize().y != grid_data.height())
						{
						throw std::runtime_error{"Saved plant does not match the associated level."};
						}

					auto indices{utils::indices(grid_data)};

					std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&grid_data, &sf_image](const size_t& index)
						{
						utils::math::vec2s coords{grid_data.get_coords(index)};

						auto& tile_data{grid_data[index]};
						const auto& sf_rgba{sf_image.getPixel(static_cast<unsigned int>(coords.x), static_cast<unsigned int>(coords.y))};
						const utils::graphics::colour::rgba_f rgba
							{
							static_cast<float>(sf_rgba.r) / 255.f,
							static_cast<float>(sf_rgba.g) / 255.f,
							static_cast<float>(sf_rgba.b) / 255.f,
							static_cast<float>(sf_rgba.a) / 255.f,
							};
						auto loaded_tile_data{tile_from_rgba(rgba)};
						tile_data.plant = loaded_tile_data.plant;
						tile_data.terrain.wall_health = loaded_tile_data.wall_health;
						});
					}

				utils::containers::matrix_dyn<tile> from_file(const std::filesystem::path& path, utils::containers::matrix_dyn<tile>& grid_data)
					{
					sf::Image sf_image;
					if (!sf_image.loadFromFile(path.string())) { throw std::runtime_error{"Failed to load image from path: \"" + path.string() + "\"."}; }

					from_image(sf_image, grid_data);
					}
				}
			
			
			data_cpu from_file(const std::filesystem::path& path)
				{
				std::ifstream file;
				file.exceptions(std::ifstream::badbit | std::ifstream::failbit);
				file.open(path);

				const auto json{nlohmann::json::parse(file)};

				const auto wind_key{json["wind"]};

				const std::string map_str{json["map"]};
				const std::filesystem::path map_path{map_str};

				const std::string grid_data_str{json["grid_data"]};
				const std::filesystem::path grid_data_path{grid_data_str};

				data_cpu data_cpu{map::from_file(map_str)};

				data_cpu.metadata.wind.direction =
					{//cast to uint16 to force nlohmann read a number rather than a char.
					static_cast<uint8_t>(uint16_t{wind_key["x"]}),
					static_cast<uint8_t>(uint16_t{wind_key["y"]})
					};
				data_cpu.metadata.time         = json["time"];
				data_cpu.metadata.build_points = json["build_points"];

				grid::from_file(grid_data_path, data_cpu.grid);

				return data_cpu;;
				}
			}
		}
	
	void game::load_map(const std::filesystem::path& path)
		{
		data_cpu = loading::map::from_file(path);
		}
	void game::load_save(const std::filesystem::path& path)
		{
		data_cpu = loading::map::from_file(path);
		}
	}