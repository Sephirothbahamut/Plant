#pragma once

#include <memory>
#include <algorithm>
#include <execution>
#include <filesystem>

#include <utils/index_range.h>
#include <utils/graphics/colour.h>
#include <utils/matrix_interface.h>
#include <utils/math/math.h>

#include <SFML/Graphics.hpp>

namespace utils::graphics
	{
	template <typename pixel_T = utils::graphics::colour::rgba_f>
	struct image : utils::matrix<pixel_T>
		{
		using pixel_t = pixel_T;
		using matrix_t = utils::matrix<pixel_t>;
		image(const utils::math::vec2s& size) noexcept : matrix_t{size} {}
		image(const std::filesystem::path& filename) noexcept : image{[&filename]()
			{
			sf::Image sf_image;
			if (!sf_image.loadFromFile(filename.string()))
				{
				throw std::runtime_error{"Failed to load image from path: \"" + filename.string() + "\"."};
				}
			return sf_image;
			}()}
			{}

		image(const sf::Image& sf_image) noexcept : matrix_t{{sf_image.getSize().x, sf_image.getSize().y}}
			{
			const auto indices{utils::indices(*this)};
			std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [this, &sf_image](const size_t& index)
				{
				const auto  coords{matrix_t::get_coords(index)};
				const auto& sf_pixel{sf_image.getPixel(coords.x, coords.y)};
				matrix_t::operator[](index) = pixel_t
					{
					utils::math::type_based_numeric_range<uint8_t>::template cast_to<utils::math::type_based_numeric_range<typename pixel_t::value_type>>(sf_pixel.r),
					utils::math::type_based_numeric_range<uint8_t>::template cast_to<utils::math::type_based_numeric_range<typename pixel_t::value_type>>(sf_pixel.g),
					utils::math::type_based_numeric_range<uint8_t>::template cast_to<utils::math::type_based_numeric_range<typename pixel_t::value_type>>(sf_pixel.b),
					utils::math::type_based_numeric_range<uint8_t>::template cast_to<utils::math::type_based_numeric_range<typename pixel_t::value_type>>(sf_pixel.a)
					};
				});
			}
		};
	}