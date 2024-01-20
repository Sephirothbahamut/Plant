#pragma once

#include <utils/math/vec2.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

namespace iige
	{
	class window : public sf::RenderWindow
		{
		public:
			struct create_info
				{
				std::string title;
				utils::math::vec2u size;
				};

			window(const create_info& create_info) : sf::RenderWindow
				{
				sf::VideoMode{create_info.size.x, create_info.size.y},
				create_info.title
				}
				{}
		};
	}