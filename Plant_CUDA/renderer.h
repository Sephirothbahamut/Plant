#pragma once

#include "texture.h"
#include "game.h"

namespace renderer
	{
	class render_target
		{
		public:
			render_target(const utils::math::vec2s& sizes) : gl_texture{sizes} {}

			void resize(const utils::math::vec2s& sizes) noexcept
				{
				gl_texture = utils::cuda::gl_texture{sizes};
				}

			void draw(sf::RenderTarget& sf_rt) const noexcept
				{
				sf::Sprite sprite{gl_texture.get_texture()};
				sf_rt.draw(sprite);
				}

			utils::cuda::gl_texture gl_texture;
		};

	class renderer
		{
		public:
			utils::cuda::kernel::texture<utils::graphics::colour::rgba_f> kernel_tileset_texture;
			utils::observer_ptr<game::game> game_ptr;

			void draw(utils::cuda::render_target render_target, float time);
		private:
		};
	}