#pragma once

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

sf::Vector2f view_center;

class window_draggable_view
	{
	public:
		window_draggable_view(sf::RenderWindow& sfwnd) : sfwnd_ptr{&sfwnd}
			{
			view_center = sfwnd.getDefaultView().getCenter();
			}

		bool evaluate_event(const sf::Event& event)
			{
			switch (event.type)
				{
				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Button::Right)
						{
						moving = true;
						old_pos = sf::Vector2f(event.mouseButton.x, event.mouseButton.y);
						return true;
						}
					break;
				case sf::Event::MouseMoved:
					if (moving) { move({event.mouseMove.x, event.mouseMove.y}); return true; }
					break;
				case sf::Event::MouseWheelScrolled:
					if (event.mouseWheelScroll.delta > 0) 
						{
						zoom -= .1f;
						}
					else { zoom += .1f; }
					update_view(); 
					return true;
					break;

				case sf::Event::MouseButtonReleased:
				case sf::Event::MouseLeft:
					moving = false;
					break;

				case sf::Event::Resized:
					update_view();
					break;
				}
			return false;
			}

	private:
		utils::observer_ptr<sf::RenderWindow> sfwnd_ptr;

		bool moving{false};
		float zoom{1};
		sf::Vector2f old_pos;
		sf::Vector2f view_center;

		void move(sf::Vector2i new_pos_src)
			{
			sf::Vector2f new_pos{new_pos_src};
			sf::Vector2f delta{old_pos - new_pos};
			view_center += delta * zoom;
			old_pos = new_pos;
			update_view();
			}

		void update_view()
			{
			auto view{sfwnd_ptr->getView()};
			view.setCenter(view_center);
			view_center = view.getCenter();
			view.setSize(sf::Vector2f{static_cast<float>(sfwnd_ptr->getSize().x) * zoom, static_cast<float>(sfwnd_ptr->getSize().y) * zoom});
			//view.setViewport({0.f, 0.f, static_cast<float>(sfwnd_ptr->getSize().x), static_cast<float>(sfwnd_ptr->getSize().y)});
			sfwnd_ptr->setView(view);
			}
	};
