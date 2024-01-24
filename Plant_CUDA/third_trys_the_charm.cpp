#include "OpenGLTest.cuh"
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/System.hpp>
#include <SFML/Graphics/RenderWindow.hpp>


void main_third()
    {
        // create the window
    sf::RenderWindow window(sf::VideoMode(1024, 1024), "OpenGL");
    //window.setVerticalSyncEnabled(true);
    sf::Vector2u windowSize;

    windowSize = sf::Vector2u(window.getSize());

    bool running = true;
    glewInit();
    window.resetGLStates();
    std::printf("OpenGL: %s:", glGetString(GL_VERSION));
    // We will not be using SFML's gl states.

    OpenGLTest* test = new OpenGLTest(window.getSize());

    sf::Time time;

    while (running)
        {
            // handle events
        sf::Event event;
        while (window.pollEvent(event))
            {
            if (event.type == sf::Event::Closed)
                {
                    // end the program
                running = false;
                }
            else if (event.type == sf::Event::Resized)
                {
                    // adjust the viewport when the window is resized
                glViewport(0, 0, event.size.width, event.size.height);
                windowSize = window.getSize();
                }

            }

            // clear the buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        test->createFrame(time.asMilliseconds());
        test->drawFrame();
        window.display();
        }

        // release resources...
    delete test;

    }