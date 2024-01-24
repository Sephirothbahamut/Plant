#pragma once
#include "glew.h"
#include <GL/GL.h>
#include <SFML/OpenGL.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>

class OpenGLTest
    {
    public:
        uchar4* image;
        GLuint gltexture;
        GLuint pbo;
        cudaGraphicsResource_t cudaPBO;
        uchar4* d_textureBufferData;

        sf::Vector2u windowSize;

        OpenGLTest(sf::Vector2u windowSize)
            {
            this->windowSize = sf::Vector2u(windowSize);
            this->setupOpenGL();
            };

        ~OpenGLTest()
            {
            delete image;
            image == nullptr;
            cudaFree(d_textureBufferData);
            d_textureBufferData == nullptr;
            glDeleteTextures(1, &gltexture);
            }

        void drawFrame();
        void createFrame(float time);
    private:
        void setupOpenGL();
    };