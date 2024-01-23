#pragma once

#include <memory>
#include <filesystem>

#include <utils/math/ranged.h>
#include <utils/matrix_interface.h>
#include <utils/containers/matrix_dyn.h>

#include <SFML/Graphics.hpp>
#include "glew.h"
#include <SFML/OpenGL.hpp>

#include <GL/GL.h>
#include "cuda_gl_interop.h"
/*
struct cuda_gl_interop_texture_stuff
	{
	utils::math::vec2s texture_size;

	unsigned int size_tex_data;
	unsigned int num_texels;
	unsigned int num_values;

	GLuint tex_cudaResult;

	GLuint pbo_dest;
	cudaGraphicsResource* cuda_pbo_dest_resource;

	cuda_gl_interop_texture_stuff(utils::math::vec2s texture_size) : texture_size{texture_size} { initGLBuffers(); }
	~cuda_gl_interop_texture_stuff() { deletePBO(&pbo_dest); }

	void initGLBuffers()
		{
		// create pbo
		createPBO(&pbo_dest, &cuda_pbo_dest_resource);
		}

	void createPBO(GLuint* pbo, struct cudaGraphicsResource** pbo_resource) {
	  // set up vertex data parameter
		num_texels = texture_size.x * texture_size.y;
		num_values = num_texels * 4;
		size_tex_data = sizeof(GLubyte) * num_values;
		void* data = malloc(size_tex_data);

		// create buffer object
		glGenBuffers(1, pbo);
		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
		free(data);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// register this buffer object with CUDA
		cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone);
		}

	void deletePBO(GLuint* pbo) {
		glDeleteBuffers(1, pbo);
		*pbo = 0;
		}
	};


struct cuda_gl_texture
	{
	sf::Texture texture;
	cuda_gl_interop_texture_stuff cuda_gl_interop_texture_stuff;
	cuda_gl_texture(const utils::math::vec2s& sizes) : texture{sizes}
		{
		
		}
	};*/

namespace renderer
	{
	class renderer
		{
		public:
			void resize(const utils::math::vec2s& sizes) noexcept
				{
				texture.create(sizes.x, sizes.y);
				}

		private:
			sf::Texture texture;
		};
	}