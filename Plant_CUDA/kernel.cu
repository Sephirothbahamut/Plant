#include <cmath>
#include <cstddef>
#include <algorithm>

#include <utils/math/vec2.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <SFML/Graphics.hpp>
#include "glew.h"
#include <SFML/OpenGL.hpp>

#include <GL/GL.h>
#include "cuda_gl_interop.h"






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


__global__ void kernel(std::byte* g_odata, utils::math::vec2s texture_size)
	{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	if (x > texture_size.x || y > texture_size.y) { return; }

	size_t base_index{(y * texture_size.x + x) * 4};

	float r{x / static_cast<float>(texture_size.x)};
	float g{y / static_cast<float>(texture_size.y)};
	float b{1.f};
	float a{1.f};

	g_odata[base_index + 0] = static_cast<std::byte>(r * 255.f);
	g_odata[base_index + 1] = static_cast<std::byte>(g * 255.f);
	g_odata[base_index + 2] = static_cast<std::byte>(b * 255.f);
	g_odata[base_index + 3] = static_cast<std::byte>(a * 255.f);
	}



#include <iostream>
int main()
	{
	sf::Context context;
	context.setActive(true);
	glewInit(); //glewInit MUST be called after initializing a context, wether real or unused. Otherwise opengl functions won't be available

	utils::math::vec2s texture_size{805, 600};

	sf::Texture texture;
	texture.create(texture_size.x, texture_size.y);

	cuda_gl_interop_texture_stuff cgits{texture_size};

	// calculate grid size
	dim3 block(16, 16, 1);

	dim3 grid
		{
		static_cast<unsigned int>(std::ceil(static_cast<float>(texture_size.x) / static_cast<float>(block.x))),
		static_cast<unsigned int>(std::ceil(static_cast<float>(texture_size.y) / static_cast<float>(block.y))),
		static_cast<unsigned int>(1)
		};

	if (true)
		{
		std::byte* out_data;
		cudaGraphicsMapResources(1, &cgits.cuda_pbo_dest_resource, 0);
		size_t num_bytes;
		cudaGraphicsResourceGetMappedPointer((void**)&out_data, &num_bytes, cgits.cuda_pbo_dest_resource);

		kernel<<<grid, block>>>(out_data, texture_size);

		cudaDeviceSynchronize();

		cudaGraphicsUnmapResources(1, &cgits.cuda_pbo_dest_resource, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, cgits.pbo_dest);

		glBindTexture(GL_TEXTURE_2D, texture.getNativeHandle());
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture_size.x, texture_size.y, GL_RGBA,
			GL_UNSIGNED_BYTE, NULL);

		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}

	auto image{texture.copyToImage()};
	image.saveToFile("hello.png");

	return 0;
	}
