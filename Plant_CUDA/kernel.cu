#include <cmath>
#include <cstddef>
#include <algorithm>

#include <utils/math/vec2.h>
#include <utils/matrix_interface.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <SFML/Graphics.hpp>
#include "glew.h"
#include <SFML/OpenGL.hpp>

#include <GL/GL.h>
#include "cuda_gl_interop.h"

#include "cuda.cuh"

#include "texture.h"

/*class cuda_gl_texture
	{
	public:
		class cuda_resource_mapper
			{
			friend class cuda_gl_texture;
			public:
				~cuda_resource_mapper() { cudaGraphicsUnmapResources(1, &cuda_gl_texture.cuda_pbo_dest_resource, 0); }

				utils::matrix_wrapper<std::span<utils::graphics::colour::rgba_u>> get_kernel_side()
					{
					auto ptr{get_mapped_pointer()};
					utils::math::vec2s sizes{cuda_gl_texture.texture.getSize().x, cuda_gl_texture.texture.getSize().y};
					std::span<utils::graphics::colour::rgba_u> span{ptr, sizes.x * sizes.y};
					return utils::matrix_wrapper<std::span<utils::graphics::colour::rgba_u>>{sizes, span};
					}

			private:
				cuda_resource_mapper(cuda_gl_texture& cuda_gl_texture) : cuda_gl_texture{cuda_gl_texture} { cudaGraphicsMapResources(1, &cuda_gl_texture.cuda_pbo_dest_resource, 0); }
				cuda_gl_texture& cuda_gl_texture;

				utils::graphics::colour::rgba_u* get_mapped_pointer()
					{
					utils::graphics::colour::rgba_u* ptr;
					size_t num_bytes;
					cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, cuda_gl_texture.cuda_pbo_dest_resource);
					return ptr;
					}
			};

		cuda_gl_texture(const utils::math::vec2s& sizes)
			{
			texture.create(sizes.x, sizes.y);
			init_pbo();
			}

		cuda_resource_mapper map_to_cuda() noexcept { return cuda_resource_mapper{*this}; }

		sf::Texture texture;

		void init_pbo()
			{
			//unsigned int num_texels{texture.getSize().x * texture.getSize().y};
			//unsigned int num_values{num_texels * 4};
			//unsigned int size_tex_data{sizeof(GLubyte) * num_values};
			//void* data{malloc(size_tex_data)};
			//
			//// create buffer object
			//glGenBuffers(1, &pbo_dest);
			//glBindBuffer(GL_ARRAY_BUFFER, pbo_dest);
			//glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
			//free(data);
			//
			//glBindBuffer(GL_ARRAY_BUFFER, 0);

			// register this buffer object with CUDA
			pbo_dest = texture.getNativeHandle();
			cudaGraphicsGLRegisterBuffer(&cuda_pbo_dest_resource, pbo_dest, cudaGraphicsMapFlagsNone);
			}

	private:
		cudaGraphicsResource* cuda_pbo_dest_resource;
		GLuint pbo_dest;
	};*/

struct cuda_gl_interop_texture_stuff
	{
	utils::math::vec2s texture_size;

	unsigned int size_tex_data;
	unsigned int num_texels;
	unsigned int num_values;

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

__global__ void kernel2(cudaSurfaceObject_t surface, utils::math::vec2s sizes)//utils::matrix_wrapper<std::span<utils::graphics::colour::rgba_u>> matrix)
	{
	const auto coords{utils::cuda::kernel::coordinates::total::vec3()};
	//printf("(%u, %u, %u)\n", coords.x, coords.y, coords.z);

	//if (!matrix.validate_coords(coords)) { return; }
	if (!(coords.x < sizes.x && coords.y < sizes.y)) { return; }


	/*matrix[coords] = utils::graphics::colour::rgba_u
		{
		static_cast<uint8_t>((coords.x / matrix.width ()) * 255.f),
		static_cast<uint8_t>((coords.y / matrix.height()) * 255.f),
		static_cast<uint8_t>(255.f),
		static_cast<uint8_t>(255.f)
		};*/

	uchar4 pixel_data = make_uchar4
		(
		static_cast<uint8_t>((coords.x / (sizes.x / 2.f)) * 255.f),
		static_cast<uint8_t>((coords.y / (sizes.y / 2.f)) * 255.f),
		static_cast<uint8_t>(255.f),
		static_cast<uint8_t>(255.f)
		);
	surf2Dwrite(pixel_data, surface, coords.x * sizeof(uchar4), coords.y);
	}

void false_main()
	{
	sf::Context context;
	//glewInit(); //glewInit MUST be called after initializing a context, wether real or unused. Otherwise opengl functions won't be available

	utils::math::vec2s texture_size{32, 32};

	sf::Texture texture;
	texture.create(texture_size.x, texture_size.y);
	//cuda_gl_interop_texture_stuff cgits{texture_size};

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
		utils::cuda::gl_texture cuda_gl_texture{texture_size};
		if (true)
			{
			auto mapper{cuda_gl_texture.map_to_cuda()};
			auto surface{mapper.get_surface_object()};
			kernel2<<<grid, block>>>(surface, texture_size);
			}


		auto image{cuda_gl_texture.get_texture().copyToImage()};
		image.saveToFile("hello.png");

		//display image beg
		//display image end
		}

	//if (true)
	//	{
	//
	//
	//	std::byte* out_data;
	//	utils::cuda::check_throwing(cudaGraphicsMapResources(1, &cgits.cuda_pbo_dest_resource, 0));
	//	size_t num_bytes;
	//	utils::cuda::check_throwing(cudaGraphicsResourceGetMappedPointer((void**)&out_data, &num_bytes, cgits.cuda_pbo_dest_resource));
	//
	//	kernel<<<grid, block>>>(out_data, texture_size);
	//
	//	cudaDeviceSynchronize();
	//
	//	utils::cuda::check_throwing(cudaGraphicsUnmapResources(1, &cgits.cuda_pbo_dest_resource, 0));
	//	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, cgits.pbo_dest);
	//
	//	glBindTexture(GL_TEXTURE_2D, texture.getNativeHandle());
	//	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture_size.x, texture_size.y, GL_RGBA,
	//		GL_UNSIGNED_BYTE, NULL);
	//
	//	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	//	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	//	}
	//
	//if (true)
	//	{
	//	utils::cuda::gl_texture cgt{{32, 32}};
	//	auto mapper{cgt.map_to_cuda()};
	//	auto kernel_side{mapper.get_kernel_side()};
	//
	//	kernel2<<<grid, block>>>(kernel_side);
	//	}

	//auto image{texture.copyToImage()};
	//image.saveToFile("hello.png");
	}