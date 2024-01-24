#pragma once

#include <memory>
#include <filesystem>

#include <utils/math/ranged.h>
#include <utils/graphics/colour.h>
#include <utils/matrix_interface.h>
#include <utils/compilation/debug.h>
#include <utils/containers/matrix_dyn.h>

#include <thrust/device_vector.h>
#include "image.h"
#include "utils_cuda.h"


#include <SFML/Graphics.hpp>
#include "glew.h"
#include <SFML/OpenGL.hpp>

#include <GL/GL.h>
#include "cuda_gl_interop.h"

namespace utils::cuda
	{
	namespace kernel
		{
		template <typename pixel_t = utils::graphics::colour::rgba_f>
		using texture = utils::matrix_wrapper<std::span<pixel_t>>;
		}

	template <typename pixel_t = utils::graphics::colour::rgba_f>
	class texture
		{
		public:
			texture(const utils::math::vec2s& size);
			texture(const utils::graphics::image<pixel_t>& image);
			~texture();

			kernel::texture<pixel_t> get_kernel_side() noexcept;

		private:
			utils::math::vec2s sizes;
			thrust::device_vector<pixel_t> data;
		};


	class gl_texture
		{
		public:
			class cuda_resource_mapper
				{
				friend class gl_texture;
				public:
					~cuda_resource_mapper() 
						{
						//if constexpr (utils::compilation::debug) { cuda_gl_texture.debug_is_mapped_to_cuda = false; }
						cudaGraphicsUnmapResources(1, &cuda_gl_texture.cuda_resource_handle, 0); 
						cuda_gl_texture.apply_changes_to_texture();
						}

					kernel::texture<utils::graphics::colour::rgba_u> get_kernel_side()
						{
						auto ptr{get_mapped_pointer()};
						std::span<utils::graphics::colour::rgba_u> span{ptr, cuda_gl_texture.sizes().x * cuda_gl_texture.sizes().y};
						return utils::matrix_wrapper<std::span<utils::graphics::colour::rgba_u>>{cuda_gl_texture.sizes(), span};
						}

				private:
					cuda_resource_mapper(gl_texture& cuda_gl_texture) : cuda_gl_texture{cuda_gl_texture} 
						{
						/*if constexpr (utils::compilation::debug)
							{
							if (cuda_gl_texture.debug_is_mapped_to_cuda) { throw std::runtime_error{"Attempting to map to cuda a cuda::gl_texture that was already mapped."}; }
							cuda_gl_texture.debug_is_mapped_to_cuda = true;
							}*/
						utils::cuda::check_throwing(cudaGraphicsMapResources(1, &cuda_gl_texture.cuda_resource_handle, 0));
						}
					gl_texture& cuda_gl_texture;

					utils::graphics::colour::rgba_u* get_mapped_pointer()
						{
						utils::graphics::colour::rgba_u* ptr;
						size_t num_bytes;
						utils::cuda::check_throwing(cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, cuda_gl_texture.cuda_resource_handle));
						
						const size_t expected_bytes{evaluate_required_bytes(cuda_gl_texture.sizes())};
						if (expected_bytes != num_bytes)
							{
							throw std::runtime_error{"SHOULD NOT HAPPEN!"};
							}
						
						return ptr;
						}
				};

			gl_texture(const utils::math::vec2s& sizes) :
				opengl_pixel_buffer_object_handle{create_opengl_pixel_buffer_object(sizes)},
				cuda_resource_handle{create_cuda_gl_registered_buffer(opengl_pixel_buffer_object_handle)}
				{
				texture.create(sizes.x, sizes.y);
				}
			~gl_texture()
				{
				utils::cuda::check(cudaGraphicsUnregisterResource(cuda_resource_handle));
				}
				
			cuda_resource_mapper map_to_cuda() 
				{
				return cuda_resource_mapper{*this}; 
				}

			utils::math::vec2s sizes() const noexcept { return {texture.getSize().x, texture.getSize().y}; }
			const sf::Texture get_texture() const noexcept { return texture; }

		private:
			unsigned int opengl_pixel_buffer_object_handle{0};
			cudaGraphicsResource* cuda_resource_handle{nullptr};
			sf::Texture texture;

			bool debug_is_mapped_to_cuda{false};

			inline static size_t evaluate_required_bytes(const utils::math::vec2s sizes) noexcept
				{
				const size_t num_texels{sizes.x * sizes.y};
				const size_t num_values{num_texels * 4};
				return sizeof(GLubyte) * num_values;
				}

			inline static unsigned int create_opengl_pixel_buffer_object(const utils::math::vec2s sizes) noexcept
				{
				unsigned int ret;
				const size_t required_bytes{evaluate_required_bytes(sizes)};
				void* data{malloc(required_bytes)};
			
				glGenBuffers(1, &ret);
				glBindBuffer(GL_ARRAY_BUFFER, ret);
				glBufferData(GL_ARRAY_BUFFER, required_bytes, data, GL_DYNAMIC_DRAW);
				free(data);
			
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				return ret;
				}
			inline static cudaGraphicsResource* create_cuda_gl_registered_buffer(unsigned int opengl_pixel_buffer_object_handle) noexcept
				{
				cudaGraphicsResource* ret{nullptr};
				utils::cuda::check_throwing(cudaGraphicsGLRegisterBuffer(&ret, opengl_pixel_buffer_object_handle, cudaGraphicsMapFlagsNone));
				return ret;
				}
			

			unsigned int opengl_handle() const noexcept
				{
				/*if constexpr (utils::compilation::debug)
					{
					if (debug_is_mapped_to_cuda) { throw std::runtime_error{"Attempting to access the opengl handle of a cuda::gl_texture while it's mapped for cuda usage."}; }
					}*/
				return opengl_pixel_buffer_object_handle;
				}
			
			const sf::Texture& apply_changes_to_texture() noexcept
				{
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, opengl_handle());

				glBindTexture(GL_TEXTURE_2D, texture.getNativeHandle());
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture.getSize().x, texture.getSize().y, GL_RGBA,
					GL_UNSIGNED_BYTE, NULL);

				glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

				return texture;
				}
		};
	}