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
						utils::cuda::check_throwing(cudaDestroySurfaceObject(surfObject));
						utils::cuda::check_throwing(cudaGraphicsUnmapResources(1, &cuda_gl_texture.cuda_resource_handle, 0));
						}

					/*kernel::texture<utils::graphics::colour::rgba_u> get_kernel_side()
						{
						auto ptr{get_mapped_pointer()};
						std::span<utils::graphics::colour::rgba_u> span{ptr, cuda_gl_texture.sizes().x * cuda_gl_texture.sizes().y};
						return utils::matrix_wrapper<std::span<utils::graphics::colour::rgba_u>>{cuda_gl_texture.sizes(), span};
						}*/
					cudaSurfaceObject_t get_surface_object() const noexcept { return surfObject; }

				private:
					cuda_resource_mapper(gl_texture& cuda_gl_texture) : cuda_gl_texture{cuda_gl_texture} 
						{
						/*if constexpr (utils::compilation::debug)
							{
							if (cuda_gl_texture.debug_is_mapped_to_cuda) { throw std::runtime_error{"Attempting to map to cuda a cuda::gl_texture that was already mapped."}; }
							cuda_gl_texture.debug_is_mapped_to_cuda = true;
							}*/
						utils::cuda::check_throwing(cudaGraphicsMapResources(1, &cuda_gl_texture.cuda_resource_handle, 0));

						cudaArray_t texture_ptr;
						utils::cuda::check_throwing(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_gl_texture.cuda_resource_handle, 0, 0));
						cudaResourceDesc resDesc;
						memset(&resDesc, 0, sizeof(resDesc));
						resDesc.resType = cudaResourceTypeArray;
						resDesc.res.array.array = texture_ptr;
						
						cudaCreateSurfaceObject(&surfObject, &resDesc);
						}

					gl_texture& cuda_gl_texture;
					cudaSurfaceObject_t surfObject;
				};
			friend class cuda_resource_mapper;

			gl_texture(const utils::math::vec2s& sizes) :
				texture{create_texture(sizes)},
				cuda_resource_handle{create_cuda_image(texture.getNativeHandle())}
				{}

			~gl_texture()
				{
				utils::cuda::check(cudaGraphicsUnregisterResource(cuda_resource_handle));
				}
				
			cuda_resource_mapper map_to_cuda() 
				{
				return cuda_resource_mapper{*this}; 
				}

			utils::math::vec2s sizes() const noexcept { return {texture.getSize().x, texture.getSize().y}; }
			const sf::Texture& get_texture() const noexcept { return texture; }

		private:
			sf::Texture texture;
			cudaGraphicsResource* cuda_resource_handle{nullptr};

			inline static sf::Texture create_texture(const utils::math::vec2s& sizes) noexcept 
				{
				sf::Texture ret;
				ret.create(sizes.x, sizes.y);
				unsigned int opengl_pixel_buffer_object_handle{ret.getNativeHandle()};
				return ret;
				}
			inline static cudaGraphicsResource* create_cuda_image(unsigned int opengl_pixel_buffer_object_handle)
				{
				cudaGraphicsResource* ret{nullptr};
				cudaDeviceSynchronize();
				utils::cuda::check_throwing(cudaGraphicsGLRegisterImage(&ret, opengl_pixel_buffer_object_handle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
				return ret;
				}
		};
	}