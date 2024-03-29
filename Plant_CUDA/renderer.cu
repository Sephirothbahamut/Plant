#include "renderer.h"


#include "cuda.cuh"
#include "cuda_noise.cuh"


template <typename T, size_t size>
utils_gpu_available constexpr utils::math::vec<T, size> operator*(const utils::matrix<T, size, size>& mat, const utils::math::vec<T, size>& vec)
	{
	utils::math::vec<T, size> ret{static_cast<T>(0)};
	for (size_t i = 0; i < size; i++)
		{
		for (size_t j = 0; j < size; j++)
			{
			ret[i] += mat[utils::math::vec2s{i, j}] * vec[j];
			}
		}
	return ret;
	}


template <size_t size, /*std::invocable<float, utils::math::vec<float, size>>*/auto noise>
utils_gpu_available float fbm(utils::math::vec<float, size> x, size_t passes_count)
	{
	constexpr utils::matrix<float, 3, 3> m3
		{
		 0.00f,  0.80f,  0.60f,
		-0.80f,  0.36f, -0.48f,
		-0.60f, -0.48f,  0.64f
		};
	constexpr utils::matrix<float, 2, 2> m2
		{
		 0.80f,  0.60f,
		-0.60f,  0.80f
		};

	float f{1.90};
	float s{0.55};
	float a{0.00};
	float b{0.50};

	for (size_t i{0}; i < passes_count; i++)
		{
		float n{noise(x * f)};
		a += b * n;
		b *= s;
		if constexpr (size == 2) { x = (m2 * x) * f; }
		if constexpr (size == 3) { x = (m3 * x) * f; }
		}
	return a;
	}

__device__ float simplex_noise(utils::math::vec2f coords)
	{
	return cudaNoise::simplexNoise({coords.x, coords.y, 0.f}, 1.f, 123456);
	}
__device__ float perlin_noise(utils::math::vec2f coords)
	{
	return cudaNoise::perlinNoise({coords.x, coords.y, 0.f}, 1.f, 123456);
	}

__device__ float fancy_pattern(utils::math::vec2f p)
	{//https://iquilezles.org/articles/warp/
	constexpr size_t passes{5};

	utils::math::vec2f q
		{
		fbm<2, perlin_noise>(p + utils::math::vec2f{0.0f, 0.0f}, passes),
		fbm<2, perlin_noise>(p + utils::math::vec2f{5.2f, 1.3f}, passes),
		};
	utils::math::vec2f r
		{
		fbm<2, perlin_noise>(p + (q * 4.f) + utils::math::vec2f{1.7f, 9.2f}, passes),
		fbm<2, perlin_noise>(p + (q * 4.f) + utils::math::vec2f{8.3f, 2.8f}, passes),
		};

	return fbm<2, perlin_noise>(p + (r * 4.0), 3);
	}




__global__ void draw_texture(utils::cuda::render_target render_target, const utils::cuda::kernel::texture<utils::graphics::colour::rgba_f>& texture)
	{
	const auto coords{utils::cuda::kernel::coordinates::total::vec2()};
	if (!render_target.validate_coords(coords)) { return; }

	uchar4 pixel_data;
	if (texture.validate_coords(coords))
		{
		pixel_data =
			{
			static_cast<uint8_t>(texture[coords].r * 255.f),
			static_cast<uint8_t>(texture[coords].g * 255.f),
			static_cast<uint8_t>(texture[coords].b * 255.f),
			static_cast<uint8_t>(texture[coords].a * 255.f)
			};
		}
	else
		{
		pixel_data =
			{
			static_cast<uint8_t>((coords.x / (render_target.sizes.x / 2.f)) * 255.f),
			static_cast<uint8_t>((coords.y / (render_target.sizes.y / 2.f)) * 255.f),
			static_cast<uint8_t>(255.f / 2.f),
			static_cast<uint8_t>(255.f)
			};
		}
	surf2Dwrite(pixel_data, render_target.surface, coords.x * sizeof(uchar4), coords.y);
	}

__device__ utils::graphics::colour::rgba_f texture_tile_pixel(const utils::cuda::kernel::texture<utils::graphics::colour::rgba_f>& texture, const utils::math::vec2s& tile_coords, const utils::math::vec2s& pixel_in_tile)
	{
	const auto pixel_coords{(tile_coords * size_t{64}) + pixel_in_tile};
	return texture[pixel_coords];
	}
__device__ utils::graphics::colour::rgba_f texture_tile_pixel_animated(const utils::cuda::kernel::texture<utils::graphics::colour::rgba_f>& texture, size_t tile_id, const utils::math::vec2s& pixel_in_tile, float time)
	{
	const utils::math::vec2s tile_coords{std::fmod((time * 3.f), 8.f), tile_id};
	return texture_tile_pixel(texture, tile_coords, pixel_in_tile);
	}

__device__ float terrain_colour_green_patch_mask(const utils::math::vec2f& coords)
	{
	const float ret{cudaNoise::repeaterTurbulence({coords.x, coords.y, 0.f}, 0.05f, 0.01f, 123456, 8.f, 2, cudaNoise::BASIS_PERLIN, cudaNoise::BASIS_PERLIN)};
	return (ret + 1.f) / 2.f;
	}

__device__ float terrain_colour_value(const utils::math::vec2f& coords)
	{
	auto ret{utils::math::clamp(::fancy_pattern(coords * .001f), 0.f, 1.f)};
	return utils::math::map(0.f, 1.f, .5f, 1.f, ret);
	}

template <size_t size>
__device__ utils::graphics::colour::details::rgb<float, size> colour_apply_humidity_saturation(const utils::graphics::colour::details::rgb<float, size>& colour, float humidity)
	{
	auto hsv{colour.hsv()};
	hsv.s = utils::math::map(0.f, 1.f, .2f, 1.f, humidity);
	return hsv.rgb();
	}

__device__ utils::graphics::colour::rgb_f terrain_colour(float humidity, float green_patch)
	{
	constexpr float grass_threshold{.6f};

	constexpr utils::graphics::colour::rgb_f c_green{0.0f, 1.00f, 0.f};
	constexpr utils::graphics::colour::rgb_f c_brown{0.5f, 0.35f, 0.f};

	auto brown{colour_apply_humidity_saturation(c_brown, humidity)};
	if (humidity < grass_threshold) { return brown; }

	float remaining_range{utils::math::map(grass_threshold, 1.f, 0.f, 1.f, humidity)};

	auto green{colour_apply_humidity_saturation(c_green, remaining_range)};

	float t{remaining_range * green_patch};
	//return utils::math::lerp(brown, green, t);
	return utils::math::lerp(brown, c_green, green_patch * remaining_range);
	}

__device__ utils::graphics::colour::rgba_f terrain_colour(const utils::math::vec2f& coords, float humidity)
	{
	const auto rgb{terrain_colour(humidity, terrain_colour_green_patch_mask(coords))};
	const auto value{terrain_colour_value(coords)};
	return {rgb.r * value, rgb.g * value, rgb.b * value, 1.f};
	}

__device__ utils::graphics::colour::rgba_f plant_humidity_colour(float humidity)
	{
	constexpr utils::graphics::colour::rgba_f colour_plant_max{0.0f, 1.0f, 0.0f};
	constexpr utils::graphics::colour::rgba_f colour_plant_mid{.98f, .98f, .40f};
	constexpr utils::graphics::colour::rgba_f colour_plant_min{.39f, .22f, .12f};
	constexpr float                           mid_threshold   {.3f};

	if (humidity > mid_threshold)
		{
		const float t{utils::math::map(mid_threshold, 1.f, 0.f, 1.f, humidity)};
		return utils::math::lerp(colour_plant_mid, colour_plant_max, t);
		}
	else
		{
		const float t{utils::math::map(0.f, mid_threshold, 0.f, 1.f, humidity)};
		return utils::math::lerp(colour_plant_min, colour_plant_mid, t);
		}
	}

__device__ size_t plant_get_tileset_index(const utils::matrix_wrapper<std::span<game::tile>>& game_grid, const utils::math::vec2s coords_of_tile)
	{
	const bool ll{game_grid[game::coords::tile_neighbour(coords_of_tile, game_grid.sizes(), {-1,  0})].plant.humidity > 0.f};
	const bool up{game_grid[game::coords::tile_neighbour(coords_of_tile, game_grid.sizes(), { 0, -1})].plant.humidity > 0.f};
	const bool rr{game_grid[game::coords::tile_neighbour(coords_of_tile, game_grid.sizes(), { 1,  0})].plant.humidity > 0.f};
	const bool dw{game_grid[game::coords::tile_neighbour(coords_of_tile, game_grid.sizes(), { 0,  1})].plant.humidity > 0.f};

	if ( ll &&  up &&  rr &&  dw) { return  0; }//all

	if (!ll &&  up && !rr &&  dw) { return  1; }//ver
	if ( ll && !up &&  rr && !dw) { return  2; }//hor

	if (!ll &&  up &&  rr &&  dw) { return  5; }//wall ll
	if ( ll && !up &&  rr &&  dw) { return  4; }//wall up
	if ( ll &&  up && !rr &&  dw) { return  3; }//wall rr
	if ( ll &&  up &&  rr && !dw) { return  6; }//wall dw
	
	if ( ll &&  up && !rr && !dw) { return  9; }//curve ll up
	if (!ll &&  up &&  rr && !dw) { return  8; }//curve up rr
	if (!ll && !up &&  rr &&  dw) { return  7; }//curve rr dw
	if ( ll && !up && !rr &&  dw) { return 10; }//curve dw ll
	
	if (!ll && !up &&  rr && !dw) { return 12; }//tail ll
	if (!ll && !up && !rr &&  dw) { return 13; }//tail up
	if ( ll && !up && !rr && !dw) { return 14; }//tail rr
	if (!ll &&  up && !rr && !dw) { return 11; }//tail dw

	return 15;
	}

__device__ utils::graphics::colour::rgba_f plant_colour(const utils::matrix_wrapper<std::span<game::tile>>& game_grid, const utils::cuda::kernel::texture<utils::graphics::colour::rgba_f>& texture, const game::tiles::plant& tile, const utils::math::vec2s& coords_of_tile, const utils::math::vec2s& coords_in_tile, float time, float interpolation)
	{
	if(tile.humidity == 0.f) { return {0.f, 0.f, 0.f, 0.f}; }

	const auto texture_plant_multiplier{texture_tile_pixel_animated(texture, plant_get_tileset_index(game_grid, coords_of_tile), coords_in_tile, time)};
	auto base_plant{plant_humidity_colour(utils::math::lerp(tile.humidity, tile.humidity_next, interpolation))};
	base_plant.r *= texture_plant_multiplier.r;
	base_plant.g *= texture_plant_multiplier.g;
	base_plant.b *= texture_plant_multiplier.b;
	base_plant.a  = texture_plant_multiplier.a;

	auto base_flower{texture_tile_pixel_animated(texture, 15, coords_in_tile, time)};
	base_flower.r =       tile.absorption;
	base_flower.b = 1.f - tile.absorption;
	return base_plant.blend(base_flower); //TODO blend, irrelevant now cause we don't have half-transparency in our tileset
	}

__global__ void draw(utils::cuda::render_target render_target, utils::cuda::kernel::texture<utils::graphics::colour::rgba_f> tileset_texture, game::kernel_data_for_draw game_state, float time, float interpolation)
	{
	const auto coords{utils::cuda::kernel::coordinates::total::vec2()};
	if (!render_target.validate_coords(coords)) { return; }

	const auto evaluated_coords{game::coords::evaluate(coords, game_state.grid.sizes(), game_state.camera_transform)};

	const auto& tile{game_state.grid[evaluated_coords.of_tile]};

	//utils::graphics::colour::rgba_f colour_terrain{terrain_colour(evaluated_coords.in_world, tile.terrain.get_humidity(time))};
	float colour_terrain_shade{utils::math::map(0.f, 1.f, .5f, 1.f, terrain_colour_green_patch_mask(evaluated_coords.in_world))};
	utils::graphics::colour::rgba_f colour_terrain
		{
		tile.terrain.sunlight_starting  * colour_terrain_shade,
		tile.terrain.get_humidity(time) * colour_terrain_shade,
		tile.terrain.get_humidity(time) * colour_terrain_shade
		};


	utils::graphics::colour::rgba_f colour_plant  {plant_colour  (game_state.grid, tileset_texture, tile.plant, evaluated_coords.of_tile, evaluated_coords.in_tile, time, interpolation)};

	utils::graphics::colour::rgba_f pixel_colour{colour_terrain.blend(colour_plant)};



	////////////////////////////////// Mouseover tile begin
	if (evaluated_coords.of_tile == game_state.mouse_tile)
		{
		if (evaluated_coords.in_tile.x == 0 || evaluated_coords.in_tile.x == 63 ||
			evaluated_coords.in_tile.y == 0 || evaluated_coords.in_tile.y == 63)
			{
			pixel_colour.b = utils::math::map(0.f, 1.f, .8f, 1.f, pixel_colour.b);
			}
		else
			{
			pixel_colour.b = utils::math::map(0.f, 1.f, .3f, 1.f, pixel_colour.b);
			}
		}
	////////////////////////////////// Mouseover tile end
	else
	////////////////////////////////// World edge begin
	if ((evaluated_coords.of_tile.x == 0 || evaluated_coords.of_tile.y == 0) &&
		(
		evaluated_coords.in_tile.x == 0 || evaluated_coords.in_tile.y == 0 ||
		evaluated_coords.in_tile.x == 1 || evaluated_coords.in_tile.y == 1
		))
		{
		pixel_colour.r = utils::math::map(0.f, 1.f, 0.5f, 1.f, pixel_colour.r);
		pixel_colour.g = utils::math::map(0.f, 1.f, 0.5f, 1.f, pixel_colour.g);
		pixel_colour.b = utils::math::map(0.f, 1.f, 0.5f, 1.f, pixel_colour.b);
		}
	else
	if ((evaluated_coords.of_tile.x == game_state.grid.width() - 1|| evaluated_coords.of_tile.y == game_state.grid.height() - 1) &&
		(
		evaluated_coords.in_tile.x == 63 || evaluated_coords.in_tile.y == 63 ||
		evaluated_coords.in_tile.x == 62 || evaluated_coords.in_tile.y == 62
		))
		{
		pixel_colour.r = utils::math::map(0.f, 1.f, 0.f, 0.5f, pixel_colour.r);
		pixel_colour.g = utils::math::map(0.f, 1.f, 0.f, 0.5f, pixel_colour.g);
		pixel_colour.b = utils::math::map(0.f, 1.f, 0.f, 0.5f, pixel_colour.b);
		}
	////////////////////////////////// World edge end
	else
	////////////////////////////////// Tile edge begin
	if (evaluated_coords.in_tile.x == 0 || evaluated_coords.in_tile.x == 63 ||
		evaluated_coords.in_tile.y == 0 || evaluated_coords.in_tile.y == 63)
		{
		pixel_colour.r = utils::math::map(0.f, 1.f, .3f, 1.f, pixel_colour.r);
		pixel_colour.g = utils::math::map(0.f, 1.f, .3f, 1.f, pixel_colour.g);
		pixel_colour.b = utils::math::map(0.f, 1.f, .3f, 1.f, pixel_colour.b);
		}
	////////////////////////////////// Tile edge end





	//utils::math::vec2s game_coords{coords / 64};
	//const auto& game_tile{game_state[coords]};
	//
	//
	//float base_terrain_value{base_terrain({coords})};
	//
	//utils::graphics::colour::rgba_f pixel;
	//
	//utils::graphics::colour::hsva_f pixel_hsv;
	//pixel_hsv.h = 0.f;
	//pixel_hsv.s = game_tile.terrain.get_humidity(time);
	//pixel_hsv.v = base_terrain_value;
	//
	//pixel = pixel_hsv.rgb();
	//
	//pixel.r = base_terrain_value;
	//
	//pixel.g = 1.f;

	/*utils::graphics::colour::rgba_f pixel{1.f, 0.5f, 0.f, 1.f};

	float base_terrain_value{base_terrain({coords})};
	pixel.r = base_terrain_value;
	pixel.g = base_terrain_value;
	pixel.b = base_terrain_value;*/

	uchar4 render_target_pixel
		{
		static_cast<uint8_t>(pixel_colour.r * 255.f),
		static_cast<uint8_t>(pixel_colour.g * 255.f),
		static_cast<uint8_t>(pixel_colour.b * 255.f),
		static_cast<uint8_t>(pixel_colour.a * 255.f)
		};
	surf2Dwrite(render_target_pixel, render_target.surface, coords.x * sizeof(uchar4), coords.y);
	}


namespace renderer
	{
	void renderer::draw(utils::cuda::render_target render_target, float interpolation)
		{
		utils::cuda::device::params_t threads
			{
			.threads
				{
				utils::cuda::device::params_t::threads_t::deduce
					(
					utils::cuda::dim3{8u, 8u},
					utils::cuda::dim3
						{
						render_target.sizes.x,
						render_target.sizes.y,
						}
					)
				}
			};

		const float time{utils::math::lerp(game_ptr->data_cpu.time, game_ptr->data_cpu.next_time, interpolation)};
		utils::cuda::device::call(&::draw, threads, render_target, kernel_tileset_texture, game_ptr->kernel_data_for_draw(), time, interpolation);

		//cudaDeviceSynchronize();
		}
	}