// Constants
tile_size   = 64;
grid_size = new (globals().utils.vec2)(3, 2);

map_size = new (globals().utils.vec2)
	(
	map_grid_size.x * tile_size, 
	map_grid_size.y * tile_size
	);

// Create and init map
map_grid = new (globals().utils.grid)(grid_size.x, grid_size.y);
map_grid.set_each(function(coords, grid_size)
	{
	return 
		{
		tile_index: (coords.x + (coords.y * grid_size.x)) % 10,
		humidity: random_range(0, 1),
		sunlight: random_range(0, 1)
		};
	}, grid_size);

// Write map data to buffer, transfer buffer to surface
map_render_data_surface_tile_indices = surface_create(map_size.x, map_size.y, surface_r8unorm);
map_render_data_surface_tile_data    = surface_create(map_size.x, map_size.y, surface_rgba16float);

map_render_data_buffer_tile_indices = buffer_create(map_grid.width() * map_grid.height() * buffer_sizeof(buffer_u8 )    , buffer_fixed, 1);
map_render_data_buffer_tile_data    = buffer_create(map_grid.width() * map_grid.height() * buffer_sizeof(buffer_f16) * 4, buffer_fixed, 2);
map_grid.for_each(function(value, coords)
	{
	buffer_write(map_render_data_buffer_tile_indices, buffer_u8 , value.tile_index);
	buffer_write(map_render_data_buffer_tile_data   , buffer_f16, value.humidity);
	buffer_write(map_render_data_buffer_tile_data   , buffer_f16, value.sunlight);
	buffer_write(map_render_data_buffer_tile_data   , buffer_f16, 0);
	buffer_write(map_render_data_buffer_tile_data   , buffer_f16, 0);
	});
buffer_set_surface(map_render_data_buffer_tile_indices, map_render_data_surface_tile_indices, 0);
buffer_set_surface(map_render_data_buffer_tile_data   , map_render_data_surface_tile_data   , 0);
buffer_delete(map_render_data_buffer_tile_indices);
buffer_delete(map_render_data_buffer_tile_data   );

map_render_data_texture_tile_indices = surface_get_texture(map_render_data_surface_tile_indices);
map_render_data_texture_tile_data    = surface_get_texture(map_render_data_surface_tile_data   );

surface_free(map_render_data_surface_tile_indices);
surface_free(map_render_data_surface_tile_data   );

vertex_format_begin();
	vertex_format_add_position();
//	vertex_format_add_colour();
vertex_format = vertex_format_end();

vertex_buffer = vertex_create_buffer();
vertex_begin(vertex_buffer, vertex_format);
	vertex_position(vertex_buffer, 0, 0);
	vertex_position(vertex_buffer, map_size.x, 0);
	vertex_position(vertex_buffer, map_size.x, map_size.y);
	vertex_position(vertex_buffer, map_size.x, map_size.y);
	vertex_position(vertex_buffer, 0, map_size.y);
	vertex_position(vertex_buffer, 0, 0);
vertex_end(vertex_buffer);

map_render_data_texture_tileset = sprite_get_texture(spr_sample_tileset, 0);

shader_uniform_tile_size    = shader_get_uniform(sh_terrain2, "tile_size");
shader_uniform_grid_size    = shader_get_uniform(sh_terrain2, "grid_size");
shader_sampler_grid_indices = shader_get_sampler_index(sh_terrain2, "grid_indices");
shader_sampler_grid_data    = shader_get_sampler_index(sh_terrain2, "grid_data"   );

shader_set(sh_terrain2);
shader_set_uniform_f(shader_uniform_tile_size, tile_size);
shader_set_uniform_f(shader_uniform_grid_size, grid_size);
texture_set_stage(shader_sampler_tile_indices, map_render_data_texture_tile_indices);
texture_set_stage(shader_sampler_tile_data   , map_render_data_texture_tile_data   );
vertex_submit(vertex_buffer, pr_trianglelist, map_render_data_texture_tileset);
shader_reset();

shader_set_uniform_f_buffer()