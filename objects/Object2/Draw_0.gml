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


vertex_format_begin();
	vertex_format_add_position();
	vertex_format_add_texcoord(); // (tile_id, unused)
	vertex_format_add_texcoord(); // (humidity, sunlight)
vertex_format = vertex_format_end();


vertex_buffer = vertex_create_buffer();
vertex_begin(vertex_buffer, vertex_format);
	map_grid.for_each(function(value, coords) 
		{
		x1 = coords.x * tile_size;
		y1 = coords.y * tile_size;
		x2 = x1       + tile_size;
		y2 = y1       + tile_size;
		
		vertex_position(vertex_buffer, x1, y1);
		vertex_texcoord(vertex_buffer, value.tile_index, 0);
		vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
		vertex_position(vertex_buffer, x2, y1);
		vertex_position(vertex_buffer, x2, y2);
		vertex_texcoord(vertex_buffer, value.tile_index, 0);
		vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
		vertex_position(vertex_buffer, x2, y2);
		vertex_texcoord(vertex_buffer, value.tile_index, 0);
		vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
		vertex_position(vertex_buffer, x1, y2);
		vertex_texcoord(vertex_buffer, value.tile_index, 0);
		vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
		vertex_position(vertex_buffer, x1, y1);
		vertex_texcoord(vertex_buffer, value.tile_index, 0);
		vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
		});
vertex_end(vertex_buffer);

shader_uniform_tile_size   = shader_get_uniform(sh_terrain2, "tile_size");
shader_uniform_grid_size   = shader_get_uniform(sh_terrain2, "grid_size");
shader_uniform_buffer_data = shader_get_uniform(sh_terrain2, "buffer_data");

shader_set(sh_terrain2);
shader_set_uniform_f(shader_uniform_tile_size, tile_size);
shader_set_uniform_f(shader_uniform_grid_size, grid_size);
vertex_submit(vertex_buffer, pr_trianglelist, map_render_data_texture_tileset);
shader_reset();



	
buffer_delete(map_render_data_buffer);

//////////////// shader_set_uniform_f_buffer

// Write map data to buffer
//var buffer_data_count_per_element = 3; // Tile_index, humidity, sunlight
//var buffer_elements_count = map_grid.width() * map_grid.height();
//var buffer_floats_count = buffer_data_count_per_element * buffer_floats_count;
//var buffer_data_type_size = buffer_sizeof(buffer_f32);
//var buffer_total_size = buffer_floats_count * buffer_data_type_size;
//
//map_render_data_buffer = buffer_create(buffer_total_size, buffer_fixed, 1);
//
//map_grid.for_each(function(value, coords)
//	{
//	buffer_write(map_render_data_buffer, buffer_f32, value.tile_index);
//	buffer_write(map_render_data_buffer, buffer_f32, value.humidity);
//	buffer_write(map_render_data_buffer, buffer_f32, value.sunlight);
//	});
//
//map_render_data_texture_tile_indices = surface_get_texture(map_render_data_surface_tile_indices);
//map_render_data_texture_tile_data    = surface_get_texture(map_render_data_surface_tile_data   );
//
//surface_free(map_render_data_surface_tile_indices);
//surface_free(map_render_data_surface_tile_data   );
//
//vertex_format_begin();
//	vertex_format_add_position();
////	vertex_format_add_colour();
//vertex_format = vertex_format_end();
//
//vertex_buffer = vertex_create_buffer();
//vertex_begin(vertex_buffer, vertex_format);
//	vertex_position(vertex_buffer, 0, 0);
//	vertex_position(vertex_buffer, map_size.x, 0);
//	vertex_position(vertex_buffer, map_size.x, map_size.y);
//	vertex_position(vertex_buffer, map_size.x, map_size.y);
//	vertex_position(vertex_buffer, 0, map_size.y);
//	vertex_position(vertex_buffer, 0, 0);
//vertex_end(vertex_buffer);
//
//map_render_data_texture_tileset = sprite_get_texture(spr_sample_tileset, 0);
//
//shader_uniform_tile_size   = shader_get_uniform(sh_terrain2, "tile_size");
//shader_uniform_grid_size   = shader_get_uniform(sh_terrain2, "grid_size");
//shader_uniform_buffer_data = shader_get_uniform(sh_terrain2, "buffer_data");
//
//shader_set(sh_terrain2);
//shader_set_uniform_f(shader_uniform_tile_size, tile_size);
//shader_set_uniform_f(shader_uniform_grid_size, grid_size);
//shader_set_uniform_f_buffer(shader_sampler_buffer_data, map_render_data_buffer, 0, buffer_floats_count);
//vertex_submit(vertex_buffer, pr_trianglelist, map_render_data_texture_tileset);
//shader_reset();
//
//
//
//	
//buffer_delete(map_render_data_buffer);