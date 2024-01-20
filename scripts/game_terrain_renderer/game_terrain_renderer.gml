
function game_terrain_renderer_globals() constructor
	{
	vertex_format_begin();
	vertex_format_add_position();
	vertex_format_add_texcoord(); // (u, v)
	vertex_format_add_texcoord(); // (tile_id, unused)
	vertex_format_add_texcoord(); // (humidity, sunlight)
	vertex_format = vertex_format_end();
	
	shader_uniform_tile_size = shader_get_uniform(sh_terrain2, "tile_size");
	shader_uniform_grid_size = shader_get_uniform(sh_terrain2, "grid_size");
	
	tileset_texture = sprite_get_texture(spr_sample_tileset, 0);
	
	destructor = function()
		{
		vertex_format_delete(vertex_format);
		}
	}
	
function game_terrain_renderer() constructor
	{
	static renderer_globals = new game_terrain_renderer_globals();
	
	tile_size = 64;
	
	//draw_texture_buffer = function(map_grid)
	//	{
	//	
	//	};
	
	draw_vertexbuffer = function(map_grid)
		{
		vertex_buffer = vertex_create_buffer();
		vertex_begin(vertex_buffer, renderer_globals.vertex_format);
		map_grid.for_each_subregion
			(
			function(value, coords) 
				{
				var x1 = coords.x * tile_size;
				var y1 = coords.y * tile_size;
				var x2 = x1       + tile_size;
				var y2 = y1       + tile_size;
				vertex_position(vertex_buffer, x1, y1);
					vertex_texcoord(vertex_buffer, 0, 0);
					vertex_texcoord(vertex_buffer, value.tile_index, 0);
					vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
				vertex_position(vertex_buffer, x2, y1);
					vertex_texcoord(vertex_buffer, 1, 0);
					vertex_texcoord(vertex_buffer, value.tile_index, 0);
					vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
				vertex_position(vertex_buffer, x2, y2);
					vertex_texcoord(vertex_buffer, 1, 1);
					vertex_texcoord(vertex_buffer, value.tile_index, 0);
					vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
				vertex_position(vertex_buffer, x2, y2);
					vertex_texcoord(vertex_buffer, 1, 1);
					vertex_texcoord(vertex_buffer, value.tile_index, 0);
					vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
				vertex_position(vertex_buffer, x1, y2);
					vertex_texcoord(vertex_buffer, 0, 1);
					vertex_texcoord(vertex_buffer, value.tile_index, 0);
					vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
				vertex_position(vertex_buffer, x1, y1);
					vertex_texcoord(vertex_buffer, 0, 0);
					vertex_texcoord(vertex_buffer, value.tile_index, 0);
					vertex_texcoord(vertex_buffer, value.humidity, value.sunlight);
				},
			new (globals().utils.vec2)( 0,  0), 
			new (globals().utils.vec2)(30, 30)
			);
		vertex_end(vertex_buffer);

		shader_set(sh_terrain2);
		shader_set_uniform_f(renderer_globals.shader_uniform_tile_size, tile_size, tile_size);
		shader_set_uniform_f(renderer_globals.shader_uniform_grid_size, map_grid.width(), map_grid.height());
		vertex_submit(vertex_buffer, pr_trianglelist, renderer_globals.tileset_texture);
		shader_reset();
		vertex_delete_buffer(vertex_buffer);
		};
	}

