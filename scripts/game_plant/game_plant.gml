function game_plant(width, height) constructor
	{
	data = new utils_grid(width, height);

	data.set_each(function(coords)
		{
		return new game_plant_tile_data();
		});
		
	draw = function()
		{
		shader_set(globals().shaders.plant.shader);
		
		data.for_each(function(value, coords)
			{
			var x1 = coords.x * globals().constants.tile_size;
			var y1 = coords.y * globals().constants.tile_size;
			var x2 = x1       + globals().constants.tile_size;
			var y2 = y1       + globals().constants.tile_size;
		
			shader_set_uniform_f(globals().shaders.plant.parameters.humidity, value.humidity);
			shader_set_uniform_f(globals().shaders.plant.parameters.sunlight, value.sunlight);
	
			var scale = utils_sprite().size_to_scale
				(
				spr_plant_default,
				globals().constants.tile_size,
				globals().constants.tile_size
				);
			draw_sprite_ext(spr_plant_default, -1, x1, y1, scale.x, scale.y, 0, c_white, 1);
			});
			
		shader_reset();
		}
	}

