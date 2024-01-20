function game_terrain(width, height) constructor
	{
	data = new utils_grid(width, height);
	data1 = new utils_grid(width, height);
	renderer = new (globals().game.terrain.renderer)();

	data.set_each(function(coords, passthrough)
		{
		tile_data = new game_terrain_tile_data();
		tile_data.tile_index = (coords.x + (coords.y * passthrough.grid_width)) % 10;
		return tile_data;
		}, {grid_width: width});
	data1.set_each(function(coords, passthrough)
		{
		tile_data = new game_terrain_tile_data();
		tile_data.tile_index = (coords.x + (coords.y * passthrough.grid_width)) % 10;
		return tile_data;
		}, {grid_width: width});
		
	draw = function()
		{
		renderer.draw_vertexbuffer(data);
		}
		
	step = function()
		{
		data.for_each(function(tile_data)
			{
			tile_data.humidity = (sin(get_timer() / 1000000) + 1)/2;
			})
			
		var pippo = {value: 0};
		data1.for_each(function(tile_data, coords, pippo)
			{
			pippo.value += tile_data.humidity;
			tile_data.humidity /= 2;
			}, pippo)
		}
	}

