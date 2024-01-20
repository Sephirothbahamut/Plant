function globals()
	{
	static instance =
		{
		constants:
			{
			tile_size: 64
			},
	
		utils:
			{
			//t     : utils_t(),
			vec2  : utils_vec2,
			grid  : utils_grid,
			sprite: utils_sprite()
			},
	
		game:
			{
			plant: 
				{
				class: game_plant,
				tile_data: game_plant_tile_data
				},
			terrain: 
				{
				class: game_terrain,
				tile_data: game_terrain_tile_data,
				renderer: game_terrain_renderer
				}
			},
	
		shaders:
			{
			terrain: 
				{
				shader: sh_terrain,
				parameters:
					{
					humidity: shader_get_uniform(sh_terrain, "humidity"),
					sunlight: shader_get_uniform(sh_terrain, "sunlight")
					}
				},
			plant: 
				{
				shader: sh_plant,
				parameters:
					{
					humidity: shader_get_uniform(sh_plant, "humidity"),
					sunlight: shader_get_uniform(sh_plant, "sunlight")
					}
				},
			}
		};
		
	return instance;
	}