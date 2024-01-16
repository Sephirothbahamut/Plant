uniform vec2  grid_size;
uniform float tile_size;
uniform sampler2D grid_indices;
uniform sampler2D grid_data;

void main()
	{
	vec2 pixel_coords_in_tile = mod(gl_Position, tile_size);
	vec2 tile_coords_in_grid = floor(gl_Position / tile_size);
	
	uint tile_index = texture2D(grid_indices, tile_coords_in_grid);	
	float humidity = texture2D(grid_data)
		
		
		
	gl_FragColor = vec4(1, 0, 0, 1);
    //gl_FragColor = v_vColour * texture2D( gm_BaseTexture, v_vTexcoord );
	gl_FragColor = texture
	}
