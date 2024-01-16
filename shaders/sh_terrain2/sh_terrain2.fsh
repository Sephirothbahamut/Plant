uniform vec2  grid_size;
uniform float tile_size;
uniform floatBuffer buffer_data;

void main()
	{
	vec2 pixel_coords_in_tile = mod(gl_Position, tile_size);
	vec2 tile_coords_in_grid = floor(gl_Position / tile_size);
	
		
		
		
	gl_FragColor = vec4(1, 0, 0, 1);
    //gl_FragColor = v_vColour * texture2D( gm_BaseTexture, v_vTexcoord );
	gl_FragColor = texture
	}
