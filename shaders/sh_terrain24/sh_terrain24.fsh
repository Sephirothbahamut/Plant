uniform vec2  grid_size;
uniform float tile_size;

varying vec2  uv;
varying float tile_index;
varying float humidity;
varying float sunlight;

void main()
	{
	//vec2 pixel_coords_in_tile = mod(gl_Position, tile_size);
	//vec2 tile_coords_in_grid = floor(gl_Position / tile_size);
	
		
		
		
	gl_FragColor = vec4(1, uv.x, uv.y, 1);
    //gl_FragColor = v_vColour * texture2D( gm_BaseTexture, v_vTexcoord );
	//gl_FragColor = texture;
	}
