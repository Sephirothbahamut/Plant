//
// Simple passthrough vertex shader
//
attribute vec3 in_Position;
attribute vec2 in_uv;
attribute vec2 in_tile_index;
attribute vec2 in_tile_data;

varying vec2 uv;
varying float tile_index;
varying float humidity;
varying float sunlight;

void main()
{
    vec4 object_space_pos = vec4( in_Position.x, in_Position.y, in_Position.z, 1.0);
	
	uv         = in_uv;
	tile_index = in_tile_index.x;
	humidity   = in_tile_data .x;
	sunlight   = in_tile_data .y;
	
    gl_Position = gm_Matrices[MATRIX_WORLD_VIEW_PROJECTION] * object_space_pos;
}
