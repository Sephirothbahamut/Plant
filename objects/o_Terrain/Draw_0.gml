/// @description Insert description here
// You can write your code in this editor
/*
for (var ix = 0; ix < ds_grid_width(data); ix += 1)
	{
    for (var iy = 0; iy < ds_grid_height(data); iy += 1)
		{
		var tile_data = ds_grid_get(data, ix, iy);
		
		
		
		var colour = make_colour_hsv(colour_get_hue(c_olive), tile_data.humidity * 255, 255);
		}
	}
*/
//draw_sprite
//draw_tile


grid.for_each(function(value, coords)
	{
	var x1 = coords.x * tile_size;
	var y1 = coords.y * tile_size;
	var x2 = x1 + tile_size;
	var y2 = y1 + tile_size;
		
	shader_set(sh_terrain);
	var shader_param_humidity = shader_get_uniform(sh_terrain, "terrain_humidity");
	var shader_param_sunlight = shader_get_uniform(sh_terrain, "terrain_sunlight");
	shader_set_uniform_f(shader_param_humidity, value.humidity);
	shader_set_uniform_f(shader_param_sunlight, value.sunlight);
	
	var scale = namespace().utils.sprite.size_to_scale(spr_terrain_default, tile_size, tile_size);
	draw_sprite_ext(spr_terrain_default, -1, x1, y1, scale.x, scale.y, 0, c_white, 1);
	
	shader_reset();
	});
	