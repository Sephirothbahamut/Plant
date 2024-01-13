/// @description Insert description here
// You can write your code in this editor

var tilemap_layer = layer_get_id("tiles_terrain");
var tilemap_id = layer_tilemap_get_id(tilemap_layer);

for (var _x = 0; _x < ds_grid_width(data); _x += 1)
	{
    for (var _y = 0; _y < ds_grid_height(data); _y += 1)
		{
		var tile_data = ds_grid_get(data, _x, _y);
		var colour = make_colour_rgb(tile_data.humidity, tile_data.humidity, tile_data.humidity);
		draw_set_colour(colour);
		
		//draw_rectangle(x1, y1, x2, y2, false);
		
		var tilemap_tile = tilemap_get(tilemap_id, _x, _y);
		tile_set_index(tilemap_tile, tile_data.humidity < .5 ? 1 : 2);
		tilemap_set(tilemap_id, tilemap_tile, _x, _y);
		}
	}

draw_sprite
draw_tile