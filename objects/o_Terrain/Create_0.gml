/// @description Insert description here
// You can write your code in this editor

data = ds_grid_create(10, 10);

for (var _x = 0; _x < ds_grid_width(data); _x += 1)
	{
    for (var _y = 0; _y < ds_grid_height(data); _y += 1)
		{
		ds_grid_set(data, _x, _y, new Tile());
		}
	}

