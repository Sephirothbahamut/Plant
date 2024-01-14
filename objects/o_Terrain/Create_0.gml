tile_size = 64;

grid = new (namespace().utils.grid)(5, 5);

grid.set_each(function(coords)
	{
	return new (namespace().terrain.tile_data)();
	});