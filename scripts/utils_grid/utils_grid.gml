function utils_grid(in_width = 1, in_height = 1) constructor
	{
	data = ds_grid_create(globals().utils.t.int(in_width), globals().utils.t.int(in_height));
	
	for_each = function(callback)
		{
		for(var iy = 0; iy < width(); iy++)
			{
			for(var ix = 0; ix < height(); ix++)
				{
				callback(ds_grid_get(data, ix, iy), new (globals().utils.vec2)(ix, iy))
				}
			}
		}
		
	set_each = function(callback)
		{
		for(var iy = 0; iy < width(); iy++)
			{
			for(var ix = 0; ix < height(); ix++)
				{
				ds_grid_set(data, ix, iy, callback(new (globals().utils.vec2)(ix, iy)));
				}
			}
		}
		
	width  = function() { return ds_grid_width (data); }
	height = function() { return ds_grid_height(data); }
	}