function utils_grid(in_width = 1, in_height = 1) constructor
	{
	data = ds_grid_create(globals().utils.t.int(in_width), globals().utils.t.int(in_height));
	
	for_each = function(callback, captures = {})
		{
		for(var iy = 0; iy < height(); iy++)
			{
			for(var ix = 0; ix < width(); ix++)
				{
				callback(ds_grid_get(data, ix, iy), new (globals().utils.vec2)(ix, iy), captures)
				}
			}
		}
		
	set_each = function(callback, captures = {})
		{
		for(var iy = 0; iy < height(); iy++)
			{
			for(var ix = 0; ix < width(); ix++)
				{
				ds_grid_set(data, ix, iy, callback(new (globals().utils.vec2)(ix, iy), captures));
				}
			}
		}
		
	width  = function() { return ds_grid_width (data); }
	height = function() { return ds_grid_height(data); }
	}