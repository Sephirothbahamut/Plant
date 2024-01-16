function utils_sprite()
	{
	return
		{
		size_to_scale: function(sprite_index, width, height)
			{
			var sprite_w = sprite_get_width (sprite_index);
			var sprite_h = sprite_get_height(sprite_index);
		
			return new utils_vec2(sprite_w / width, sprite_h / height);
			}
		}
	}