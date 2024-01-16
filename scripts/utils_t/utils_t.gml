function error(type_name)	
	{
	show_error("Expected type \"" + type_name + "\".", true);
	}

function utils_t()
	{
	return 
		{
		int: function(value, default_value)
			{
			if(is_undefined(value))
				{
				if(is_undefined(default_value)) { error("int"); }
				else { return default_value; }
				}
			else if(!(is_int32(value) || is_int64(value) || (is_numeric(value) && (floor(value) == value))))
				{
				error("int");
				}
			return value;
			},
		float: function(value, default_value)
			{
			if(is_undefined(value))
				{
				if(is_undefined(default_value)) { error("int"); }
				else { return default_value; }
				}
			else if(!is_numeric(value)) { error("float"); }
			return value;
			},
		str: function(value, default_value)
			{
			if(is_undefined(value))
				{
				if(is_undefined(default_value)) { error("string"); }
				else { return default_value; }
				}
			else if(!is_string(value)) { error("string"); }
			return value;
			},
		struct: function(value, type)
			{
			if(is_undefined(type )) { error("Struct type check must specify the struct type."); }
			if(is_undefined(value) || !is_struct(value) || !is_instanceof(value, type)) { error("struct " + struct); }
			return value;
			}
		}
	}