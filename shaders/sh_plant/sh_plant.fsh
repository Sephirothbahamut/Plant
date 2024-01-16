//
// Simple passthrough fragment shader
//
varying vec2 v_vTexcoord;
varying vec4 v_vColour;

uniform float humidity;
uniform float sunlight;



float map(float value, float min1, float max1, float min2, float max2)
	{
	float perc = (value - min1) / (max1 - min1);
	return perc * (max2 - min2) + min2;
	}

vec3 evaluate_base_colour(float percent)
	{
	const float split_point = .35;
	vec3 a = vec3(  0, 127,  14) / 255.;
	vec3 b = vec3(107,  56,  23) / 255.;
	vec3 c = vec3( 68,  50,  37) / 255.;
	
	if(percent > split_point)
		{
		percent = map(percent, split_point, 1., 0., 1.);
		return mix(b, a, percent);
		}
	else
		{
		percent = map(percent, 0., split_point, 0., 1.);
		return mix(c, b, percent);
		}
		
	return vec3(0., 0., 0.);
	}


void main()
	{
	float humidity_multiplier = humidity;
	
	gl_FragColor = texture2D(gm_BaseTexture, v_vTexcoord) * vec4(evaluate_base_colour(humidity_multiplier), 1.);
	}
