//
// Simple passthrough fragment shader
//
varying vec2 v_vTexcoord;
varying vec4 v_vColour;

uniform float terrain_humidity;
uniform float terrain_sunlight;

// Colour conversions: https://forum.gamemaker.io/index.php?threads/color-math.105462/

vec3 hsv(vec3 c)
	{
    //Near-zero epsilon (to avoid division by 0.0)
    #define E 1e-7

    //Channel shift vector
    const vec4 S = vec4(0, -1, 2, -3) / 3.0;
    //Sort green-blue channels (highest to lowest)
    vec4 P = (c.b<c.g) ? vec4(c.gb, S.rg) : vec4(c.bg, S.wz);
    //Sort red-green-blue channels (highest to lowest)
    vec4 Q = (P.r<c.r) ? vec4(c.r, P.gbr) : vec4(P.rga, c.r);
    //Find the difference between the highest and lowest RGB for saturation
    float D = Q.x - min(Q.w, Q.y);
    //Put it all together
    return vec3(abs(Q.z + (Q.w - Q.y) / (6.0*D+E)), D / (Q.x+E), Q.x);
	}

vec3 rgb(float h, float s, float v)
	{
    //Compute RGB hue
    vec3 RGB = clamp(abs(mod(h*6.0+vec3(0,4,2), 6.0)-3.0)-1.0, 0.0, 1.0);
    //Multiply by value and mix for saturation
    return v * mix(vec3(1), RGB, s);
	}


void main()
	{
	float humidity_multiplier = terrain_humidity;
	float sunlight_multiplier = .25 + (terrain_sunlight / 2.);
		
		
	vec4 base_colour = v_vColour * texture2D(gm_BaseTexture, v_vTexcoord);
	float base_a = base_colour.w;
	vec3 base_rgb = base_colour.xyz;
	vec3 base_hsv = hsv(base_rgb);
	vec3 final_rgb = rgb(
		base_hsv.x, 
		base_hsv.y * humidity_multiplier,
		base_hsv.z * sunlight_multiplier
		);
		
	gl_FragColor = vec4(final_rgb, base_a);
    //gl_FragColor = v_vColour * texture2D( gm_BaseTexture, v_vTexcoord );
	}
