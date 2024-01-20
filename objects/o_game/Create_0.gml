
initialized = true;

level_size = new (globals().utils.vec2)(80, 80);

terrain = new (globals().game.terrain.class)(level_size.x, level_size.y);
//plant   = new (globals().game.plant  .class)(level_size.x, level_size.y);