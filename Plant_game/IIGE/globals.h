#pragma once

#include <utils/logging/logger.h>

namespace iige::globals
	{
	using logger_t = utils::logging::logger<utils::logging::message<utils::logging::output_style_t::on_line>>;
	inline static logger_t logger;
	}