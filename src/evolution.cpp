#include "evolution.h"

#define _USE_MATH_DEFINES
#include <math.h>

lmu::ScheduleType lmu::scheduleTypeFromString(std::string scheduleType)
{
	std::transform(scheduleType.begin(), scheduleType.end(), scheduleType.begin(), ::tolower);

	if (scheduleType == "log")
		return ScheduleType::LOG;
	else if (scheduleType == "exp")
		return ScheduleType::EXP;

	return ScheduleType::IDENTITY;
}

