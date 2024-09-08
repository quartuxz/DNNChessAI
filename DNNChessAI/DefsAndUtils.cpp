#include "DefsAndUtils.h"
std::default_random_engine globalRandEngine;


std::default_random_engine* getGlobalRandomEngine()
{
	return &globalRandEngine;
}

void seedRandEngine()
{
	globalRandEngine.seed(time(NULL));
}
