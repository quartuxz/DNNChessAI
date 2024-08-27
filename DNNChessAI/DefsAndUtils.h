#pragma once

#include <algorithm>
#include <chrono>

#define START_CHRONO auto start = std::chrono::high_resolution_clock::now();
#define END_CHRONO_LOG auto finish = std::chrono::high_resolution_clock::now();\
						std::cout << std::endl;\
						std::cout << "time taken in milliseconds: " <<std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << std::endl;
