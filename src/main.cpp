//#define TEST

#ifdef TEST 

#include "optimizer_test.h"

int main()
{
	RUN_TEST(OptimizerRedundancyTest);

	return 0; 
}

#else 

#include <string>
#include <iostream>

#include "optimizer_pipeline_runner.h"

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		std::cerr << "Wrong number of arguments." << std::endl;
		return 1;
	}
		
	try
	{
		auto input_config = std::string(argv[1]);
		auto output_folder = std::string(argv[2]);

		lmu::PipelineRunner runner(input_config, output_folder);
		return runner.run();
	}
	catch (const std::exception& ex)
	{
		std::cerr << "An Error occurred: " << ex.what() << std::endl;
		return 1;
	}
}

#endif 