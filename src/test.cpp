#include "..\include\test.h"

void lmu::report(const char *msg, const char *file, int line)
{
	std::cout << "  Assertion " << msg << " in file " << file << " at line " << line << " failed." << std::endl;
}

