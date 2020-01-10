#ifndef TEST_H
#define TEST_H

#include <iostream>

#include <type_traits>

namespace lmu
{
	void report(const char *msg, const char *file, int line);

	template <typename T, typename S>
	typename std::enable_if<!std::is_pointer<T>::value, void>::type reportEqError(const char* v1s, const char* v2s, const T& v1, const S& v2, const char *file, int line)
	{
		std::cout << "    Equality assertion in file " << file << " at line " << line << " failed. " << v1s << " which is " << v1  << " should be: " << v2s << " which is "  << v2 << std::endl;
	}
	template <typename T, typename S>
	typename std::enable_if<std::is_pointer<T>::value, void>::type reportEqError(const char* v1s, const char* v2s, const T& v1, const S& v2, const char *file, int line)
	{
		std::cout << "    Equality assertion in file " << file << " at line " << line << " failed. " << v1s << " should be " << v2s << std::endl;
	}

	template<typename T, typename S>
	bool testEq(const T& v1, const S& v2)
	{
		return v1 == v2;
	}

#define ASSERT_TRUE(EX) (void)((EX) || (lmu::report (#EX, __FILE__, __LINE__),0))
#define ASSERT_FALSE(EX) (void)(!(EX) || (lmu::report (#EX, __FILE__, __LINE__),0))

#define ASSERT_EQ(V1, V2) (void)(lmu::testEq(V1,V2) || (lmu::reportEqError (#V1, #V2, V1, V2, __FILE__, __LINE__),0))

#define TEST(TestCase) void test_ ## TestCase ## ()

#define RUN_TEST(TestCase) do { std::cout << "Start Test " << #TestCase << std::endl;  test_ ## TestCase ## (); std::cout << "End Test " << #TestCase << std::endl; } while(0)

}

#endif