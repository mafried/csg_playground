#ifndef PARAMS_H
#define PARAMS_H

#include <string>
#include <memory>

class INIReader;

namespace lmu
{	
	struct ParameterSet
	{
		ParameterSet(const std::string& path);
		~ParameterSet();

		double getDouble(const std::string& group, const std::string& value, double def) const;
		int getInt(const std::string& group, const std::string& value, int def) const;
		std::string getStr(const std::string& group, const std::string& value, const std::string& def) const;
		bool getBool(const std::string& group, const std::string& value, bool def) const;

		void print() const;

	private: 
		std::unique_ptr<INIReader> _reader;
	};
}

#endif