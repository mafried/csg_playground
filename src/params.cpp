#include "..\include\params.h"
#include "INIReader.h"
#include <iostream>

lmu::ParameterSet::ParameterSet(const std::string & path) : 
	_reader(std::make_unique<INIReader>(path))
{
	if (_reader->ParseError() < 0)
		throw std::runtime_error("Cannot read ini file from " + path + ".");
}

lmu::ParameterSet::~ParameterSet()
{
}

double lmu::ParameterSet::getDouble(const std::string & group, const std::string & value, double def) const
{
	return _reader->GetReal(group, value, def);
}

int lmu::ParameterSet::getInt(const std::string & group, const std::string & value, int def) const
{
	return _reader->GetInteger(group, value, def);
}

std::string lmu::ParameterSet::getStr(const std::string & group, const std::string & value, const std::string & def) const
{
	return _reader->Get(group, value, def);
}

bool lmu::ParameterSet::getBool(const std::string & group, const std::string & value, bool def) const
{
	return _reader->GetBoolean(group, value, def);
}

void lmu::ParameterSet::print() const
{
	const auto& values = _reader->Values();

	for (auto it = values.begin(); it != values.end(); ++it)
		std::cout << it->first << ": " << it->second << std::endl;	
}
