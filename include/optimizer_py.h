#ifndef OPTIMIZER_PY_H
#define OPTIMIZER_PY_H

#include "csgnode.h"

namespace lmu 
{
	enum class TokenType
	{
		OR,
		AND,
		NOT,
		SYMBOL,
		LITERAL,
		COMMA,
		LEFT,
		RIGHT,
		UNKNOWN
	};

	struct Token
	{
		Token(TokenType type) :
			type(type)
		{
		}

		Token(TokenType type, const std::string& value) :
			type(type),
			value(value)
		{
		}

		TokenType type;
		std::string value;
	};


	struct CSGNodeParseException : std::runtime_error
	{
		CSGNodeParseException(const std::string& msg, int error_pos) :
			std::runtime_error("Unable to parse CSG tree."),
			msg(msg),
			error_pos(error_pos)
		{
		}

		std::string msg;
		int error_pos;
	};

	struct TokenizeResult
	{
		TokenizeResult(const std::vector<Token>& tokens, int error_pos) :
			error_pos(error_pos),
			tokens(tokens)
		{
		}

		TokenizeResult(const std::vector<Token>& tokens) :
			error_pos(-1),
			tokens(tokens)
		{
		}

		bool error_occurred()
		{
			return error_pos != -1;
		}

		int error_pos;
		std::vector<Token> tokens;
	};

	TokenizeResult tokenize_py_string(const std::string& str);

	CSGNode parse_py_string(const std::string& str, const std::vector<ImplicitFunctionPtr>& primitives);
}

#endif
