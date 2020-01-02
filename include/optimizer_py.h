#ifndef OPTIMIZER_PY_H
#define OPTIMIZER_PY_H

#include "csgnode.h"

typedef struct _object PyObject;

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

	enum class SimplifierMethod
	{
		ESPRESSO,
		SIMPY_SIMPLIFY_LOGIC,
		SIMPY_TO_DNF
	};

	struct PythonInterpreter
	{
		PythonInterpreter(const std::string& simplifier_module_path);
		~PythonInterpreter();

		std::string simplify(const std::string& expression, SimplifierMethod method) const;

	private:
		PyObject *simp_method_name, *simp_module, *simp_dict, *simp_method;

	};

	std::string espresso_expression(const CSGNode& n);

	CSGNode optimize_with_python(const CSGNode& node, SimplifierMethod method, const lmu::PythonInterpreter& py_interpreter);
}

#endif
