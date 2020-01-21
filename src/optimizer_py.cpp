#include "optimizer_py.h"
#include "csgnode_helper.h"

#include <algorithm>

using namespace lmu;

using PrimitiveLookup = std::unordered_map<std::string, lmu::ImplicitFunctionPtr>;

bool is_token(const std::string& str, const std::string& token_str, int pos)
{
	return str.substr(pos, token_str.size()) == token_str;
}

bool is_literal_token(const std::string& str, int& pos, std::string& literal_str)
{
	if (str[pos] != '\'') return false;

	int start_pos = pos;

	while (++pos < str.size() && str[pos] != '\'');

	if (pos == str.size()) return false;

	literal_str = str.substr(start_pos + 1, pos - start_pos - 1);
	pos++;

	return true;
}

lmu::TokenizeResult lmu::tokenize_py_string(const std::string& s)
{
	std::string str = s;
	str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());

	int pos = 0;
	std::vector<Token> tokens;

	while (pos < str.size())
	{
		if		(is_token(str, "(", pos)) { pos++; tokens.push_back(Token(TokenType::LEFT)); }

		else if (is_token(str, ")", pos)) { pos++; tokens.push_back(Token(TokenType::RIGHT)); }

		else if (is_token(str, ",", pos)) { pos++; tokens.push_back(Token(TokenType::COMMA)); }

		else if (is_token(str, "Or", pos)) { pos += 2; tokens.push_back(Token(TokenType::OR)); }

		else if (is_token(str, "And", pos)) { pos += 3; tokens.push_back(Token(TokenType::AND)); }

		else if (is_token(str, "Not", pos)) { pos += 3; tokens.push_back(Token(TokenType::NOT)); }

		else if (is_token(str, "Symbol", pos)) { pos += 6; tokens.push_back(Token(TokenType::SYMBOL)); }

		else
		{
			std::string literal_str;
			if (is_literal_token(str, pos, literal_str))
			{
				tokens.push_back(Token(TokenType::LITERAL, literal_str));
			}
			else
			{
				//ERROR
				return TokenizeResult(tokens, pos);
			}
		}

	}

	return TokenizeResult(tokens);
}

PrimitiveLookup create_prim_lookup(const std::vector<lmu::ImplicitFunctionPtr>& primitives)
{
	PrimitiveLookup lookup;

	for (const auto& prim : primitives)
		lookup[prim->name()] = prim;

	return lookup;
}

struct TokenProcessResult
{
	TokenProcessResult(const std::vector<Token> tokens, bool success, int pos) :
		tokens(tokens),
		error_pos(success ? -1 : pos)
	{
	}
	
	bool error_occurred()
	{
		return error_pos != -1;
	}

	int error_pos;
	std::vector<Token> tokens;
};

bool is_avail(int tp, const std::vector<Token>& t)
{
	return tp < t.size();
}

bool is(const Token& token, TokenType type)
{
	return token.type == type;
}

bool is_op(const Token& token)
{
	return token.type == TokenType::OR || token.type == TokenType::AND || token.type == TokenType::NOT;
}

lmu::CSGNode create_op(TokenType type)
{
	switch (type)
	{
	case TokenType::AND:
		return lmu::opInter();
	case TokenType::OR:
		return lmu::opUnion();
	case TokenType::NOT:
		return lmu::opComp();
	default:
		return lmu::opNo();
	}
}

lmu::CSGNode create_prim(const std::string& prim_name, const PrimitiveLookup& prim_lookup)
{
	auto prim_it = prim_lookup.find(prim_name);
	return prim_it != prim_lookup.end() ? lmu::geometry(prim_it->second) : lmu::opNo();
}

bool process_single(int& tp, const std::vector<Token>& t, TokenType type)
{
	if (is_avail(tp, t) && is(t[tp], type))
	{
		tp++;
		return true;
	}
	return false;
}

bool process_primitive(int& tp, const std::vector<Token>& t, const PrimitiveLookup& prim_lookup, lmu::CSGNode& node)
{
	if (is_avail(tp, t) && is(t[tp], TokenType::SYMBOL))
	{
		tp++;
		bool success = process_single(tp, t, TokenType::LEFT) && 
			process_single(tp, t, TokenType::LITERAL) && 
			process_single(tp, t, TokenType::RIGHT);

		if (success)
		{
			node.childsRef().push_back(create_prim(t[tp - 2 /*literal token index*/].value, prim_lookup));
			return true;
		}
	}
	return false;
}

bool process_op(int& tp, const std::vector<Token>& t, const PrimitiveLookup& pl, lmu::CSGNode& n)
{
	if (is_avail(tp, t) && is_op(t[tp]))
	{
		n.childsRef().push_back(create_op(t[tp].type));
		auto& new_n = n.childsRef().back();
		tp++;

		bool left = process_single(tp, t, TokenType::LEFT);
		bool first_arg = (process_op(tp, t, pl, new_n) || process_primitive(tp, t, pl, new_n));
		
		bool other_args = true;
		while (process_single(tp, t, TokenType::COMMA))
		{
			other_args = (process_op(tp, t, pl, new_n) || process_primitive(tp, t, pl, new_n));
		}
		
		bool right = process_single(tp, t, TokenType::RIGHT);

		return left && first_arg && other_args && right;
	}
	return false;
}

TokenProcessResult process_tokens(const std::vector<Token>& tokens, const PrimitiveLookup& pl, lmu::CSGNode& n)
{
	int token_pos = 0;

	bool success = process_primitive(token_pos, tokens, pl, n) || process_op(token_pos, tokens, pl, n);

	return TokenProcessResult(tokens, success, token_pos);
}

lmu::CSGNode lmu::parse_py_string(const std::string& str, const std::vector<ImplicitFunctionPtr>& primitives)
{		
	auto tokenize_result = tokenize_py_string(str);
	if (tokenize_result.error_occurred())	
		throw CSGNodeParseException("Tokenizer error", tokenize_result.error_pos);
	
	auto primitive_lookup = create_prim_lookup(primitives);

	CSGNode node = opNo();

	auto process_result = process_tokens(tokenize_result.tokens, primitive_lookup, node);
	if (process_result.error_occurred())
		throw CSGNodeParseException("Token processing error.", process_result.error_pos);

	return node.childsCRef()[0];
}

std::string lmu::espresso_expression(const CSGNode& n)
{
	if (n.type() == CSGNodeType::Geometry)
		return n.name();

	std::stringstream ss;
	
	std::string op;
	std::string prefix;
	switch (n.operationType())
	{
	case CSGNodeOperationType::Intersection:
		op = "&";
		break;
	case CSGNodeOperationType::Union:
		op = "|";
		break;
	case CSGNodeOperationType::Complement:
		prefix = "~";
		break;
	case CSGNodeOperationType::Difference:
		op = "& ~";
		break;
	}

	ss << prefix;
	ss << "(";
	for (int i = 0; i < n.childsCRef().size(); ++i)
	{
		ss << espresso_expression(n.childsCRef()[i]);
		if (i < n.childsCRef().size() - 1)
			ss << " " << op << " ";
	}
	ss << ")";

	return ss.str();
}

#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

CSGNode lmu::optimize_with_python(const CSGNode & node, SimplifierMethod method, const lmu::PythonInterpreter& py_interpreter)
{
	std::string python_input_expr = espresso_expression(node);

	std::string python_output_expr = py_interpreter.simplify(python_input_expr, method);

	/*PyObject *pName, *pModule, *pDict, *pFunc, *pValue, *presult;

	// Initialize the Python Interpreter
	Py_Initialize();
	
	// Build the name object
	pName = PyUnicode_FromString((char*)"simplifier");

	PyRun_SimpleString("import sys\nsys.path.append('C:/Users/friedrich/PycharmProjects/dnf_minimizer')");

	// Load the module object
	pModule = PyImport_Import(pName);

	std::cout << "Module:" << (pModule == nullptr) << std::endl;

	// pDict is a borrowed reference 
	pDict = PyModule_GetDict(pModule);
	
	// pFunc is also a borrowed reference 
	pFunc = PyDict_GetItemString(pDict, (char*)"simplify");

	if (PyCallable_Check(pFunc))
	{
		pValue = Py_BuildValue("(z, z)", (char*)"(s5 | (s1 & ~(s3) & ~(s4)) | (~(s3) & ~(s4) & s2))", (char*)"espresso");
		PyErr_Print();
		presult = PyObject_CallObject(pFunc, pValue);
		PyErr_Print();
	}
	else
	{
		PyErr_Print();
	}

	std::string ret = PyUnicode_AsUTF8(presult);

	std::cout << "RET: " << ret << std::endl;

	Py_DECREF(pValue);

	// Clean up
	Py_DECREF(pModule);
	Py_DECREF(pName);

	// Finish the Python Interpreter
	Py_Finalize();*/

	try
	{
		return parse_py_string(python_output_expr, allDistinctFunctions(node));
	}
	catch(const CSGNodeParseException& ex)
	{
		std::cerr << "Could not parse python expression. Reason: " << ex.msg << " Index: " << ex.error_pos << std::endl;
		return opNo();
	}
}

lmu::PythonInterpreter::PythonInterpreter(const std::string & simplifier_module_path)
{	
	Py_Initialize();

	simp_method_name = PyUnicode_FromString((char*)"simplifier");
	set_cover_method_name = PyUnicode_FromString((char*)"set_cover");

	PyRun_SimpleString(("import sys\nsys.path.append('" + simplifier_module_path + "')").c_str());

	simp_module = PyImport_Import(simp_method_name);
	simp_dict = PyModule_GetDict(simp_module);
	simp_method = PyDict_GetItemString(simp_dict, (char*)"simplify");

	set_cover_module = PyImport_Import(set_cover_method_name);
	set_cover_dict = PyModule_GetDict(set_cover_module);
	set_cover_method = PyDict_GetItemString(set_cover_dict, (char*)"solve_specialized_set_cover_str");

}

lmu::PythonInterpreter::~PythonInterpreter()
{
	Py_DECREF(simp_module);
	Py_DECREF(simp_method_name);
	Py_DECREF(set_cover_module);
	Py_DECREF(set_cover_method_name);

	Py_Finalize();
}

std::string lmu::PythonInterpreter::simplify(const std::string & expression, SimplifierMethod method) const 
{
	PyObject *arg, *res;

	std::string method_str;
	switch (method)
	{
	case SimplifierMethod::ESPRESSO:
		method_str = "espresso";
		break;
	case SimplifierMethod::SIMPY_TO_DNF:
		method_str = "sympy_todnf";
		break;
	case SimplifierMethod::SIMPY_SIMPLIFY_LOGIC:
		method_str = "sympy_symplifylogic";
	}

	if (PyCallable_Check(simp_method))
	{
		arg = Py_BuildValue("(z, z)", (char*)expression.c_str(), (char*)method_str.c_str());
		PyErr_Print();
		res = PyObject_CallObject(simp_method, arg);
		PyErr_Print();
	}
	else
	{
		PyErr_Print();
	}

	std::string res_str = PyUnicode_AsUTF8(res);

	Py_DECREF(res);

	return res_str;
}

#include <boost/algorithm/string.hpp>

std::vector<std::unordered_set<int>> from_str(const std::string& str)
{
	std::vector<std::unordered_set<int>> res;
	
	std::vector<std::string> sets_str;
	boost::algorithm::split(sets_str, str, boost::is_any_of("|"));

	for (const auto& set_str : sets_str)
	{
		std::vector<std::string> elements_str;
		std::unordered_set<int> set;
		boost::algorithm::split(elements_str, set_str, boost::is_any_of(" "));
		for (const auto& element_str : elements_str)
		{
			if (element_str.empty())
				continue;
			try
			{
				set.insert(std::stoi(element_str));
			}
			catch (const std::invalid_argument& ex)
			{
				std::cout << ex.what() << std::endl;
			}
		}
		res.push_back(set);
	}
	return res;
}

std::string to_str(const std::vector<std::unordered_set<int>>& sets)
{
	std::stringstream ss; 
	for (const auto& set : sets)
	{
		for (const auto& element : set)
		{
			ss << element << " ";
		}
		ss << "|";
	}

	auto res = ss.str();
	res.pop_back();
	return res;
}

std::vector<std::unordered_set<int>> lmu::PythonInterpreter::set_cover(const std::vector<std::unordered_set<int>>& sets, const std::unordered_set<int>& set_to_cover) const
{
	PyObject *arg, *res;

	auto sets_str = to_str(sets);
	auto set_to_cover_str = to_str({ set_to_cover });

	if (PyCallable_Check(set_cover_method))
	{
		arg = Py_BuildValue("(z, z)", (char*)sets_str.c_str(), (char*)set_to_cover_str.c_str());
		PyErr_Print();
		res = PyObject_CallObject(set_cover_method, arg);
		PyErr_Print();
	}
	else
	{
		PyErr_Print();
	}

	std::string res_str = PyUnicode_AsUTF8(res);

	Py_DECREF(res);
	
	return from_str(res_str);
}
