#include "RefCount.h"
#ifdef _DEBUG
#include <assert.h>
#endif
#include <iostream>

using namespace MiscLib;

RefCount::~RefCount()
{
#ifdef _DEBUG
	assert(m_refCount == 0);
#endif
}
