#include "..\include\collision.h"
#include "..\include\mesh.h"

bool lmu::collides(const lmu::ImplicitFunction & f1, const lmu::ImplicitFunction & f2)
{
	if (f1.type() == ImplicitFunctionType::Sphere && f2.type() == ImplicitFunctionType::Sphere)
		return collides(static_cast<const lmu::IFSphere&>(f1), static_cast<const lmu::IFSphere&>(f2));
	else
		throw std::runtime_error("Implicit function combination is not supported.");

	return false;
}

bool lmu::collides(const lmu::IFSphere & f1, const lmu::IFSphere & f2)
{
	return (f1.pos() - f2.pos()).squaredNorm() < (f1.radius() + f2.radius())*(f1.radius() + f2.radius());
}
