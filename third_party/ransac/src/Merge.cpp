#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdlib>

#include <PointCloud.h>

// For RANSAC
#include <RansacShapeDetector.h>
#include <PlanePrimitiveShapeConstructor.h>
#include <CylinderPrimitiveShapeConstructor.h>
#include <SpherePrimitiveShapeConstructor.h>
#include <ConePrimitiveShapeConstructor.h>
#include <TorusPrimitiveShapeConstructor.h>

#include <PlanePrimitiveShape.h> // for PlanePrimitiveShape
#include <Plane.h>
#include <SpherePrimitiveShape.h>
#include <Sphere.h>
#include <CylinderPrimitiveShape.h>
#include <Cylinder.h>
#include <TorusPrimitiveShape.h>
#include <Torus.h>
#include <ConePrimitiveShape.h>
#include <Cone.h>

#include <basic.h> // for Vec3f


typedef std::vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > ShapeVector;
typedef MiscLib::RefCountPtr<PrimitiveShape> Primitive;


std::vector<Primitive>
MergeCandidatePrimitives(const std::vector<std::vector<Primitive>>& all_candidates)
{
	std::vector<Primitive> all_primitives;
	for (const auto& candidates : all_candidates) {
		const int candidate_num = static_cast<int>(candidates.size());
		//PrimitiveShape new_prim = nullptr;
		int type = candidates[0]->Identifier();
		if (type == 0) {
			// Plane
			Vec3f normal(0.f, 0.f, 0.f);
			float offset = 0.f;

			// keep for reference
			PlanePrimitiveShape* pcref = (PlanePrimitiveShape*)candidates[0]->Clone();
			Plane plane_ref = pcref->Internal();
			Vec3f normal_ref = plane_ref.getNormal();
			float sgn_offset = plane_ref.SignedDistToOrigin() > 0.0 ? 1.0 : -1.0;

			for (const auto& c : candidates) {
				PlanePrimitiveShape* pc = (PlanePrimitiveShape*)c->Clone();
				Plane plane = pc->Internal();
				if (normal_ref.dot(plane.getNormal()) > 0) {
					normal += plane.getNormal();
				}
				else {
					normal += -plane.getNormal();
				}

				float sgn_dist = plane.SignedDistToOrigin() > 0.0 ? 1.0 : -1.0;
				if (sgn_dist*sgn_offset > 0.0) {
					offset += plane.SignedDistToOrigin();
				}
				else {
					offset += -plane.SignedDistToOrigin();
				}

			}

			normal /= candidate_num;
			offset /= candidate_num;
			Vec3f pt = offset * normal;
			Plane p(pt, normal);
			Primitive new_prim = MiscLib::RefCountPtr<PrimitiveShape>(new PlanePrimitiveShape(p));
			all_primitives.push_back(new_prim);
		}
		else if (type == 1) {
			// Sphere
			Vec3f center(0.f, 0.f, 0.f);
			float radius = 0.f;

			for (const auto& c : candidates) {
				SpherePrimitiveShape* sc = (SpherePrimitiveShape*)c->Clone();
				Sphere sphere = sc->Internal();
				center += sphere.Center();
				radius += sphere.Radius();
			}

			center /= candidate_num;
			radius /= candidate_num;
			Sphere s(center, radius);
			Primitive new_prim = MiscLib::RefCountPtr<PrimitiveShape>(new SpherePrimitiveShape(s));
			all_primitives.push_back(new_prim);
		}
		else if (type == 2) {
			// Cylinder
			Vec3f center(0.f, 0.f, 0.f);
			Vec3f dir(0.f, 0.f, 0.f);
			float radius = 0.0;

			for (const auto& c : candidates) {
				CylinderPrimitiveShape* cc = (CylinderPrimitiveShape*)c->Clone();
				Cylinder cylinder = cc->Internal();
				center += cylinder.AxisPosition();
				dir += cylinder.AxisDirection();
				radius += cylinder.Radius();
			}

			center /= candidate_num;
			dir /= candidate_num;
			radius /= candidate_num;
			Cylinder c(dir, center, radius);
			Primitive new_prim = MiscLib::RefCountPtr<PrimitiveShape>(new CylinderPrimitiveShape(c));
			all_primitives.push_back(new_prim);
		}
		else if (type == 3) {
			// Cone
			Vec3f axis(0.f, 0.f, 0.f);
			Vec3f center(0.f, 0.f, 0.f);
			float angle = 0.f;

			for (const auto& c : candidates) {
				ConePrimitiveShape* cc = (ConePrimitiveShape*)c->Clone();
				Cone cone = cc->Internal();
				center += cone.Center();
				axis += cone.AxisDirection();
				angle += cone.Angle();
			}

			center /= candidate_num;
			axis /= candidate_num;
			angle /= candidate_num;

			Cone c(center, axis, angle);
			Primitive new_prim = MiscLib::RefCountPtr<PrimitiveShape>(new ConePrimitiveShape(c));
			all_primitives.push_back(new_prim);
		}
		else if (type == 4) {
			// Torus
			Vec3f center(0.f, 0.f, 0.f);
			Vec3f axis(0.f, 0.f, 0.f);
			float major_radius = 0.0f;
			float minor_radius = 0.0f;

			for (const auto& c : candidates) {
				TorusPrimitiveShape* tc = (TorusPrimitiveShape*)c->Clone();
				Torus torus = tc->Internal();
				center += torus.Center();
				axis += torus.AxisDirection();
				major_radius += torus.MajorRadius();
				minor_radius += torus.MinorRadius();
			}

			center /= candidate_num;
			axis /= candidate_num;
			major_radius /= candidate_num;
			minor_radius /= candidate_num;

			Torus t(axis, center, minor_radius, major_radius);
			Primitive new_prim = MiscLib::RefCountPtr<PrimitiveShape>(new TorusPrimitiveShape(t));
			all_primitives.push_back(new_prim);
		}
		else {
			std::cout << "Merge primitive error: unrecognized primitive type" << std::endl;
		}
	}
	return all_primitives;
}


bool
ArePrimitivesClose(const Primitive& p, const Primitive& c,
	float DIST_THRESHOLD, float DOT_THRESHOLD, float ANGLE_THRESHOLD)
{
	if (p->Identifier() == 0) {
		// Planes
		PlanePrimitiveShape* pps = (PlanePrimitiveShape*)p->Clone();
		const Plane& pp = pps->Internal();

		PlanePrimitiveShape* cps = (PlanePrimitiveShape*)c->Clone();
		const Plane& cp = cps->Internal();

		Vec3f normalp = pp.getNormal();
		float distp = std::fabs(pp.SignedDistToOrigin());

		Vec3f normalc = cp.getNormal();
		float distc = std::fabs(cp.SignedDistToOrigin());

		if (std::fabs(distc - distp) > DIST_THRESHOLD) return false;

		float dot = normalp.dot(normalc);
		float abs_dot = std::fabs(dot);
		if (abs_dot < DOT_THRESHOLD) return false;

		return true;
	}

	if (p->Identifier() == 1) {
		// Sphere
		SpherePrimitiveShape* pps = (SpherePrimitiveShape*)p->Clone();
		const Sphere& ps = pps->Internal();

		SpherePrimitiveShape* cps = (SpherePrimitiveShape*)c->Clone();
		const Sphere& cs = cps->Internal();

		Vec3f centerp = ps.Center();
		float rp = ps.Radius();

		Vec3f centerc = cs.Center();
		float rc = cs.Radius();

		float dist = (centerp - centerc).length();
		if (dist > DIST_THRESHOLD) return false;

		if (std::fabs(rp - rc) > DIST_THRESHOLD) return false;

		return true;
	}

	if (p->Identifier() == 2) {
		// Cylinder
		CylinderPrimitiveShape* pps = (CylinderPrimitiveShape*)p->Clone();
		const Cylinder& pc = pps->Internal();

		CylinderPrimitiveShape* cps = (CylinderPrimitiveShape*)c->Clone();
		const Cylinder& cc = cps->Internal();

		float radiusp = pc.Radius();
		Vec3f axisp = pc.AxisDirection();
		Vec3f axisPosp = pc.AxisPosition();

		float radiusc = cc.Radius();
		Vec3f axisc = cc.AxisDirection();
		Vec3f axisPosc = cc.AxisPosition();

		if (std::fabs(radiusp - radiusc) > DIST_THRESHOLD) return false;

		if ((axisPosp - axisPosc).length() > DIST_THRESHOLD) {
			Vec3f axisPosCP = axisPosp - axisPosc;
			float axisPosCPlen = axisPosCP.length();
			axisPosCP /= axisPosCPlen;
			float dotPos = axisPosCP.dot(axisp);
			float abs_dotPos = std::fabs(dotPos);
			if (abs_dotPos < DOT_THRESHOLD) return false;
		}
		// otherwise axisPosp == axisPosc and we continue with the other tests

		float dot = axisp.dot(axisc);
		float abs_dot = std::fabs(dot);
		if (abs_dot < DOT_THRESHOLD) return false;

		return true;
	}

	if (p->Identifier() == 3) {
		// Cone
		ConePrimitiveShape* pps = (ConePrimitiveShape*)p->Clone();
		const Cone& pc = pps->Internal();

		ConePrimitiveShape* cps = (ConePrimitiveShape*)c->Clone();
		const Cone& cc = cps->Internal();

		float anglep = pc.Angle();
		Vec3f centerp = pc.Center();
		Vec3f axisp = pc.AxisDirection();

		float anglec = cc.Angle();
		Vec3f centerc = cc.Center();
		Vec3f axisc = cc.AxisDirection();

		if (std::fabs(anglep - anglec) > ANGLE_THRESHOLD) return false;
		if ((centerp - centerc).length() > DIST_THRESHOLD) return false;

		float dot = axisp.dot(axisc);
		float abs_dot = std::fabs(dot);
		if (abs_dot < DOT_THRESHOLD) return false;

		return true;
	}

	if (p->Identifier() == 4) {
		// Torus
		TorusPrimitiveShape* pps = (TorusPrimitiveShape*)p->Clone();
		const Torus& pt = pps->Internal();

		TorusPrimitiveShape* cps = (TorusPrimitiveShape*)c->Clone();
		const Torus& ct = cps->Internal();

		const Vec3f centerp = pt.Center();
		const Vec3f axisp = pt.AxisDirection();
		float mradiusp = pt.MinorRadius();
		float Mradiusp = pt.MajorRadius();

		const Vec3f centerc = ct.Center();
		const Vec3f axisc = ct.AxisDirection();
		float mradiusc = ct.MinorRadius();
		float Mradiusc = ct.MajorRadius();

		if ((centerp - centerc).length() > DIST_THRESHOLD) return false;

		float dot = axisp.dot(axisc);
		float abs_dot = std::fabs(dot);
		if (abs_dot < DOT_THRESHOLD) return false;

		if (std::fabs(mradiusp - mradiusc) > DIST_THRESHOLD) return false;
		if (std::fabs(Mradiusp - Mradiusc) > DIST_THRESHOLD) return false;

		return true;
	}

	// unrecognized primitive type
	std::cout << "ArePrimitivesClose: Unrecognized primitive type\n";
	return false;
}


// primitives : list of primitives obtained from ransac
std::vector<Primitive>
MergeSimilarPrimitives(std::vector<Primitive>& primitives,
	float dist_thresh, float dot_thresh, float angle_thresh)
{
	std::vector<std::vector<Primitive>> all_candidates(0);
	int prim_num = static_cast<int>(primitives.size());

	for (int i = 0; i < prim_num; ++i) {
		const Primitive& p = primitives[i];
		bool is_new = true;
		int cand_num = static_cast<int>(all_candidates.size());
		for (int j = 0; j < cand_num; ++j) {
			if (p->Identifier() != all_candidates[j][0]->Identifier()) continue;
			// See if p is close to all primitives in all_candidates[j].
			bool all_close = true;
			for (const auto& c : all_candidates[j]) {
				if (!ArePrimitivesClose(p, c, dist_thresh, dot_thresh, angle_thresh)) {
					all_close = false;
					break;
				}
			}
			if (all_close) {
				// p belongs to this cluster.
				is_new = false;
				all_candidates[j].push_back(p);
				break;
			}
		}
		if (is_new) {
			// Create a new cluster.
			all_candidates.push_back({ p });
		}
	}

	// Now merge non-singleton elements in all_candidates.
	std::vector<Primitive> all_primitives =
		MergeCandidatePrimitives(all_candidates);

	return all_primitives;
}


std::vector<Primitive>
MergeSimilarPrimitives(ShapeVector& primitives, float dist_thresh, float dot_thresh, float angle_thresh)
{
	std::vector<std::vector<Primitive>> all_candidates(0);
	int prim_num = static_cast<int>(primitives.size());

	for (int i = 0; i < prim_num; ++i) {
		const Primitive& p = primitives[i].first;
		bool is_new = true;
		int cand_num = static_cast<int>(all_candidates.size());
		for (int j = 0; j < cand_num; ++j) {
			if (p->Identifier() != all_candidates[j][0]->Identifier()) continue;
			// See if p is close to all primitives in all_candidates[j].
			bool all_close = true;
			for (const auto& c : all_candidates[j]) {
				if (!ArePrimitivesClose(p, c, dist_thresh, dot_thresh, angle_thresh)) {
					all_close = false;
					break;
				}
			}
			if (all_close) {
				// p belongs to this cluster.
				is_new = false;
				all_candidates[j].push_back(p);
				break;
			}
		}
		if (is_new) {
			// Create a new cluster.
			all_candidates.push_back({ p });
		}
	}

	// Now merge non-singleton elements in all_candidates.
	std::vector<Primitive> all_primitives = MergeCandidatePrimitives(all_candidates);

	return all_primitives;
}


void
MergeSimilarPrimitives(std::vector<Primitive>& primitives, std::vector<PointCloud>& pointClouds,
	float dist_thresh, float dot_thresh, float angle_thresh,
	std::vector<Primitive>& mergedPrimitives, std::vector<PointCloud>& mergedPointClouds)
{
	std::vector<std::vector<Primitive>> all_candidates(0);
	int prim_num = static_cast<int>(primitives.size());

	for (int i = 0; i < prim_num; ++i) {
		const Primitive& p = primitives[i];
		PointCloud pc = pointClouds[i];
		bool is_new = true;
		int cand_num = static_cast<int>(all_candidates.size());
		for (int j = 0; j < cand_num; ++j) {
			if (p->Identifier() != all_candidates[j][0]->Identifier()) continue;
			// See if p is close to all primitives in all_candidates[j].
			bool all_close = true;
			for (const auto& c : all_candidates[j]) {
				if (!ArePrimitivesClose(p, c, dist_thresh, dot_thresh, angle_thresh)) {
					all_close = false;
					break;
				}
			}
			if (all_close) {
				// p belongs to this cluster.
				is_new = false;
				all_candidates[j].push_back(p);
				mergedPointClouds[j] += pc;
				break;
			}
		}
		if (is_new) {
			// Create a new cluster.
			all_candidates.push_back({ p });
			mergedPointClouds.push_back(pc);
		}
	}

	mergedPrimitives = MergeCandidatePrimitives(all_candidates);
}


void SplitPointsPrimitives(const ShapeVector& shapes, const PointCloud& pc,
	std::vector<Primitive>& primitives, std::vector<PointCloud>& pointClouds)
{
	std::size_t sum = 0;

	for (std::size_t i = 0; i < shapes.size(); ++i) {
		unsigned int numPts = shapes[i].second;
		Point* pts = new Point[numPts];

		std::size_t k = 0;

		for (std::size_t j = pc.size() - (sum + shapes[i].second);
			j < pc.size() - sum; ++j) {

			pts[k++] = pc[j];
		}

		sum += shapes[i].second;

		pointClouds.push_back(PointCloud(pts, numPts));

		delete[] pts;

		primitives.push_back(shapes[i].first);
	}
}

