#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::SizedCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#include <vcg/complex/complex.h>
# include "derivatives/apss.h"
# include "derivatives/meshmodel.hpp"
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export_ply.h>

#include<vcg/complex/complex.h>
#include<vcg/complex/algorithms/cell2.h>
//#include<vcg/complex/algorithms/ransac_matching3.h>
#include <algorithm>            // std::min, std::max

#include<vcg/complex/complex.h>
#include<wrap/io_trimesh/import.h>
#include<wrap/io_trimesh/export.h>
#include <vcg/complex/algorithms/clustering.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/space/index/kdtree/priorityqueue.h>


#include <assert.h>
#include <unordered_map> 
#include <set>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory> 
#include <atomic>
#include <deque>
#include <future>
//#include "decoder.hpp"
//#include<vcg/complex/algorithms/point_sampling_pi.h>
#include <vcg/space/index/kdtree/kdtree.h>


#include <vcg/complex/complex.h>
#include <vcg/simplex/face/component_ep.h>
#include <vcg/complex/algorithms/point_sampling.h>
#include <vcg/complex/algorithms/update/component_ep.h>
#include <vcg/complex/algorithms/update/normal.h>

// io
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export_ply.h>

#include <cstdlib>

#include <sys/timeb.h>
#include <iostream>
#include <string>

# include<Eigen/Dense>
using namespace std;
using namespace vcg;
using namespace GaelMls;
using namespace Eigen;


typedef CMeshO MeshType;
typedef typename MeshType::CoordType			CoordType;
typedef typename MeshType::BoxType				BoxType;
typedef typename MeshType::ScalarType			ScalarType;
typedef typename MeshType::VertexType			VertexType;
typedef typename MeshType::VertexPointer		VertexPointer;
typedef typename MeshType::VertexIterator		VertexIterator;
typedef typename MeshType::BoxType			Box3x;
typedef Matrix33<ScalarType>					MatrixType;
typedef Cell<MeshType>						CellType;
typedef typename MeshType::BoxType			Box3x;
typedef Matrix33<ScalarType>				MatrixType;
typedef Cell<MeshType>						CellType;
typedef CellType* CellPointer;
typedef  KdTree<ScalarType> KdTreeType;
typedef unordered_map<Point3i, std::vector<CoordType>, vcg::HashFunctor> CellContainerType;
typedef unordered_map<int , CellPointer> CellMotionVector;


void clusterMesh(MeshType& orgMesh, MeshType& sampledMesh,
	ScalarType samplingRadiusAbs, Box3x &_bbox,
	vcg::Point3i &grid_size, CellMotionVector& cmv)
{
	vcg::tri::Clustering<MeshType, vcg::tri::NearestToCenter<MeshType> > ClusteringGrid;
	ClusteringGrid.Init(_bbox, 100000, samplingRadiusAbs);
	ClusteringGrid.AddPointSet(orgMesh, false);
	ClusteringGrid.SelectPointSet(orgMesh);
	tri::UpdateSelection<MeshType>::FaceFromVertexLoose(orgMesh);
	ClusteringGrid.ExtractPointSet(sampledMesh);
	cout << endl << "BBox sent to function ";
	cout << endl << "max" << "x =" << _bbox.max.X() << "y =" << _bbox.max.Y() << "z =" << _bbox.max.Z();
	cout << endl << "min" << "x =" << _bbox.min.X() << "y =" << _bbox.min.Y() << "z =" << _bbox.min.Z();
	cout << endl << "BBox clustering grid ";
	Box3x clustBbox = ClusteringGrid.Grid.bbox;
	cout << endl << "max" << "x =" << clustBbox.max.X() << "y =" << clustBbox.max.Y() << "z =" << clustBbox.max.Z();
	cout << endl << "min" << "x =" << clustBbox.min.X() << "y =" << clustBbox.min.Y() << "z =" << clustBbox.min.Z();
	vcg::tri::UpdateBounding<MeshType>::Box(sampledMesh);
	grid_size = ClusteringGrid.Grid.siz;
	//tri::UpdateNormal<MeshType>::PerVertexNormalized(sampledMesh);
	//tri::Smooth<MeshType>::VertexNormalLaplacian(sampledMesh, 10);
	vcg::tri::Clustering<MeshType, vcg::tri::NearestToCenter<MeshType> > ClusteringGrid1;
	ClusteringGrid1.Init(orgMesh.bbox, 100000, samplingRadiusAbs);
	tri::io::ExporterPLY<MeshType>::Save(sampledMesh, "sampledMesh.ply");

	CoordType zero;
	zero.SetZero();
	MatrixType identity;
	identity.SetIdentity();

	unordered_map<Point3i, CellPointer, vcg::HashFunctor> cellMap;

	for (int i = 0; i < sampledMesh.vn; i++)
	{

		CoordType pt = sampledMesh.vert[i].cP();
		Point3i index;
		ClusteringGrid.Grid.PToIP(pt, index);
		CellPointer c = new CellType(i);
		c->P() = pt;
		c->p3Index = index;
		c->setCorrespondingPosition(zero);
		c->setTrans(zero);
		c->setMatrix(identity);
		cmv[i] = c;
		cellMap.insert(make_pair(index, c));
	}
	/*
	for (int i = 0; i < orgMesh.vn; i++)
	{
		
		CoordType pt = orgMesh.vert[i].cP();
		Point3i index;
		ClusteringGrid1.Grid.PToIP(pt, index);
		CellPointer c = cellMap.at(index);
		(*c).AddVertex(&(orgMesh.vert[i]));
	}*/


}


#define INNER_PRODUCT(x1, y1, z1, x2, y2, z2) (((x1)*(x2))+((y1)*(y2))+((z1)*(z2)))
#define SQUARE_NORM(x, y, z) (((x)*(x))+((y)*(y))+((z)*(z)))

// default column-major in eigen
struct RigidFunctor {
    RigidFunctor(double coeff)
        : _coeff(coeff) {

    }
    template <typename T>
    bool operator()(const T *const affi_m, T *residual) const {
        residual[0] = INNER_PRODUCT(affi_m[0], affi_m[1], affi_m[2], affi_m[3], affi_m[4], affi_m[5]);
        residual[1] = INNER_PRODUCT(affi_m[0], affi_m[1], affi_m[2], affi_m[6], affi_m[7], affi_m[8]);
        residual[2] = INNER_PRODUCT(affi_m[3], affi_m[4], affi_m[5], affi_m[6], affi_m[7], affi_m[8]);
        residual[3] = T(1) - SQUARE_NORM(affi_m[0], affi_m[1], affi_m[2]);
        residual[4] = T(1) - SQUARE_NORM(affi_m[3], affi_m[4], affi_m[5]);
        residual[5] = T(1) - SQUARE_NORM(affi_m[6], affi_m[7], affi_m[8]);

        for (size_t i = 0; i < 6; i ++) {
            residual[i] = _coeff * residual[i];
        }

        return true;
    }

private:
    double _coeff;
};
// default column-major in eigen

struct SmoothFunctor {
	SmoothFunctor(double coeff, double samplingRad, CoordType master, CoordType slave)
		: _coeff(coeff), _master(master), _slave(slave),_samplingRad(samplingRad) {
	}

	template <typename T>
	bool operator()(const T *const affi_m, const T *const trans_m, const T *const trans_s, T *residual) const {
		CoordType delta_2 = _slave - _master;

		residual[0] = INNER_PRODUCT((affi_m[0] - T(1)), affi_m[3], affi_m[6],
			delta_2.X(), delta_2.Y(), delta_2.Z()) + trans_m[0] - trans_s[0];
		residual[1] = INNER_PRODUCT(affi_m[1], (affi_m[4] - T(1)), affi_m[7],
			delta_2.X(), delta_2.Y(), delta_2.Z()) + trans_m[1] - trans_s[1];
		residual[2] = INNER_PRODUCT(affi_m[2], affi_m[5], (affi_m[8] - T(1)),
			delta_2.X(), delta_2.Y(), delta_2.Z()) + trans_m[2] - trans_s[2];

		for (size_t i = 0; i < 3; i++) {
			residual[i] = _coeff * residual[i];
		}

		return true;
	}

private:
	double _coeff,_samplingRad;
	CoordType _master;
	CoordType _slave;
};


class FitFunctor:public SizedCostFunction<3,3,9,3> 
	{

	public:

	FitFunctor(double coeff, CoordType point, APSS<MeshType>* apss,KdTreeType* kdtree,double samplingRad) : _coeff(coeff), 
								_apss(apss),_point(point),_kdtree(kdtree), _samplingRad(samplingRad)
	{
	}

	virtual bool Evaluate(double const* const* params,
		double* residuals,
		double** jacobians) const {

		residuals[0] = ((double)rand() / (RAND_MAX)) + 1;
		residuals[1] = ((double)rand() / (RAND_MAX)) + 1;
		residuals[2] = ((double)rand() / (RAND_MAX)) + 1;

		double corresX = params[0][0];
		double corresY = params[0][1];
		double corresZ = params[0][2];
		Point3d corres(corresX, corresY, corresZ);
		Matrix33d aff;
		double a[3][3];
		a[0][0] = params[1][0];
		a[0][1] = params[1][1];
		a[0][2] = params[1][2];
		a[1][0] = params[1][3];
		a[1][1] = params[1][4];
		a[1][2] = params[1][5];
		a[2][0] = params[1][6];
		a[2][1] = params[1][7];
		a[2][2] = params[1][8];

		aff.SetRow(0, &(a[0][0]));
		aff.SetRow(1, &(a[1][0]));
		aff.SetRow(2, &(a[2][0]));

		double b[3];
		b[0] = params[2][0];
		b[1] = params[2][1];
		b[2] = params[2][2];

		Point3d tB(b[0], b[1], b[2]);

		Point3d res = (aff * _point) + tB - corres;
		//cout << endl << "res.X() =" << res.X() << " res.Y() = " << res.Y() << "res.Z()" << res.Z();
		residuals[0] = res.X();
		residuals[1] = res.Y();
		residuals[2] = res.Z();


		if (!jacobians ) return true;
		if (jacobians != NULL)
		{

			// initialize jacobians
			// Corresponding point
			int noParameters = 3;
			int noResiduals = 3;
			for (int i = 0; i < noParameters; i++)
			{
				int noElements = 0;
				switch (i)
				{
				case 0:
					noElements = 3;
					break;
				case 1:
					noElements = 9;
					break;
				case 2:
					noElements = 3;
					break;
				}

				if (jacobians[i] == NULL)
					continue;

				double* jacob = jacobians[i];
				for (int j = 0; j < noElements*noResiduals; j++)
					jacob[j] = 0;
			}

			double* jacobian0 = jacobians[0];
			if (jacobian0 != NULL)
			{
				APSS<MeshType>& ap = *_apss;

				int mask;
				Point3d proj = ap.project(_point, NULL, &mask);
				mask = 0;
				Point3d grad = ap.gradient(proj, &mask);

				/*if ((mask == MLS_TOO_FAR) || (mask == MLS_TOO_MANY_ITERS))
				{
					jacobian0[0] = 0;
					jacobian0[1] = 0;
					jacobian0[2] = 0;
				}*/
				//Point3d pr = apssM.project(m2.vert[i].P());
				if (isnan(grad.X()) || isnan(grad.Y()) || isnan(grad.Z()) || isinf(grad.X()) || isinf(grad.Y()) || isinf(grad.Z()))
				{
					jacobian0[0] = 0;
					jacobian0[4] = 0;
					jacobian0[8] = 0;
				}

				else if (!((mask == MLS_TOO_FAR) || (mask == MLS_TOO_MANY_ITERS)))
				{
					jacobian0[0] = grad.X();
					jacobian0[4] = grad.Y();
					jacobian0[8] = grad.Z();
				}
			}

			double* jacobian1 = jacobians[1];

			if (jacobian1 != NULL)
			{
				jacobian1[0] = a[0][0];
				jacobian1[1] = a[0][1];
				jacobian1[2] = a[0][2];
				jacobian1[12] = a[1][0];
				jacobian1[13] = a[1][1];
				jacobian1[14] = a[1][2];
				jacobian1[24] = a[2][0];
				jacobian1[25] = a[2][1];
				jacobian1[26] = a[2][2];
			}

			double* jacobian2 = jacobians[2];

			if (jacobian2 != NULL)
			{
				jacobian2[0] = 1;
				jacobian2[4] = 1;
				jacobian2[8] = 1;
			}
		}
			return true;
	}
	private:

		double _coeff, _samplingRad;
		CoordType _point;
		APSS<MeshType>* _apss;
		KdTreeType* _kdtree;

};


void solver(CellMotionVector& cmv, MeshType& orgMesh, MeshType& targetMesh,
				ScalarType samplingRadius)
{
	cout << endl << "Inside solver";
	targetMesh.vert.EnableRadius();
	APSS<MeshType> ap(targetMesh);
	ap.setFilterScale(4);

	MeshType sampledMesh;
	Box3x  _bbox; vcg::Point3i grid_size;
	clusterMesh(orgMesh, sampledMesh, samplingRadius, _bbox,grid_size, cmv);

	MeshType morphedMesh;
	tri::Append<MeshType, MeshType>::MeshCopy(morphedMesh, sampledMesh);
	int noIter = 700;
	double coeffRigid = 100;
	double coeffFit = 0.1;
	double coeffSmooth = 100;

	VertexConstDataWrapper<MeshType> vc(targetMesh);
	KdTreeType kdTarget(vc);

	for (int i = 0; i < cmv.size(); i++)
	{
		CellType& curCell = *(cmv[i]);
		CoordType pt = curCell.cP();;
		unsigned int closestInd;
		ScalarType dist;
		kdTarget.doQueryClosest(pt, closestInd, dist);
		Point3d ptz = targetMesh.vert[closestInd].cP();
		//cout << endl << ptz.X() << ptz.Y() << ptz.Z();
		//curCell.correspondingPos = targetMesh.vert[closestInd].cP();
		curCell.setCorrespondingPosition(targetMesh.vert[closestInd].cP());
		double* ca = curCell.getCorrespondingPositionArray();
		//cout << endl << ca[0] << ca[1] << ca[2];
	}


	for (int j = 0; j < noIter; j++)
	{
		cout << endl << "iter " << j;
		VertexConstDataWrapper<MeshType> vc(morphedMesh);
		KdTreeType kd(vc);
		int noNeighbors = 5;

		Problem problem;

		for (int i = 0; i < morphedMesh.vn; i++)
		{
			
			CellType& curCell = *(cmv[i]);
			KdTreeType::PriorityQueue queue;
			CoordType curPoint = morphedMesh.vert[i].cP();
			kd.doQueryK(curPoint, noNeighbors, queue);
			int neighbours = queue.getNofElements();

			ScalarType * affMatrixArray = curCell.getMatrixArray();
			ScalarType* tArray = curCell.getTranslateArray();
			ScalarType* cPosArray = curCell.getCorrespondingPositionArray();

			CostFunction* rigidFunctor = new AutoDiffCostFunction<RigidFunctor, 6, 9>(new RigidFunctor(coeffRigid));
			problem.AddResidualBlock(rigidFunctor, NULL, affMatrixArray);

			// Set up the only cost function (also known as residual). This uses
			// auto-differentiation to obtain the derivative (jacobian).
			//cout << endl << curPoint.X() << curPoint.Y() << curPoint.Z();
			CostFunction* fit_function = new FitFunctor(coeffFit, curPoint, &ap,&kdTarget, samplingRadius);
			//ScalarType* (params[5]);
			ScalarType* p[4];
			p[0] = cPosArray;
			vector<ScalarType*> params;
			//params.data();
			params.resize(5);
			//params = new 
			p[0] = cPosArray;
			p[1] = &(affMatrixArray[0]);
			p[2] = &(affMatrixArray[3]);
			p[3] = &(affMatrixArray[6]);
			p[4] = tArray;

			problem.AddResidualBlock(fit_function, NULL, cPosArray, affMatrixArray,tArray);
			for (int k = 0; k < neighbours; k++)
			{
				int neightId = queue.getIndex(k);
				if (neightId == i)
					continue;

				CellType& neighborCell = *(cmv[neightId]);

				//ScalarType** affMatArr = neighborCell.getMatrixArray();
				ScalarType* tArrNeigh = neighborCell.getTranslateArray();
				//ScalarType* cPos = neighborCell.getCorrespondingPositionArray();

				CoordType neighPoint = morphedMesh.vert[neightId].cP();

				CostFunction* smooth_function = new AutoDiffCostFunction<SmoothFunctor, 3, 9,3,3>(new SmoothFunctor(
												coeffSmooth, samplingRadius, curPoint, neighPoint));
				problem.AddResidualBlock(smooth_function, NULL, affMatrixArray, tArray, tArrNeigh);

			}
		}

		// Run the solver!
		Solver::Options options;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = true;
		options.max_num_iterations = 1000;
		options.minimizer_progress_to_stdout = true;
		Solver::Summary summary;
		Solve(options, &problem, &summary);
		if (abs(summary.fixed_cost - summary.final_cost) / (1 + summary.fixed_cost) < 1.0e-5)
		{
			coeffSmooth = coeffSmooth / 1.01;
			coeffRigid = coeffRigid / 1.01;
		}

		if (abs(summary.fixed_cost - summary.final_cost) / (1 + summary.fixed_cost) < 1.0e-6)
		{
			break;
		}

		//std::cout << summary.BriefReport() << "\n";

		// Update positions and save mesh
		for (int l = 0; l < morphedMesh.vn; l++)
		{
			CellType& curCell = *(cmv[l]);
			curCell.update();
			morphedMesh.vert[l].P() = (curCell.affMat * morphedMesh.vert[l].cP()) + (curCell.trans);
			curCell.clear();
			curCell.cP() = morphedMesh.vert[l].P();
		}

		MeshType morphMeshTemp;
		tri::Append<MeshType, MeshType>::MeshCopy(morphMeshTemp, morphedMesh);
		APSS<MeshType> morphAP(morphMeshTemp);
		morphAP.setFilterScale(8);
		for (int l = 0; l < morphedMesh.vn; l++)
		{
			int error;
			morphedMesh.vert[l].P() = morphAP.project(morphedMesh.vert[l].P(), NULL, &error);
		}

		tri::io::ExporterPLY<MeshType>::Save(morphedMesh, (to_string(j) + "morphedMesh.ply").c_str() );
	}
}

int main()
{

	MeshType targetMesh, orgMesh;
	vcg::tri::io::ImporterPLY<MeshType>::Open(orgMesh, "32xp.ply");
	vcg::tri::io::ImporterPLY<MeshType>::Open(targetMesh, "18xp.ply");

	vcg::tri::UpdateBounding<MeshType>::Box(orgMesh);
	vcg::tri::UpdateBounding<MeshType>::Box(targetMesh);

	ScalarType samplingRadius = orgMesh.bbox.Diag() *1 / 100;
	cout << endl << "samplingRadius = " << samplingRadius;

	CellMotionVector cmv;
	solver(cmv, orgMesh, targetMesh, samplingRadius);

}
