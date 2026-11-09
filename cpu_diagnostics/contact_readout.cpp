// CPU-only, fixed diagnostic of a floating load supported by two driven ankles.
// Not the G1, a sitting trajectory, or an alternative physical success scorer.
// No renderer, CUDA context, parameter search, or production configuration edits.
// Source premise: PhysX 5.9.1 DyTGSDynamics.cpp and forwardDynamic2.cu solve
// velocity constraints after pose integration. The diagnostic hook copies the
// exact joint velocities used for each position integration, without changing
// solver state. Only presence of the static floor differs between the cases.
#include "PxPhysicsAPI.h"
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace physx;
constexpr float dt = .005f;
constexpr unsigned steps = 2000;
constexpr unsigned positionIterations = 8, velocityIterations = 4;
std::array<std::array<float,4>,positionIterations> positionVelocity;
unsigned positionCalls=0;
bool hookError=false;

extern "C" void rest_diagnostic_position_velocity(unsigned count, const float* values) {
    if (count!=4 || positionCalls>=positionIterations) {hookError=true;return;}
    for (unsigned j=0;j<4;++j) positionVelocity[positionCalls][j]=values[j];
    ++positionCalls;
}

struct Errors : PxErrorCallback {
    unsigned failures = 0;
    void reportError(PxErrorCode::Enum code, const char* message, const char*, int) override {
        std::cerr << "PhysX " << unsigned(code) << ": " << message << '\n';
        if (code != PxErrorCode::eDEBUG_INFO) ++failures;
    }
};

struct Contacts : PxSimulationEventCallback {
    double impulseZ = 0.;
    bool overflow = false;
    void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
    void onWake(PxActor**, PxU32) override {}
    void onSleep(PxActor**, PxU32) override {}
    void onTrigger(PxTriggerPair*, PxU32) override {}
    void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) override {}
    void onContact(const PxContactPairHeader&, const PxContactPair* pairs, PxU32 count) override {
        for (PxU32 i=0; i<count; ++i) {
            PxContactPairPoint points[32];
            overflow |= pairs[i].contactCount > 32;
            const auto n=pairs[i].extractContacts(points,32);
            for (PxU32 k=0; k<n; ++k) impulseZ += std::abs(points[k].impulse.z);
        }
    }
};

PxFilterFlags filter(PxFilterObjectAttributes, PxFilterData,
                     PxFilterObjectAttributes, PxFilterData,
                     PxPairFlags& flags, const void*, PxU32) {
    flags=PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_FOUND |
          PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
    return PxFilterFlag::eDEFAULT;
}

void shape(PxPhysics& physics, PxRigidActor& actor, PxMaterial& material,
           PxVec3 half, PxVec3 offset=PxVec3(0.f)) {
    auto* s=physics.createShape(PxBoxGeometry(half),material,true);
    if (!s) throw std::runtime_error("shape creation failed");
    s->setLocalPose(PxTransform(offset));
    s->setContactOffset(.02f); s->setRestOffset(0.f);
    if (!actor.attachShape(*s)) throw std::runtime_error("shape attach failed");
    s->release();
}

void inertia(PxArticulationLink& body, float mass, PxVec3 half, PxVec3 com) {
    body.setMass(mass); body.setCMassLocalPose(PxTransform(com));
    body.setMassSpaceInertiaTensor(PxVec3(mass*(half.y*half.y+half.z*half.z)/3.f,
        mass*(half.x*half.x+half.z*half.z)/3.f,
        mass*(half.x*half.x+half.y*half.y)/3.f));
    body.setLinearDamping(0.f); body.setAngularDamping(0.f);
    body.setMaxDepenetrationVelocity(1.f);
}

void hinge(PxArticulationJointReducedCoordinate& j, PxVec3 parentOffset,
           PxArticulationAxis::Enum axis, float lower, float upper) {
    j.setJointType(PxArticulationJointType::eREVOLUTE);
    j.setParentPose(PxTransform(parentOffset)); j.setChildPose(PxTransform(PxIdentity));
    j.setMotion(axis,PxArticulationMotion::eLIMITED);
    j.setLimitParams(axis,PxArticulationLimit(lower,upper));
    j.setDriveParams(axis,PxArticulationDrive(160.f,4.f,40.f,PxArticulationDriveType::eFORCE));
    j.setFrictionCoefficient(0.f);
    j.setDriveTarget(axis,0.f); j.setDriveVelocity(axis,0.f);
    j.setJointPosition(axis,0.f); j.setJointVelocity(axis,0.f);
}

void run(PxPhysics& physics, PxMaterial& linkMaterial, PxMaterial& floorMaterial,
         const std::string& directory, unsigned caseIndex, bool floor,
         const std::chrono::steady_clock::time_point& start) {
    Contacts contacts;
    PxSceneDesc desc(physics.getTolerancesScale());
    desc.gravity=PxVec3(0.f,0.f,-9.81f);
    desc.solverType=PxSolverType::eTGS;
    desc.flags |= PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS;
    desc.flags &= ~PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST;
    desc.flags &= ~PxSceneFlag::eENABLE_STABILIZATION;
    desc.flags &= ~PxSceneFlag::eENABLE_GPU_DYNAMICS;
    desc.broadPhaseType=PxBroadPhaseType::eABP;
    desc.cpuDispatcher=PxDefaultCpuDispatcherCreate(1);
    desc.filterShader=filter;desc.simulationEventCallback=&contacts;
    if (desc.cudaContextManager || (desc.flags & PxSceneFlag::eENABLE_GPU_DYNAMICS))
        throw std::runtime_error("CPU-only contract violated");
    auto* scene=physics.createScene(desc);
    if (!scene) throw std::runtime_error("scene creation failed");
    PxRigidStatic* ground=nullptr;
    if (floor) {
        ground=physics.createRigidStatic(PxTransform(PxVec3(0.f,0.f,-.025f)));
        shape(physics,*ground,floorMaterial,PxVec3(2.f,2.f,.025f));
        scene->addActor(*ground);
    }
    auto* articulation=physics.createArticulationReducedCoordinate();
    articulation->setSolverIterationCounts(positionIterations,velocityIterations);
    articulation->setSleepThreshold(0.f);
    articulation->setArticulationFlag(PxArticulationFlag::eDISABLE_SELF_COLLISION,true);
    // A 30kg floating load with known COM eccentricity supplies simultaneous
    // pitch and roll moments. The two finite foot contacts close the chain.
    const PxVec3 origin(0.f,0.f,.6f);
    auto* base=articulation->createLink(nullptr,PxTransform(origin));
    inertia(*base,30.f,PxVec3(.15f,.10f,.10f),PxVec3(.035f,.01f,0.f));
    shape(physics,*base,linkMaterial,PxVec3(.15f,.10f,.10f));
    std::array<PxArticulationJointReducedCoordinate*,4> joints;
    for (unsigned side=0; side<2; ++side) {
        const PxVec3 offset(0.f,side ? -.12f : .12f,-.55f);
        auto* knuckle=articulation->createLink(base,PxTransform(origin+offset));
        inertia(*knuckle,.1f,PxVec3(.025f),PxVec3(0.f));
        auto* pitch=knuckle->getInboundJoint();
        hinge(*pitch,offset,PxArticulationAxis::eSWING1,-.87267f,.5236f);
        auto* foot=articulation->createLink(knuckle,PxTransform(origin+offset));
        inertia(*foot,1.5f,PxVec3(.12f,.05f,.025f),PxVec3(.035f,.01f,-.025f));
        shape(physics,*foot,linkMaterial,PxVec3(.12f,.05f,.025f),PxVec3(.035f,.01f,-.025f));
        auto* roll=foot->getInboundJoint();
        hinge(*roll,PxVec3(0.f),PxArticulationAxis::eTWIST,-.2618f,.2618f);
        joints[2*side]=pitch;
        joints[2*side+1]=roll;
    }
    scene->addArticulation(*articulation);
    auto* cache=articulation->createCache();
    if (articulation->getDofs()!=4 || !cache) throw std::runtime_error("four-DOF fixture required");
    std::ofstream output(directory+"/case_"+std::to_string(caseIndex)+".csv");
    if (!output) throw std::runtime_error("cannot create output");
    output << std::setprecision(12) << "step";
    for (unsigned j=0;j<4;++j) {
        output << ",q" << j << ",v" << j;
        for (unsigned p=0;p<positionIterations;++p) output << ",p" << p << "v" << j;
    }
    output << ",root_z,normal_load_bw\n";
    for (unsigned i=0; i<steps; ++i) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>40.)
            throw std::runtime_error("40 second compute deadline");
        contacts.impulseZ=0.;positionCalls=0;hookError=false;
        scene->simulate(dt);scene->fetchResults(true);
        if (contacts.overflow) throw std::runtime_error("contact buffer overflow");
        if (hookError || positionCalls!=positionIterations) throw std::runtime_error("position-phase hook count mismatch");
        articulation->copyInternalStateToCache(*cache,PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
        output << i+1;
        for (unsigned j=0; j<joints.size(); ++j) {
            output << ',' << cache->jointPosition[j] << ',' << cache->jointVelocity[j];
            for (unsigned p=0;p<positionIterations;++p) output << ',' << positionVelocity[p][j];
        }
        output << ',' << base->getGlobalPose().p.z << ',' << contacts.impulseZ/(dt*33.2*9.81) << '\n';
    }
    output.close();
    std::cout << "case=" << caseIndex << " floor=" << floor << " contact_last=false"
              << " physics_steps=" << steps << " gpu=false\n";
    cache->release();articulation->release();if (ground) ground->release();scene->release();
    static_cast<PxDefaultCpuDispatcher*>(desc.cpuDispatcher)->release();
}

int main(int argc, char** argv) {
    if (argc!=2) {std::cerr << "usage: contact_readout EXISTING_EMPTY_OUTPUT_DIRECTORY\n";return 2;}
    if (!std::filesystem::is_directory(argv[1]) || !std::filesystem::is_empty(argv[1])) {
        std::cerr << "output directory must exist and be empty; evidence is never overwritten\n";return 2;
    }
    const auto start=std::chrono::steady_clock::now();
    PxDefaultAllocator allocator;Errors errors;
    auto* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    if (!foundation) return 3;
    auto* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,nullptr);
    if (!physics) {foundation->release();return 3;}
    auto* linkMaterial=physics->createMaterial(.5f,.5f,0.f);
    auto* floorMaterial=physics->createMaterial(.8f,.6f,0.f);
    int status=0;
    try {
        run(*physics,*linkMaterial,*floorMaterial,argv[1],0,false,start);
        run(*physics,*linkMaterial,*floorMaterial,argv[1],1,true,start);
    } catch (const std::exception& e) {std::cerr << e.what() << '\n';status=4;}
    linkMaterial->release();floorMaterial->release();physics->release();foundation->release();
    if (errors.failures) status=5;
    std::cout << "wall_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << " errors=" << errors.failures << " status=" << status << '\n';
    return status;
}
