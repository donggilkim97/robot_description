import os
import random
import omni.usd
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

def apply_perfect_physics_settings(prim):
    # Enable Gravity and Physics (Rigid Body API) so it falls to the table
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        
    # Give it a realistic mass (e.g., 0.1 kg) so the gripper can lift it easily
    if not prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(0.1)

    # Apply the base Collision API if the object doesn't have it yet
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
        
    # Apply the Mesh Collision API to change how the physical shape is calculated
    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    else:
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
        
    # Force the physics engine to wrap the mesh perfectly (Convex Decomposition)
    mesh_collision_api.CreateApproximationAttr().Set("convexDecomposition")

    # Apply the PhysX Collision API to tweak the invisible force fields
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    else:
        physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
        
    # Shrink the contact offset to 1 millimeter (0.001) and rest offset to exactly 0.0
    physx_collision_api.CreateContactOffsetAttr().Set(0.001)
    physx_collision_api.CreateRestOffsetAttr().Set(0.0)

# Define the folder path containing your USD objects
folder_path = "/home/donggil/robot_description/src/isaac/Collected_test1/Obj_asset/" 

valid_extensions = [".usd", ".usda", ".usdc"]
object_files = [f for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in valid_extensions)]

if not object_files:
    print("Error: No USD files found in the specified folder!")
else:
    stage = omni.usd.get_context().get_stage()
    
    # Pick exactly ONE random object from the list
    chosen_file = random.choice(object_files)
    usd_path = os.path.join(folder_path, chosen_file)

    prim_path = "/World/TargetObject"

    # Add the chosen USD to the stage
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    
    # Clear existing transforms and set a new translation
    xformable.ClearXformOpOrder()
    
    # Set the EXACT position: X=0.5, Y=0.0, Z=0.05
    xform_op = xformable.AddTranslateOp()
    xform_op.Set(Gf.Vec3d(0.5, 0.0, 0.05))
    
    # ---> NEW: Add a random rotation on the Z-axis (0 to 360 degrees) <---
    rot_op = xformable.AddRotateZOp()
    random_yaw = random.uniform(0.0, 360.0)
    rot_op.Set(random_yaw)
    
    # Apply the perfect physics settings
    apply_perfect_physics_settings(prim)
    
    print(f"Spawned {chosen_file} at X:0.5, Y:0.0, Z:0.05 with Z-Rotation: {random_yaw:.2f} degrees")
