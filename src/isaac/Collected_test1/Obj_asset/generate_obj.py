import os
import random

import omni.usd
import omni.kit.commands

try:
    from isaacsim.core.utils.stage import add_reference_to_stage
except Exception:
    from omni.isaac.core.utils.stage import add_reference_to_stage

from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, UsdShade, Sdf


# -----------------------------
# User settings
# -----------------------------
folder_path = "/home/donggil/robot_description/src/isaac/Collected_test1/Obj_asset"
target_root_path = "/World/TargetObject"

spawn_position = Gf.Vec3d(0.5, 0.0, 0.08)
object_mass = 0.10

# Good default for random objects / mugs.
# Options you may try:
# "convexHull"
# "convexDecomposition"
# "sdf"
collision_approximation = "convexDecomposition"

static_friction = 1.2
dynamic_friction = 1.0
restitution = 0.0

# Safer than 0.001 for grasping/contact.
contact_offset = 0.005
rest_offset = 0.0


# -----------------------------
# Helper functions
# -----------------------------
def remove_api_if_exists(prim, api_schema):
    try:
        if prim.HasAPI(api_schema):
            prim.RemoveAPI(api_schema)
    except Exception as e:
        print(f"[WARN] Could not remove API {api_schema} from {prim.GetPath()}: {e}")


def remove_nested_rigid_bodies(root_prim):
    """
    Remove RigidBodyAPI/MassAPI from all children.
    We keep only ONE rigid body on /World/TargetObject.
    """
    for prim in Usd.PrimRange(root_prim):
        if prim.GetPath() == root_prim.GetPath():
            continue

        remove_api_if_exists(prim, UsdPhysics.RigidBodyAPI)
        remove_api_if_exists(prim, UsdPhysics.MassAPI)

        try:
            remove_api_if_exists(prim, PhysxSchema.PhysxRigidBodyAPI)
        except Exception:
            pass


def create_or_get_scope(stage, path):
    prim = stage.GetPrimAtPath(path)

    if not prim.IsValid():
        UsdGeom.Scope.Define(stage, path)

    return stage.GetPrimAtPath(path)


def create_or_get_physics_material(stage):
    """
    Create a physics material.
    This material is NOT used as a visual/render material.
    It will be bound using material:binding:physics only.
    """
    create_or_get_scope(stage, "/World/PhysicsMaterials")

    material_path = "/World/PhysicsMaterials/HighFrictionPhysicsMaterial"

    mat_prim = stage.GetPrimAtPath(material_path)

    if not mat_prim.IsValid():
        material = UsdShade.Material.Define(stage, material_path)
    else:
        material = UsdShade.Material(mat_prim)

    mat_prim = material.GetPrim()

    if not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
        physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    else:
        physics_mat = UsdPhysics.MaterialAPI(mat_prim)

    physics_mat.CreateStaticFrictionAttr().Set(static_friction)
    physics_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_mat.CreateRestitutionAttr().Set(restitution)

    return material


def bind_physics_material_only(prim, material):
    """
    IMPORTANT:
    This binds the material only for physics, using material:binding:physics.
    It does NOT overwrite the visual material / texture.
    """
    try:
        rel = prim.CreateRelationship("material:binding:physics", False)
        rel.ClearTargets(True)
        rel.AddTarget(material.GetPath())
    except Exception as e:
        print(f"[WARN] Could not bind physics material to {prim.GetPath()}: {e}")


def apply_root_rigid_body(root_prim):
    """
    Apply exactly one rigid body to the object root.
    """
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    else:
        rb_api = UsdPhysics.RigidBodyAPI(root_prim)

    try:
        rb_api.CreateRigidBodyEnabledAttr().Set(True)
    except Exception:
        pass

    if not root_prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    else:
        mass_api = UsdPhysics.MassAPI(root_prim)

    mass_api.CreateMassAttr().Set(object_mass)

    try:
        if not root_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        else:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI(root_prim)

        if hasattr(physx_rb, "CreateEnableCCDAttr"):
            physx_rb.CreateEnableCCDAttr().Set(True)

        if hasattr(physx_rb, "CreateDisableGravityAttr"):
            physx_rb.CreateDisableGravityAttr().Set(False)

    except Exception as e:
        print(f"[WARN] Optional PhysX rigid body settings failed: {e}")


def apply_colliders_to_meshes(root_prim, physics_material):
    """
    Apply collision to mesh descendants only.
    Do NOT apply RigidBodyAPI to mesh descendants.
    Preserve visual material/texture.
    """
    mesh_count = 0

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh_count += 1

        # Make sure mesh is not a separate rigid body.
        remove_api_if_exists(prim, UsdPhysics.RigidBodyAPI)
        remove_api_if_exists(prim, UsdPhysics.MassAPI)

        try:
            remove_api_if_exists(prim, PhysxSchema.PhysxRigidBodyAPI)
        except Exception:
            pass

        # Collision API
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)

        # Mesh collision approximation
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        else:
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)

        mesh_collision_api.CreateApproximationAttr().Set(collision_approximation)

        # PhysX collision tuning
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        else:
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)

        physx_collision_api.CreateContactOffsetAttr().Set(contact_offset)
        physx_collision_api.CreateRestOffsetAttr().Set(rest_offset)

        # Physics material only.
        # This will not remove the object's visual material or texture.
        bind_physics_material_only(prim, physics_material)

    print(f"[INFO] Applied colliders to {mesh_count} mesh prim(s).")

    if mesh_count == 0:
        print("[WARN] No mesh prims found under /World/TargetObject.")


def clean_camera_nested_rigidbody(stage):
    """
    Your log showed nested rigid body warning on camera child.
    Camera visual/sensor should not be a separate rigid body under camera_link.
    """
    camera_rigid_paths = [
        "/World/ur5e_custom_setup/camera_link/Realsense/RSD455"
    ]

    for path in camera_rigid_paths:
        prim = stage.GetPrimAtPath(path)

        if prim.IsValid():
            remove_api_if_exists(prim, UsdPhysics.RigidBodyAPI)
            remove_api_if_exists(prim, UsdPhysics.MassAPI)

            try:
                remove_api_if_exists(prim, PhysxSchema.PhysxRigidBodyAPI)
            except Exception:
                pass

            print(f"[INFO] Removed nested rigid body APIs from camera child: {path}")


def delete_old_target_object(stage):
    if stage.GetPrimAtPath(target_root_path).IsValid():
        omni.kit.commands.execute(
            "DeletePrims",
            paths=[target_root_path]
        )
        print(f"[INFO] Deleted old {target_root_path}")


def spawn_random_object():
    valid_extensions = [".usd", ".usda", ".usdc"]

    object_files = [
        f for f in os.listdir(folder_path)
        if any(f.endswith(ext) for ext in valid_extensions)
    ]

    if not object_files:
        print("Error: No USD files found in the specified folder!")
        return

    stage = omni.usd.get_context().get_stage()

    # Remove old object so old wrong material bindings are gone too.
    delete_old_target_object(stage)

    chosen_file = random.choice(object_files)
    usd_path = os.path.join(folder_path, chosen_file)

    add_reference_to_stage(
        usd_path=usd_path,
        prim_path=target_root_path
    )

    root_prim = stage.GetPrimAtPath(target_root_path)

    if not root_prim.IsValid():
        print(f"[ERROR] Failed to spawn object at {target_root_path}")
        return

    # Set transform on root only.
    xformable = UsdGeom.Xformable(root_prim)
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(spawn_position)

    rotate_op = xformable.AddRotateZOp()
    random_yaw = random.uniform(0.0, 360.0)
    rotate_op.Set(random_yaw)

    # Remove nested rigid bodies from referenced asset.
    remove_nested_rigid_bodies(root_prim)

    # Create physics material without overwriting visual texture.
    physics_material = create_or_get_physics_material(stage)

    # Apply one rigid body to root.
    apply_root_rigid_body(root_prim)

    # Apply colliders to mesh children.
    apply_colliders_to_meshes(root_prim, physics_material)

    # Optional cleanup for camera warning.
    clean_camera_nested_rigidbody(stage)

    print("--------------------------------------------------")
    print(f"[SUCCESS] Spawned: {chosen_file}")
    print(f"[PATH]    {usd_path}")
    print(f"[POSE]    position={spawn_position}, yaw={random_yaw:.2f} deg")
    print("[PHYSICS] one root rigid body + mesh child colliders")
    print("[MATERIAL] physics material bound only to material:binding:physics")
    print("[VISUAL] original colors/textures should be preserved")
    print("--------------------------------------------------")


spawn_random_object()
