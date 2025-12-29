# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import trimesh
import os
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path

def compute_mass_from_stl(stl_path: str, density: float = 1240.0) -> float:
    """
    Computes the mass of an object from its STL file using a given density.

    Args:
        stl_path (str): Path to the STL file.
        density (float): Density of the material in kg/m^3 (default: PLA ~ 1240 kg/m^3).

    Returns:
        float: Estimated mass in kg.
    """
    mesh = trimesh.load_mesh(stl_path)
    ##TODO: CHECK IF MESH VOLUME IS CALCULATED CORRECTLY
    volume_mm3 = mesh.volume  # Volume in cubic millimeters
    volume_m3 = volume_mm3 * 1e-9  # Convert mm^3 to m^3
    return volume_m3 * density

def convert_stl_to_usd(stl_path: str, usd_output_path: str, mass: float = None)-> str:
    """
    Converts an STL file to USD format and sets mass properties.

    Args:
        stl_path (str): Path to the input STL file.
        usd_output_path (str): Path to store the converted USD file.
        mass (float, optional): Mass of the object in kg. Defaults to None.

    Returns:
        str: Path to the generated USD file.
    """
    # Check if the STL file exists
    if not os.path.isabs(stl_path):
        stl_path = os.path.abspath(stl_path)
    if not check_file_path(stl_path):
        raise ValueError(f"Invalid mesh file path: {stl_path}")

    # Ensure the output path is absolute
    if not os.path.isabs(usd_output_path):
        usd_output_path = os.path.abspath(usd_output_path)

    # Mass properties
    mass_props = schemas_cfg.MassPropertiesCfg(mass=mass) if mass is not None else None
    rigid_props = schemas_cfg.RigidBodyPropertiesCfg() if mass is not None else None
    
    # Collision properties (default convex decomposition)
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)

    # Create Mesh converter config
    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=stl_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(usd_output_path),
        usd_file_name=os.path.basename(usd_output_path),
        make_instanceable=True,
        collision_approximation="convexDecomposition",
    )

    # Convert mesh
    mesh_converter = MeshConverter(mesh_converter_cfg)
    print("Mesh importer output:")
    print(f"Generated USD file: {mesh_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)
    
    return str(mesh_converter.usd_path)
