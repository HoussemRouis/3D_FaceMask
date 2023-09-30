import open3d as o3d
import trimesh
from pathlib import Path
import numpy as np


class Mesh:
    """
    Class to store and manipulate the 3D mesh
    """
    _mesh = None
    _position = [0, 0, 0]
    _orientation =[0, 0, 0]

    def __init__(self, path: str):
        if not Path(path).is_file():
            print("Bad mesh file")
            return

        # Load the 3D mesh using trimesh
        mesh = trimesh.load(path)
        vertices = np.array(mesh.vertices)
        triangles = np.array(mesh.faces)
        self._mesh = o3d.geometry.TriangleMesh()
        self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self._mesh.triangles = o3d.utility.Vector3iVector(triangles)

    def transform(self, translation: np.ndarray, rotation: np.ndarray):
        """
        Transforms the 3D mesh pose given a translation and rotation vector
        :param translation: an array of 3 floats describing the cartesian translations
        :param rotation: an array of 3 floats describing the euler angles
        """
        if not self._mesh:
            print("Cannot transform an empty mesh!")
            return

        if len(translation) != 3 or len(rotation) != 3:
            return
        self._position = [translation[0] - self._position[0], translation[1] - self._position[1], translation[2] - self._position[2]]
        self._orientation = [rotation[0] - self._orientation[0], rotation[1] - self._orientation[1], rotation[2] - self._orientation[2]]
        # Create a transformation matrix
        # translation_matrix = np.identity(4)
        # translation_matrix[:3, 3] = self._position
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(self._orientation))
        transformation_matrix = np.identity(4)  # Initialize a 4x4 identity matrix
        transformation_matrix[:3, :3] = rotation_matrix[:3, :3]  # Copy the rotation part
        transformation_matrix[:3, 3] = self._position  # Set the translation part
        vertices = np.asarray(self._mesh.vertices)

        # Compute the centroid of the vertices
        centroid = np.mean(vertices, axis=0)
        print("Current pos : ", centroid)

        # Apply the transformation on the mesh
        self._mesh.transform(transformation_matrix)