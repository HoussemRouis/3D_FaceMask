import open3d as o3d


class Visualizer:
    """
    Class to handle visualisation
    """
    _vis = None

    def __init__(self):
        # o3d.visualization.VisualizerWithKeyCallback.BACKEND = "OpenGL"
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window()
        # Set up renderer with camera
        renderer = self._vis.get_render_option()
        renderer.mesh_show_back_face = True

    def add_object(self, mesh):
        self._vis.add_geometry(mesh)

    def clear(self):
        self._vis.destroy_window()

    def show(self):
        self._vis.poll_events()
        self._vis.update_renderer()
        self._vis.clear_geometries()
