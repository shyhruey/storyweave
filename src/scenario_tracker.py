class ScenarioTracker:
    def __init__(self):
        self.current_scene = ""

    def set_scene(self, scene_description):
        self.current_scene = scene_description

    def get_scene(self):
        return self.current_scene
