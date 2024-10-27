class ContextManager:
    def __init__(self):
        self.history = []
        self.entities = {}

    def add_to_history(self, event):
        """add a new story event or user choice to the history."""
        if len(self.history) >= 3:  # limit context to last 3
            self.history.pop(0)
        self.history.append(event)

    def add_entity(self, name, value):
        """store entities encountered to ensure continuity."""
        self.entities[name] = value

    def get_context(self):
        """combine the history and entity details for story continuity."""
        context_str = " ".join(self.history)
        entities_str = ", ".join([f"{k}: {v}" for k, v in self.entities.items()])
        return f"Story so far: {context_str}. Entities: {entities_str}"
