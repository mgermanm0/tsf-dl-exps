class Transform():
    def __init__(self) -> None:
        pass
    def transform(self, data):
        pass

class InvertibleTransform(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, data):
        return super().transform(data)
    
    def invert(self, data):
        pass
    
