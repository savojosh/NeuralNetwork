class DataPoint:
    
    def __init__(self, values: list[float], y: list[float]):
        
        self.values = values
        self.y = y
        self.label = max(range(len(y)), key=y.__getitem__)
