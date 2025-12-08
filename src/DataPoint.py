from pandas import Series
from enum import Enum
from math import isinf, isnan

class _Prefix(Enum):
    Y = "y"
    VALUES = "v"

class DataPoint:
    
    #-------------[ CONSTRUCTORS ]-------------#

    def __init__(self, y: list[float], values: list[float]) -> None:
        
        self.values = values
        self.y = y
        self.label = max(range(len(y)), key=y.__getitem__)

    @classmethod
    def fromString(cls, dpString: str) -> object:
        yStr, vStr = dpString.split(";")
        return cls(
            [float(y) for y in yStr.split(",")],
            [float(v) for v in vStr.split(",")]
        )
    
    @classmethod
    def fromSeries(cls, s: Series) -> object:
        y = []
        values = []

        for i, v in s.items():
            if(isinf(float(v)) or isnan(float(v))):
                raise ValueError(f"Value {v} is either infinite or nan.")
            match str(i)[0]:
                case _Prefix.Y.value:
                    y.append(float(v))
                case _Prefix.VALUES.value:
                    values.append(float(v))
                case _:
                    raise KeyError(f"Index {i} does not match any of the enumerated keys: {[e.value for e in _Prefix]}")
    
        return cls(y, values)

    #-------------[ OBJECT CONVERSION ]-------------#

    def __len__(self) -> int:
        return len(self.values)

    def __str__(self) -> str:
        return ";".join([
            ",".join([str(i) for i in self.y]), 
            ",".join([str(i) for i in self.values])
        ])

    def toSeries(self) -> Series:

        data = {}
        for i in range(len(self.y)):
            data[f"{_Prefix.Y.value}{i}"] = self.y[i]
        for i in range(len(self.values)):
            data[f"{_Prefix.VALUES.value}{i}"] = self.values[i]
        
        s = Series(data=data, index=data.keys())

        return s
    