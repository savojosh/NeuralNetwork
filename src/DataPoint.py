from pandas import Series
from enum import Enum
from math import isinf, isnan

class _Separator(Enum):
    KEY_SEPARATOR = ";"
    ATTRIBUTE_SEPARATOR = ","

class _Prefix(Enum):
    Y = "y"
    VALUES = "v"

class DataPoint:
    
    #-------------[ CONSTRUCTORS ]-------------#

    def __init__(self, y: list[float], values: list[float]) -> object:
        
        self.values = values
        self.y = y

    @classmethod
    def fromString(cls, dpString: str) -> object:
        yStr, vStr = dpString.split(_Separator.KEY_SEPARATOR)
        return cls(
            [float(y) for y in yStr.split(_Separator.ATTRIBUTE_SEPARATOR)],
            [float(v) for v in vStr.split(_Separator.ATTRIBUTE_SEPARATOR)]
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

    #-------------[ OBJECT DESCRIPTORS ]-------------#

    def __str__(self) -> str:
        return _Separator.KEY_SEPARATOR.join([
            _Separator.ATTRIBUTE_SEPARATOR.join([str(i) for i in self.y]), 
            _Separator.ATTRIBUTE_SEPARATOR.join([str(i) for i in self.values])
        ])
    
    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y), len(self.values))
    
    @property
    def ySize(self) -> int:
        return len(self.y)
    
    @property
    def xSize(self) -> int:
        return len(self.values)
    
    @property
    def label(self) -> int:
        return max(range(self.ySize), key=self.y.__getitem__)

    def toSeries(self) -> Series:

        data = {}
        for i in range(len(self.y)):
            data[f"{_Prefix.Y.value}{i}"] = self.y[i]
        for i in range(len(self.values)):
            data[f"{_Prefix.VALUES.value}{i}"] = self.values[i]
        
        s = Series(data=data, index=data.keys())

        return s
    