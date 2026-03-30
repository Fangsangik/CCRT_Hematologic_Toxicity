from dataclasses import dataclass


@dataclass
class Delta :
    absolute : float
    relative : float

    @classmethod
    def from_values(cls, baseline : float, current : float) -> "Delta" :
        absolute = current - baseline
        if abs(baseline) < 1e-6 :
            relative = 0.0
        else :
            relative = (absolute / baseline) * 100.0
        return cls(absolute=absolute, relative=relative)