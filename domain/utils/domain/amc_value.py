"""AMC"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AMCValue :
    value : float