from dataclasses import dataclass


@dataclass
class CBCRecord:
    week : int
    wbc : float
    anc : float
    alc : float
    amc : float
    plt : float
    hb : float
