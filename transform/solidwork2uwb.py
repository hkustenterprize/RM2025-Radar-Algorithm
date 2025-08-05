from driver.referee.referee_comm import FACTION
from typing import List


def solidwork2uwb(pos_3d: List[float], faction: FACTION = FACTION.RED) -> List[float]:
    x, y, z = pos_3d
    if (
        faction == FACTION.RED
        or faction == FACTION.UNKONWN
    ):
        pos_x = (-z + 14.0)
        pos_y = (-x + 7.5)
    elif faction == FACTION.BLUE:
        x, y, z = pos_3d
        pos_x = 28.0 - (-z + 14.0)
        pos_y = 15.0 - (-x + 7.5)
    return pos_x, pos_y

