from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from zones import Zone
    from devices import OpticalDevice

class Atom:
    def __init__(self, atom_id: int, x: float, y: float, zone: 'Zone', device: Optional['OpticalDevice'] = None):
        self.atom_id = atom_id
        self.x = x
        self.y = y
        self.zone = zone
        self.device = device

    def move(self, new_x: float, new_y: float):
        self.x = new_x
        self.y = new_y
 
    def __repr__(self):
        return f"Atom(id={self.atom_id}, x={self.x}, y={self.y}, zone={self.zone.__class__.__name__})"