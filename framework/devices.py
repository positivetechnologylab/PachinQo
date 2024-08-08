from typing import List, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from atom import Atom

class OpticalDevice:
    def __init__(self, name: str):
        self.name = name
        self.atoms: List[int] = []  # List to hold atom IDs

    def add_atom(self, atom: 'Atom'):
        if atom.atom_id not in self.atoms:
            self.atoms.append(atom.atom_id)
            atom.device = self

    def remove_atom(self, atom: 'Atom'):
        if atom.atom_id in self.atoms:
            self.atoms.remove(atom.atom_id)
            atom.device = None

    def __repr__(self):
        return f"{self.name} Device with {len(self.atoms)} atoms"

class AOD(OpticalDevice):
    def __init__(self):
        super().__init__('AOD')
        self.columns: List[List['Atom']] = []
        self.col_status: List = []
        
    def move_column(self, column_index: int, dx: float, dy: float) -> float:
        if column_index < 0 or column_index >= len(self.columns):
            print("Error: Invalid column index.")
            return 0
        column = self.columns[column_index]
        for atom in column:
            if atom:
                new_x = atom.x + dx
                new_y = atom.y + dy
                atom.move(new_x, new_y)
                
        distance_moved = math.sqrt(dx**2 + dy**2)
        return distance_moved

    def add_atom_to_column(self, atom: 'Atom', column_index: int):
        while len(self.columns) <= column_index:
            self.columns.append([])
        for col in self.columns:
            if atom in col:
                return  # atom has already been added
        self.columns[column_index].append(atom)
        atom.device = self  # update the atom's device to AOD

    def remove_atom(self, atom: 'Atom'):
        for col in self.columns:
            if atom in col:
                col.remove(atom)
                atom.device = None
                break

    def __repr__(self):
        column_str = ', '.join([f"Column {i+1} with {len(col)} atoms" for i, col in enumerate(self.columns)])
        return f"{self.name} Device with columns: {column_str}"

class SLM(OpticalDevice):
    def __init__(self):
        super().__init__('SLM')