from typing import Dict, List, Tuple
from atom import Atom
from constants import STORAGE_SP, ENTANG_SP

class Zone:
    def __init__(self, bottom_left_x: float, bottom_left_y: float, width: float, height: float, spacing: float):
        self.bottom_left_x = bottom_left_x
        self.bottom_left_y = bottom_left_y
        self.width = width
        self.height = height
        self.spacing = spacing
        self.atoms: Dict[int, Atom] = {}  # Dictionary mapping atom_id to atom object
        self.sites: List[Dict[str, Tuple[float, float] | bool]] = []
        self.initialize_sites()
    
    def initialize_sites(self):
        for y in range(self.bottom_left_y, self.bottom_left_y + self.height, self.spacing):
            for x in range(self.bottom_left_x, self.bottom_left_x + self.width, self.spacing):
                self.sites.append({'coords': (x, y), 'occupied': False})

    #Adds atom to a free site to store it in
    def add_atom(self, atom, site):

        if not self.is_site_valid_and_available(site):
            raise ValueError("The specified site is invalid or already occupied.")
        self.atoms[atom.atom_id] = atom
        atom.x, atom.y = site
        atom.zone = self
        for s in self.sites:
            if s['coords'] == site:
                s['occupied'] = True
                break

    def is_site_valid_and_available(self, site):
        return any(s['coords'] == site and not s['occupied'] for s in self.sites)

    def remove_atom(self, atom_id):
        if atom_id in self.atoms:
            atom = self.atoms.pop(atom_id)
            for s in self.sites:
                if s['coords'] == (atom.x, atom.y):
                    s['occupied'] = False
                    break
            atom.zone = None

    def display_zone_info(self):
        print(f"Zone Info: Bottom-left=({self.bottom_left_x}, {self.bottom_left_y}), Width={self.width}, Height={self.height}")
        print("Atoms in Zone:", list(self.atoms.keys()))

    def print_zone(self):
        pass
        
class StorageZone(Zone):
    def __init__(self, bottom_left_x: float, bottom_left_y: float, width: float, height: float):
        super().__init__(bottom_left_x, bottom_left_y, width, height, STORAGE_SP)
        
class EntanglementZone(Zone):
    def __init__(self, bottom_left_x: float, bottom_left_y: float, width: float, height: float, c: float):
        self.c = c  #used to compute spacing between sites vertically
        super().__init__(bottom_left_x, bottom_left_y, width, height, ENTANG_SP)
        
    #Initialize a grid of sites within the zone based on its dimensions (this implements big square gaps per related paper)
    def initialize_sites(self):
        self.sites = []
        num_columns = self.width // self.spacing
        for column_index in range(0, num_columns): 
            x_coord = self.bottom_left_x + column_index * self.spacing
            for y_offset in range(0, self.height, self.c * self.spacing):
                y_coord = self.bottom_left_y + y_offset
                self.sites.append({'coords': (x_coord, y_coord), 'occupied': False})

    def add_atom(self, atom, site):
        if(site['occupied']==True):
            raise ValueError("The specified site is invalid or already occupied1.")
        self.atoms[atom.atom_id] = atom
        atom.x, atom.y = site['coords']
        atom.zone = self
        #mark site as occupied
        for s in self.sites:
            if s['coords'] == site['coords']:
                s['occupied'] = True
                break

    def has_unoccupied_sites(self):
        for site in self.sites:
            if not site['occupied']:
                return True
        return False

class ReadoutZone(Zone):
    def __init__(self, bottom_left_x: float, bottom_left_y: float, width: float, height: float):
        super().__init__(bottom_left_x, bottom_left_y, width, height, STORAGE_SP)