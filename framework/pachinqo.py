"""
6 trap changes for regular run
- Make function that after each layer checks EACH QUBIT against the original QASM File and tells us if any gates were skipped. This is essential for debugging.
To try:
- Different shapes for SLM qubits (X, Hex, triangular, square with single gaps, square with row gaps)
- AOD should be somewhat organized (e.g. maximize the number of operations that occur in one layer)
- Fix "wrap_up" function (ideally remove entirely...this is no good).
- Another thing to try to add: If the ONLY operation doable in a layer are SWAPs, then we should do no extra movement for 3 layers, JUST do the whole thing in one layer.
"""

"""
--------------

- Create list of executable gates:
Basically, each qubit in the circuit will maintain a pointer to its next executable gate.
Only CZ gates where BOTH qubits are waiting to execute the same gate will make it into the layer.
Else one qubit is stuck waiting on the other.

----------
A single compilation iteration (for a layer):

1. a) Starting with leftmost/rightmost AOD column first (switching order after each full iteration on ALL AOD columns),
see if any atoms on column can execute any CZ gates on entangling zone qubits (i.e. in layer), 
starting with the L/R-most column of qubits in SLM in entangling zone and moving "inwards" with checks 
(i.e. if starting w/Leftmost qubits in SLM in entangling zone, then checking L->R in the entangling zone)

1. b) If not 1a), check AOD columns immediately to left/right that have not yet been iterated on 
AND are not entangling this layer to see if any atoms in the column can be moved into position.

1.5 If column cannot execute anything, EITHER:
1.5 a) Move to the storage zone on the other side of the entangling zone for complete evasion
1.5 b) If there is a blocking column to the left or right, 
move down/up to a storage zone above/below, and move col as 
close horizontally to "next" column (either L/R) as possible (to minimize interference).
The point of moving it as close as possible is to compact as many unused columns as possible.

2. Else, move column into position so interacting qubit is next to shared atom/ready to execute CZ.

2.5 Spread atoms in same AOD column to ensure spacing 
(expand AOD column such that all atoms ASIDE from entangling one get spaced)

2.6 In the case that interacting qubits are in adjacent AOD columns (1b), 
the two interacting columns are moved to the leftmost or rightmost site.

3. Repeat above for each subsequent column. Break either when:
a) All AOD columns are iterated on, or
b) If iterating L->R on AOD columns, if rightmost site in entangling zone is occupied (leftmost if iterating R->L) 

3.5 If all AOD columns were iterated on and NO GATES executed, we need to reorganize the entire setup 
(i.e. switch atoms in AOD columns and atoms in the entangling zone)

NOTE: Each column is restricted to interacting with atoms that are to the L/R (depending on direction of movement)
of the next column to the L/R. Thus, interactable SLM atoms in entanglement zone are limited after 
the first column moves into entangling position.

Sidenote: Switching direction each full iteration reduces the bias that the columns get
(meaning by definition, on average, each column has the same number of columns considered before it for an even
number of iterations)

Sidenote: We might need to change the name of the whole method; the way this will process is more like 
"waves" of AOD columns washing back and forth over the entanglement zone until we need to reconfigure the AOD/
"""

"""
ALL of the below must happen within the block "if self.devices['AOD'].col_status[col_index] != Status.Entangling:"

So what needs to first happen in this if statement is find the adjacent, 
previously iterated column in compile_layer's for loop (if any). 
The direction can change depending on direction = 1 if self.iterate_left_to_right else -1 earlier in the function.

If that previous column is located in the storage zone opposite the current column 
(i.e. left if it is in right, and right if it is in left), then we can move this column to that storage zone, 
just to the side of the previous column ("side" being whichever direction the current column is moving from, 
so if it left of the previous column, it should be on the left side) by STORAGE_SP units.

If that previous column is located anywhere else, the current column must be moved to the bottom storage zone, 
again just to the side of the previous column ("side" again being whichever direction the current column is 
moving from, so if it left of the previous column, it should be on the left side) by STORAGE_SP units.

If this column is the first column to be iterated on in compile_layer, 
then move the column to the opposite end of the other storage location 
(i.e. if the column is currently in the left zone, move it to the rightmost edge of the right zone).
"""
"""
How SWAPs work:
- DEPRECATED: No more relevance score as it complicates compilation dramatically...
    - what if the entanglement zone qubit that is partially done with a SWAP but needs to execute a CZ?
- Each qubit has a "relevance" score associated with it: this is just the number of CZs involving it
    - Each time a layer of CZs is "executed", decrement the relevance score for the qubits

- Also, using executable gates, maintain array of binary flags to indicate if 
next CZ is between qubits that are both in either SLM or AOD 
(corresponds to same array size/indexing as executable_gates)
    - Each time we call compile_layers, at the end, we update this list to see if the next gate of executed
    qubits is in same device (i.e. both slm or both aod)

- In compile_layer:
    - For each column:
        - If nothing can compile:
            - If at least one:
                - Any qubit in this column
                have same-device gates 
            - Then:
                    - try to SWAP with accessuble qubit in entanglement zone that also needs to SWAP
                    (so both swap when both need to in one go) OR has no more CZs
        - If the above does not occur, THEN move column out of the way as usual
    - Only consider algo "complete" if no SWAPS or CZs execute

    

- SWAP def:
    - Just switch atom/qubit mappings (both)
"""

from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from enum import Enum
import time
import pickle
import os

STORAGE_SP = 2
ENTANG_SP = 10
MIN_SP = 0.5


"""
Status enum for each AOD column.
Waiting -> AOD column has not yet been iterated on for a compilation pass
Unused -> AOD column has been iterated on, but no atoms in column can be used in a CZ gate
Entangling -> AOD column has an atom that is being used in the execution of a CZ gate
"""
class Status(Enum):
    Waiting = 0
    Unused = 1
    Entangling = 2
    
class OpticalDevice:
    def __init__(self, name):
        self.name = name
        self.atoms = []  # List to hold atom IDs

    def add_atom(self, atom):
        if atom.atom_id not in self.atoms:
            self.atoms.append(atom.atom_id)
            atom.device = self

    def remove_atom(self, atom):
        if atom.atom_id in self.atoms:
            self.atoms.remove(atom.atom_id)
            atom.device = None

    def __repr__(self):
        return f"{self.name} Device with {len(self.atoms)} atoms"

# Subclasses for specific devices
class AOD(OpticalDevice):
    def __init__(self):
        super().__init__('AOD')
        self.columns = []  # List of lists, each sublist is a column of atoms
        self.col_status = []

    def move_column(self, column_index, dx, dy):
        if column_index < 0 or column_index >= len(self.columns):
            print("Error: Invalid column index.")
            return 0  # Return zero distance if the column index is invalid

        column = self.columns[column_index]
        for atom in column:
            if atom:
                # Calculate new positions
                new_x = atom.x + dx
                new_y = atom.y + dy
                # Update atom's position and zone
                atom.move(new_x, new_y)

        # Calculate and return the distance moved
        distance_moved = math.sqrt(dx**2 + dy**2)
        return distance_moved
        
    def add_atom_to_column(self, atom, column_index):
        # Ensure the column exists
        while len(self.columns) <= column_index:
            self.columns.append([])

        # Check if atom is already in any column
        for col in self.columns:
            if atom in col:
                return  # Atom is already added, no action required

        # Add atom ID to the specified column
        self.columns[column_index].append(atom)
        atom.device = self  # Update the atom's device to this AOD

    def remove_atom(self, atom):
        # Find and remove the atom from any column
        for col in self.columns:
            if atom.atom_id in col:
                col.remove(atom.atom_id)
                atom.device = None  # Clear the atom's device
                break  # Atom found and removed, no further action required

    def __repr__(self):
        column_str = ', '.join([f"Column {i+1} with {len(col)} atoms" for i, col in enumerate(self.columns)])
        return f"{self.name} Device with columns: {column_str}"

class SLM(OpticalDevice):
    def __init__(self):
        super().__init__('SLM')
        
class Atom:
    def __init__(self, atom_id, x, y, zone, device=None):
        self.atom_id = atom_id
        self.x = x
        self.y = y
        self.zone = zone  # The zone object the atom is in
        self.device = device  # The device object the atom is trapped by

    def move(self, new_x, new_y):
        # Update atom's position and zone
        self.x = new_x
        self.y = new_y
 
    def __repr__(self):
        return f"Atom(id={self.atom_id}, x={self.x}, y={self.y}, zone={self.zone.__class__.__name__})"

ZONE_BACKGROUND_COLORS = {
    'StorageZone': 'skyblue',
    'EntanglementZone': 'lightgreen',
    'ReadoutZone': 'salmon',
}

class Zone:
    def __init__(self, bottom_left_x, bottom_left_y, width, height, spacing):
        self.bottom_left_x = bottom_left_x
        self.bottom_left_y = bottom_left_y
        self.width = width
        self.height = height
        self.spacing = spacing
        self.atoms = {}  # Dictionary mapping atom_id to atom object
        self.sites = []
        self.initialize_sites()

    def initialize_sites(self):
        """Initialize a grid of sites within the zone based on its dimensions."""
        for y in range(self.bottom_left_y, self.bottom_left_y + self.height, self.spacing):
            for x in range(self.bottom_left_x, self.bottom_left_x + self.width, self.spacing):
                self.sites.append({'coords': (x, y), 'occupied': False})

    def add_atom(self, atom, site):
        """Adds an atom to the zone at the specified site."""
        # print(site)
        if not self.is_site_valid_and_available(site):
            raise ValueError("The specified site is invalid or already occupied.")
        self.atoms[atom.atom_id] = atom
        # Update the atom's position to match the specified site
        atom.x, atom.y = site
        atom.zone = self
        # Mark the site as occupied
        for s in self.sites:
            if s['coords'] == site:
                s['occupied'] = True
                break

    def is_site_valid_and_available(self, site):
        """Checks if the specified site is valid (within zone bounds) and available (not occupied)."""
        return any(s['coords'] == site and not s['occupied'] for s in self.sites)

    def remove_atom(self, atom_id):
        if atom_id in self.atoms:
            atom = self.atoms.pop(atom_id)
            # Mark the site as available when an atom is removed
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
    def __init__(self, bottom_left_x, bottom_left_y, width, height):
        super().__init__(bottom_left_x, bottom_left_y, width, height, STORAGE_SP)
        
class EntanglementZone(Zone):
    def __init__(self, bottom_left_x, bottom_left_y, width, height, c):
        self.c = c  # Number of qubits in a column
        super().__init__(bottom_left_x, bottom_left_y, width, height, ENTANG_SP)  # Assuming ENTANG_SP is defined elsewhere

    def initialize_sites(self):
        self.sites = []
        num_columns = self.width // self.spacing
        for column_index in range(0, num_columns):  # Exclude the first and last column
            x_coord = self.bottom_left_x + column_index * self.spacing
            for y_offset in range(0, self.height, self.c * self.spacing):
                y_coord = self.bottom_left_y + y_offset
                self.sites.append({'coords': (x_coord, y_coord), 'occupied': False})

    def add_atom(self, atom, site):
        """Adds an atom to the zone at the specified site."""
        # print(site)
        if(site['occupied']==True):
            raise ValueError("The specified site is invalid or already occupied1.")
        self.atoms[atom.atom_id] = atom
        # Update the atom's position to match the specified site
        atom.x, atom.y = site['coords']
        atom.zone = self
        # Mark the site as occupied
        for s in self.sites:
            if s['coords'] == site['coords']:
                s['occupied'] = True
                break

    def has_unoccupied_sites(self):
        """Checks if there are any unoccupied sites within the Entanglement Zone."""
        for site in self.sites:
            if not site['occupied']:
                return True  # At least one unoccupied site found
        return False  # No unoccupied sites found

class ReadoutZone(Zone):
    def __init__(self, bottom_left_x, bottom_left_y, width, height):
        super().__init__(bottom_left_x, bottom_left_y, width, height, STORAGE_SP)

def load_qasm_to_circuit(qasm_file_path):
    quantum_circuit = QuantumCircuit.from_qasm_file(qasm_file_path)
    return quantum_circuit

class Grid:
    #zone_specs 
    def __init__(self, zone_specs, start_zone_ind, qasm_file_path, num_atoms=280):
        self.zones = []
        self.devices = {'AOD': AOD(), 'SLM': SLM()}
        #zone_specs.append({'type': 'CornerZone', 'bottom_left_x': 0, 'bottom_left_y': 0, 'width': 80, 'height': 40})
        self.initialize_zones(zone_specs)
        self.initialize_atoms(num_atoms,self.zones[start_zone_ind])
        
        self.corner_zone = self.zones[-1]
        self.qubit_atom_map = {} #qubit ID to atom ID
        self.atom_qubit_map = {}
        self.qc = None
        self.iterate_left_to_right = True  # Initial direction
        self.executable_gates = None
        self.has_entang_flag = False

        self.same_zone_czs = [] #list of bools of same length as executable_gates; if 1, then qubits in next CZ are in same zone
        #list of counters to see stage SWAP is in (0,1,2 -> number of CZs that have occured; at 3, resets)
        #each element of form (qubit_id, stage)
        self.partial_swaps = [] 
        start_time = time.time()
        self.qc = load_qasm_to_circuit(qasm_file_path)
        self.qubit_to_atom(self.qc) 
        compiled_czs,fs,reorder_dist = self.reorder_atoms(qasm_file_path,1000000)
        self.move_storage_atoms_to_AOD()
        init_store_dist = self.move_AOD_columns_to_right_storage()
        
        eg = [1]
        swaps = [1]
        all_egs = []
        eg_layers = []
        swap_layers = []
        self.all_layers = []
        self.total_moved_dist = init_store_dist + reorder_dist
        self.atom_entanglement_bools = {key: 0 for key in self.atom_qubit_map} #keeps track of which atoms are already entangling
        self.all_layers = []
        self.total_moved_dist = init_store_dist + reorder_dist
        self.leftmost_slm_x = float('inf')
        self.rightmost_slm_x = float('-inf')
        self.update_slm_bounds()
        
        self.print_grid_by_atoms()
        while any(gate is not None for gate in self.executable_gates):
            # self.print_grid_by_atoms()
            eg,swaps,u3s,layer_dist_moved,slm_swaps = self.compile_layer()
            
            # self.print_grid_by_atoms()
            all_egs.extend(eg)
            eg_layers.append(eg)
            # print(eg,swaps,u3s,slm_swaps,layer_dist_moved)
            swap_layers.append(swaps)
            self.all_layers.append([eg,swaps,u3s,slm_swaps,layer_dist_moved])
            # print("Layer distance moved:",layer_dist_moved)
            self.total_moved_dist += layer_dist_moved
        meas_dist = self.simulate_measurement()
        end_time = time.time()
        self.comp_time = end_time - start_time
        self.total_moved_dist += meas_dist

        self.num_trap_changes = 6

    def return_vals(self):
        return [self.all_layers,self.total_moved_dist,self.num_trap_changes,self.comp_time]

    def initialize_zones(self, zone_specs):
        zone_objects = {'StorageZone': StorageZone, 'EntanglementZone': EntanglementZone, 'ReadoutZone': ReadoutZone}
        for spec in zone_specs:
            zone_class = zone_objects.get(spec['type'])
            if not zone_class:
                raise ValueError(f"Unknown zone type: {spec['type']}")
            if zone_class == EntanglementZone:
                new_zone = zone_class(spec['bottom_left_x'], spec['bottom_left_y'], spec['width'], spec['height'],spec['col_size'])
            else:
                new_zone = zone_class(spec['bottom_left_x'], spec['bottom_left_y'], spec['width'], spec['height'])
            self.zones.append(new_zone)
        

    def initialize_atoms(self, num_atoms, starting_zone):
        if starting_zone not in self.zones:
            raise ValueError("The specified starting zone is not part of the grid.")
        self.atom_init(starting_zone, num_atoms, self.devices['SLM'])

    def atom_init(self, zone, num_atoms, device):
        available_sites = [site['coords'] for site in zone.sites if not site['occupied']]
        for i, site in enumerate(available_sites):
            if i >= num_atoms:
                break
            atom_id = i
            atom = Atom(atom_id, *site, zone, device)
            zone.add_atom(atom, site)
            device.add_atom(atom)

    
    def print_grid(self):
        fig, ax = plt.subplots(figsize=(10, 8))

        for zone in self.zones:
            # Retrieve the background color based on the zone type
            background_color = ZONE_BACKGROUND_COLORS.get(zone.__class__.__name__, 'grey')

            # Create a rectangle patch for each zone based on its actual position and size
            zone_rect = patches.Rectangle((zone.bottom_left_x, zone.bottom_left_y), zone.width, zone.height, linewidth=1, edgecolor='r', facecolor=background_color, alpha=0.3)
            ax.add_patch(zone_rect)

            # Plot each atom within the zone, using the atom's coordinates directly
            for atom in zone.atoms.values():
                ax.scatter(atom.x, atom.y, color='blue', s=6)  # Atom coordinates are now absolute and not adjusted

        # Set labels and equal aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.axis('equal')  # This ensures that the scale of the plot is the same on both the x and y axes
        plt.show()
        
    def print_grid_by_atoms(self):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot zones
        for zone in self.zones:
            background_color = ZONE_BACKGROUND_COLORS.get(zone.__class__.__name__, 'grey')
            zone_rect = patches.Rectangle((zone.bottom_left_x, zone.bottom_left_y), zone.width, zone.height, linewidth=1, edgecolor='r', facecolor=background_color, alpha=0.3)
            ax.add_patch(zone_rect)

        # Plot atoms based on qubit_atom_map and color-code based on device
        for qubit, atom_id in self.qubit_atom_map.items():
            atom = self.find_atom_by_id(atom_id)
            if atom and atom.device != self.devices['AOD']:  # Exclude AOD atoms for now
                color = 'blue' if atom.device and atom.device.name == 'SLM' else 'black'

                ax.scatter(atom.x, atom.y, color=color, s=4)
                # Get the qubit ID for the label
                qubit_id = self.atom_qubit_map[atom.atom_id]
                # Add text above the atom
                ax.text(atom.x, atom.y + 3, str(qubit_id), color=color, ha='center', va='bottom')


        # Specifically handle AOD atoms, iterating over columns
        if 'AOD' in self.devices:
            for column in self.devices['AOD'].columns:  # Access columns of AOD
                for atom in column:
                    if atom:  # Ensure atom exists
                        # Print AOD atoms in red
                        ax.scatter(atom.x, atom.y, color='red', s=8)
                        qubit_id = self.atom_qubit_map[atom.atom_id]
                        # Add text above the atom
                        ax.text(atom.x, atom.y - 9, str(qubit_id), color='red', ha='center', va='bottom')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.axis('equal')
        plt.show()

    def update_slm_bounds(self):
        ent_zone = next(zone for zone in self.zones if isinstance(zone, EntanglementZone))
        slm_atoms = [self.find_atom_by_id(atom_id) for atom_id in self.devices['SLM'].atoms 
                     if isinstance(self.find_atom_by_id(atom_id).zone, EntanglementZone)]
        if slm_atoms:
            self.leftmost_slm_x = min(atom.x for atom in slm_atoms)
            self.rightmost_slm_x = max(atom.x for atom in slm_atoms)
    
    def qubit_to_atom(self, quantum_circuit):
        """
        Maps qubits defined in a QASM file (represented as a QuantumCircuit object)
        to atoms in the grid. Updates the grid to reflect this mapping, deactivating or
        removing atoms not mapped to a qubit.

        Parameters:
        - quantum_circuit: QuantumCircuit, the quantum circuit to map to the grid's atoms.
        """
        self.qubit_atom_map = {}  # Reset mapping for the new circuit
        num_qubits = quantum_circuit.num_qubits

        # Attempt to map each qubit to an atom
        for qubit_index in range(num_qubits):
            atom_id = qubit_index
            self.qubit_atom_map[qubit_index] = atom_id

        self.same_zone_czs = [0] * len(self.qubit_atom_map)
        self.partial_swaps = [[-1, 0] for _ in range(len(self.qubit_atom_map))]

        # Identify atoms not mapped to any qubit
        unmapped_atoms = set(atom_id for zone in self.zones for atom_id in zone.atoms if atom_id not in self.qubit_atom_map.values())

        # Remove or deactivate unmapped atoms (depending on your grid's structure and requirements)
        for zone in self.zones:
            zone.atoms = {atom_id: atom for atom_id, atom in zone.atoms.items() if atom_id not in unmapped_atoms}
        self.atom_qubit_map = {atom_id: qubit for qubit, atom_id in self.qubit_atom_map.items()}
#         print('-------------')
#         print(self.qubit_atom_map)
#         print('-----------------')
#         for zone in self.zones:
#             for at in zone.atoms:
#                 print(zone.atoms[at])

    def get_nearest_available_site(self, current_zone, target_zone, atom_x, atom_y):
        """Finds the nearest available site in the target zone based on global grid coordinates."""
        available_sites = [site for site in target_zone.sites if not site['occupied']]
        if not available_sites:
            return None

        # Calculate nearest site using absolute coordinates
        nearest_site = min(
            available_sites,
            key=lambda site: math.dist(
                [atom_x, atom_y],
                site['coords'])
            )
        return nearest_site

    def move_to_ent(self, atom_id):
        atom = self.find_atom_by_id(atom_id)
        if atom is None:
            print(f"Atom {atom_id} not found.")
            return 0

        current_zone = atom.zone
        if not current_zone:
            print(f"Atom {atom_id} not found in any zone.")
            return 0
        
        target_zone = next((zn for zn in self.zones if isinstance(zn, EntanglementZone)), None)
        if not target_zone:
            print("Target zone not found.")
            return 0

        try:
            nearest_site = self.get_nearest_available_site(current_zone, target_zone, atom.x, atom.y)
        except ValueError:
            print(f"No available sites in {type(target_zone).__name__}.")
            return 0

        # Calculate the distance moved
        distance_moved = math.sqrt((nearest_site['coords'][0] - atom.x) ** 2 + (nearest_site['coords'][1] - atom.y) ** 2)

        # Update atom details
        current_zone.remove_atom(atom_id)
        atom.x, atom.y = nearest_site['coords']
        atom.zone = target_zone
        atom.device = self.devices['SLM']
        target_zone.add_atom(atom, nearest_site)

        return distance_moved

        
    def initialize_executable_gates(self):
        self.executable_gates = [None] * self.qc.num_qubits  # Initialize with None indicating no gate found yet
        for qubit_index in range(self.qc.num_qubits):
            for gate_index, (instruction, qargs, _) in enumerate(self.qc.data):
                #print(instruction.name,qargs)
                if instruction.name == 'u3' and any(qarg._index == qubit_index for qarg in qargs):
                    #U3 gate
                    self.executable_gates[qubit_index] = gate_index
                    self.same_zone_czs[qubit_index] = 0
                    break
                elif instruction.name == 'cz' and any(qarg._index == qubit_index for qarg in qargs):
                    
                    self.executable_gates[qubit_index] = gate_index
                    #initialize same_zone_czs
                    atoms = []
                    for qarg in qargs:
                        atoms.append(self.find_atom_by_id(self.qubit_atom_map[qarg._index]))
                    # print(qubit_index,qargs)
                    #if both atoms in same device or zone, then set same_zone_czs to 1 to indicate they can't execute.
                    if atoms[0].device == atoms[1].device:
                        self.same_zone_czs[qubit_index] = 1

                    break  # Found the first CZ gate for this qubit

    #O(G*Q), since we are iterating over all gates, Q because we need to find the nearest site in the entanglement zone to move to
    def reorder_atoms(self, qasm_file_path, czs):
        quantum_circuit = load_qasm_to_circuit(qasm_file_path)
        self.qc = quantum_circuit
        
        observed_qubits = set()
        compiled_cz_gates = []
        final_states = {}
        max_distance_moved = 0  # Initialize max distance moved

        ent_zone = next((zn for zn in self.zones if isinstance(zn, EntanglementZone)), None)
        if not ent_zone:
            print("Target zone not found.")
            return 0

        for instruction, qargs, _ in quantum_circuit.data:
            if instruction.name != 'cz':
                continue
            qubits = [qarg._index for qarg in qargs]

            atom_ids = [self.qubit_atom_map.get(qubit) for qubit in qubits]
            if any(atom_id is None for atom_id in atom_ids):
                print("Skipping due to unmapped qubits.")
                continue

            atom_objects = [self.find_atom_by_id(atom_id) for atom_id in atom_ids]
            observed_status = [qubit in observed_qubits for qubit in qubits]
            # print("Observed status:", observed_status, qubits)
            available_sites = any(zn.has_unoccupied_sites() for zn in self.zones if isinstance(zn, EntanglementZone))
            
            if not any(observed_status) and available_sites:
                # print("H1")
                dist = self.move_to_ent(atom_objects[0].atom_id)
            elif observed_status[0] != observed_status[1] and available_sites:
                # print("H2")
                unobserved_atom = atom_objects[0] if not observed_status[0] else atom_objects[1]
                observed_atom = atom_objects[0] if observed_status[0] else atom_objects[1]
                if observed_atom.zone == ent_zone:
                    # print("H3")
                    pass
                else:
                    # print("H4")
                    dist = self.move_to_ent(unobserved_atom.atom_id)
            else:
                dist = 0
            
            max_distance_moved = max(max_distance_moved, dist)
            observed_qubits.update(qubits)
            if len(observed_qubits) == self.qc.num_qubits:
                break
            compiled_cz_gates.append((qubits[0], qubits[1]))

        for atom_id in self.qubit_atom_map.values():
            atom = self.find_atom_by_id(atom_id)
            final_states[atom_id] = {'zone': type(atom.zone).__name__, 'coords': (atom.x, atom.y)}

        self.reorganize_observed_in_storage(observed_qubits)
        
        return compiled_cz_gates, final_states, max_distance_moved

    
    def find_atom_by_id(self, atom_id):
        # Locate the atom object by its ID across all zones
        for zone in self.zones:
            if atom_id in zone.atoms:
                return zone.atoms[atom_id]
        for col in self.devices["AOD"].columns:
            for atom in col:
                if atom_id == atom.atom_id:
                    return atom
        return None

    def reorganize_observed_in_storage(self, observed_qubits):
        storage_zone = None
        entanglement_zone = None
        for zone in self.zones:
            if isinstance(zone, StorageZone) and zone.atoms != {}:
                storage_zone = zone
            elif isinstance(zone, EntanglementZone):
                entanglement_zone = zone

        if not storage_zone or not entanglement_zone:
            print("Error: Required zones not found.")
            return

        # Filter observed atoms that are currently in storage
        observed_atoms_in_storage = [self.qubit_atom_map[qubit] for qubit in observed_qubits if qubit in self.qubit_atom_map and self.find_atom_by_id(self.qubit_atom_map[qubit]).zone == storage_zone]

        # Find unoccupied sites in storage zone
        unoccupied_sites = [site['coords'] for site in storage_zone.sites if not site['occupied']]
        unoccupied_sites.sort(key=lambda x: (x[1], x[0]))  # Sort by y then x to fill rows first

        if not unoccupied_sites:
            print("Error: No unoccupied sites in storage.")
            return

        # Align atoms with entanglement zone columns, if possible
        entanglement_columns_x = [entanglement_zone.bottom_left_x + i * entanglement_zone.spacing for i in range(entanglement_zone.width // entanglement_zone.spacing)]

        for atom_id in observed_atoms_in_storage:
            atom = self.find_atom_by_id(atom_id)
            if atom:
                # Find the closest unoccupied site that aligns with an entanglement column, if possible
                suitable_sites = [site for site in unoccupied_sites if site[0] in entanglement_columns_x]
                if not suitable_sites:  # If no suitable sites, use any unoccupied site
                    suitable_sites = unoccupied_sites

                chosen_site = suitable_sites[0]  # Choose the first suitable site
                unoccupied_sites.remove(chosen_site)  # Mark as occupied

                # Move the atom to the chosen site
                storage_zone.remove_atom(atom.atom_id)  # Update site occupancy
                storage_zone.add_atom(atom, chosen_site)

    def move_storage_atoms_to_AOD(self):
        storage_zone = None
        # Find the storage zone
        for zone in self.zones:
            if isinstance(zone, StorageZone) and zone.atoms != {}:
                storage_zone = zone
                break

        if not storage_zone:
            print("No storage zone found.")
            return

        # Assuming atoms in the storage zone are already organized by columns based on entanglement zone spacing,
        # and we know the column width or can calculate it based on the storage zone's properties.
        column_width = ENTANG_SP
        num_columns = storage_zone.width // column_width

        # Initialize a structure to hold atoms by column
        columns = [[] for _ in range(num_columns)]

        # Sort atoms into columns based on their x-coordinates
        for atom_id, atom in storage_zone.atoms.items():
            column_index = int((atom.x - storage_zone.bottom_left_x) / column_width)
            columns[column_index].append(atom)

        #Remove any empty columns
        columns = [col for col in columns if col]

        #Remove any atoms from the SLM that are not mapped to qubits
        self.devices['SLM'].atoms = [atom for atom in self.devices['SLM'].atoms if atom in self.qubit_atom_map.values()]

        # Now, move each column of atoms to the AOD device
        for column_index, column_atoms in enumerate(columns):
            for atom in column_atoms:
                #Remove atom from list of static atoms in this devise
                storage_zone.remove_atom(atom.atom_id)
                self.devices['SLM'].remove_atom(atom)
                self.devices['AOD'].add_atom_to_column(atom, column_index)
                atom.device = self.devices['AOD']
            self.devices['AOD'].col_status.append(Status.Unused)
        self.initialize_executable_gates()
        # print(self.devices['AOD'].columns)
        # print("Moved atoms from storage to AOD.")
        
    # O(C)
    def move_AOD_columns_to_right_storage(self):
        
        right_most_storage_zone = max(
            (zone for zone in self.zones if isinstance(zone, StorageZone) or isinstance(zone, ReadoutZone)), 
            key=lambda z: z.bottom_left_x, default=None
        )
        if not right_most_storage_zone:
            print("Error: No storage zone found.")
            return 0
    
        # Calculate the left edge of the rightmost storage zone
        left_edge_x = right_most_storage_zone.bottom_left_x
        start_y = (right_most_storage_zone.bottom_left_y + right_most_storage_zone.height) / 2
    
        max_distance_moved = 0
        current_x = left_edge_x + STORAGE_SP  # Start from the left edge
    
        # Iterate over AOD columns from left to right
        for i, column in enumerate(self.devices['AOD'].columns):
            if not column:  # Skip empty columns
                continue
            # print(column)
            # Calculate new position
            new_x = current_x
            new_y = start_y
    
            # Check if new position is within the storage zone
            if new_x + STORAGE_SP > right_most_storage_zone.bottom_left_x + right_most_storage_zone.width:
                print(f"Warning: Column {i} cannot fit in the storage zone. Stopping movement.")
                break
    
            # Calculate displacement
            dx = new_x - column[0].x
            dy = new_y - column[0].y
    
            # Move column
            distance_moved = self.devices['AOD'].move_column(i, dx, dy)
            max_distance_moved = max(max_distance_moved, distance_moved)
    
            # Update current_x for the next column
            current_x += STORAGE_SP
    
        return max_distance_moved
    # O(C)
    def move_AOD_columns_to_left_storage(self):
        left_most_storage_zone = min(
            (zone for zone in self.zones if isinstance(zone, StorageZone) or isinstance(zone, ReadoutZone)), 
            key=lambda z: z.bottom_left_x, default=None
        )
        if not left_most_storage_zone:
            print("Error: No storage zone found.")
            return 0
    
        # Calculate the right edge of the leftmost storage zone
        right_edge_x = left_most_storage_zone.bottom_left_x + left_most_storage_zone.width
        start_y = (left_most_storage_zone.bottom_left_y + left_most_storage_zone.height) / 2
    
        max_distance_moved = 0
        current_x = right_edge_x - STORAGE_SP  # Start from the right edge
    
        # Iterate over AOD columns from right to left
        for i in range(len(self.devices['AOD'].columns) - 1, -1, -1):
            column = self.devices['AOD'].columns[i]
            if not column:  # Skip empty columns
                continue
            # print(column)
    
            # Calculate new position
            new_x = current_x
            new_y = start_y
    
            # Check if new position is within the storage zone
            if new_x < left_most_storage_zone.bottom_left_x:
                print(f"Warning: Column {i} cannot fit in the storage zone. Stopping movement.")
                break
    
            # Calculate displacement
            dx = new_x - column[0].x
            dy = new_y - column[0].y
    
            # Move column
            distance_moved = self.devices['AOD'].move_column(i, dx, dy)
            max_distance_moved = max(max_distance_moved, distance_moved)
    
            # Update current_x for the next column
            current_x -= STORAGE_SP
    
        return max_distance_moved
    #Given list of executable gates, move pointers after execution; also update same_zone_czs and relevance scores
    # O(Q*G) <- find qubit, find gate for each qubit
    def update_executable_gates(self, executed_gate_indices):
        # O(Q) since executed_gate_indices is of size Q
        for executed_gate_index in executed_gate_indices:
            instruction, qargs, _ = self.qc.data[executed_gate_index]
            
            # if instruction.name != 'cz':
            #     continue  # Skip if not a CZ gate

            for qarg in qargs:
                qubit_index = qarg._index
                # Find the next CZ gate for this qubit
                next_gate_index = self.find_next_gate_for_qubit(qubit_index, executed_gate_index + 1)
                self.executable_gates[qubit_index] = next_gate_index
                #print("NG: for q",qubit_index,next_gate_index)
            
        #update same_zone_czs; index in executable_gates corresponds to qubit index
        for qubit_index, next_gate_ind in enumerate(self.executable_gates):
            if next_gate_ind == None:
                continue
            instruction, qargs, _ = self.qc.data[next_gate_ind]
            atoms = []
            for qarg in qargs:
                atoms.append(self.find_atom_by_id(self.qubit_atom_map[qarg._index]))
            #if both atoms in same device or zone, then set same_zone_czs to 1 to indicate they can't execute.
            if len(atoms) == 1:
                self.same_zone_czs[qubit_index] = 0
            elif atoms[0].device == atoms[1].device:
                #ONLY update the executable_gates qubit, not both qubits
                self.same_zone_czs[qubit_index] = 1
            else:
                self.same_zone_czs[qubit_index] = 0
            

    #Get next gate given a qubit, and the last CZ it executed
    #O(G)
    def find_next_gate_for_qubit(self, qubit_index, start_index):
        for index in range(start_index, len(self.qc.data)):
            instruction, qargs, _ = self.qc.data[index]
            if any(qarg._index == qubit_index for qarg in qargs):
                # print("NEXTCZ: ",qubit_index,instruction,qargs)
                return index
        return None  # Indicates no more CZ gates for this qubit
    
    #Execute layer of U3 gates 
    #Note since U3s can occur anywhere, anytime, this really just involves adding the U3 gate to the executed gates list
    #and updating the executable gates list
    #Also O(Q*G) same as above
    def process_U3_layer(self):
        executed_U3_gates = []
        for qubit_index, gate_index in enumerate(self.executable_gates):
            if gate_index == None:
                continue
            instruction, qargs, _ = self.qc.data[gate_index]
            if instruction.name == 'u3':
                executed_U3_gates.append(gate_index)
        self.update_executable_gates(executed_U3_gates)
        return executed_U3_gates
    
    def get_u3_indices_from_qasm(self, file_path):
            """Extract line indices from a QASM file where u3 gates are declared."""
            u3_indices = []
            with open(file_path, 'r') as file:
                for i, line in enumerate(file):
                    if 'u3(' in line:
                        u3_indices.append(i - 3)  # Adjust for the first three non-gate lines
            return u3_indices
    
    def wrap_up_u3s(self, qasm_file_path):
        """Analyze self.all_layers for U3 discrepancies with the QASM file and note missing U3s."""
        # Extract U3 indices from the QASM file
        qasm_u3_indices = self.get_u3_indices_from_qasm(qasm_file_path)

        # Collect all u3s indices from self.all_layers
        all_u3s = []
        for layer in self.all_layers:
            _, _, u3s, _, _ = layer
            all_u3s.extend(u3s)

        # Find missing U3 indices by comparing sets
        qasm_u3_indices_set = set(qasm_u3_indices)
        all_u3s_set = set(all_u3s)
        missing_u3_indices = list(qasm_u3_indices_set - all_u3s_set)

        # Append a new layer to self.all_layers noting the missing U3 indices
        self.all_layers.append([[], [], missing_u3_indices, [], 0])

    #O(Q*A_SLM)
    #O(Q*A_COL) 
    #O(Q*G)
    def compile_layer(self):
        #If all executable gates are None, then return (no more gates to execute)
        if all(gate == None for gate in self.executable_gates):
            return [], [], [], 0, []
        executed_gates = []
        biggest_move_dist = 0
        last_column_x = None
        last_column_status = None 
        direction = 1 if self.iterate_left_to_right else -1
        edge_gap_used_flag = False
        #print(self.executable_gates)
        has_entang_flag = False
        swap_czs = []
        slm_full_swaps = []
        # Save initial column positions,
        initial_positions = [(column[0].x, column[0].y) if column else (None, None) 
                                    for column in self.devices['AOD'].columns]
                             
        #If all executable gates are None after U3s, then return (no more gates to execute)
        if all(gate == None for gate in self.executable_gates):
            return [], [], [], 0, []

        # Ensure col_status is correctly sized and reset at start
        if not self.devices['AOD'].col_status or len(self.devices['AOD'].col_status) != len(self.devices['AOD'].columns):
            self.devices['AOD'].col_status = [Status.Unused] * len(self.devices['AOD'].columns)
        
        # Fetch the entanglement zone
        entanglement_zone = next((zone for zone in self.zones if isinstance(zone, EntanglementZone)), None)
        if not entanglement_zone:
            print("Entanglement zone not found.")
            return

        # Iterate through AOD columns according to the direction
        column_indices = range(len(self.devices['AOD'].columns)) if self.iterate_left_to_right else reversed(range(len(self.devices['AOD'].columns)))

        #A_C = number of atoms in an aod col
        #C = AOD Cols


        self.process_SLM_U3s(executed_gates)

        col_used_w_slm = False
        
        swaps = []
        for col_index in column_indices:
            # print(self.devices['AOD'].columns)
            column = self.devices['AOD'].columns[col_index]
    
            self.process_column_for_entanglement(col_index, column, entanglement_zone, executed_gates, swaps)

            if self.is_layer_complete(col_index):
                break
        
        # If no gates were executed, perform full swaps between SLM and AOD
        slm_swaps = []
        if not executed_gates:
            if not has_entang_flag:
                self.perform_full_swaps(slm_swaps)


        final_positions = [(column[0].x, column[0].y) if column else (None, None) 
                           for column in self.devices['AOD'].columns]

        # Calculate the maximum distance any column has moved
        max_distance_compiled = 0
        for init_pos, final_pos in zip(initial_positions, final_positions):
            if init_pos != (None, None) and final_pos != (None, None):
                dx = final_pos[0] - init_pos[0]
                dy = final_pos[1] - init_pos[1]
                distance = math.sqrt(dx**2 + dy**2)
                if distance > max_distance_compiled:
                    # print(init_pos, "|||", final_pos)
                    max_distance_compiled = distance
        # print("--------------------")
        # self.print_grid_by_atoms()
        self.update_executable_gates(executed_gates)
        # print("EXECUTABLE GATES: ",self.executable_gates)

        # print("SZCZ After: ",self.same_zone_czs)
        # Reset column statuses for the next layer compilation
        self.devices['AOD'].col_status = [Status.Unused] * len(self.devices['AOD'].columns)
        if self.iterate_left_to_right == 1:
            max_distance_cache = self.move_AOD_columns_to_left_storage()
        else:
            max_distance_cache = self.move_AOD_columns_to_right_storage()
        self.iterate_left_to_right = not self.iterate_left_to_right
        self.has_entang_flag = False
        for atom_id in self.atom_entanglement_bools.keys():
            self.atom_entanglement_bools[atom_id] = 0
        total_distance_moved = max_distance_compiled + max_distance_cache
        
        # executed gates should be CZs, swaps should be tuples of qubits, u3_gate_layer should extract all single qubit gates from executed gates, 
        #           , slm_swaps is a possible list of tuples of qubits for SLM FULL swaps, 
        u3_gate_layer = []
        remaining_executed_gates = []
        for gate_index in executed_gates:
            gate = self.qc[gate_index]
            if gate.operation.name == 'u3':
                u3_gate_layer.append(gate_index)
            else:
                remaining_executed_gates.append(gate_index)
        executed_gates = remaining_executed_gates
        
        return executed_gates, swaps, u3_gate_layer, total_distance_moved, slm_swaps


    def perform_full_swaps(self,slm_swaps):
        # print("d")
        unique_slm_qubits = self.get_unique_x_slm_qubits()
        column_indices = range(len(self.devices['AOD'].columns)) if self.iterate_left_to_right else reversed(range(len(self.devices['AOD'].columns)))
    
        for col_index in column_indices:
            if not unique_slm_qubits:
                break  # No more SLM qubits to swap
    
            column = self.devices['AOD'].columns[col_index]
            slm_qubit = unique_slm_qubits[0]
            
            # Find a suitable AOD qubit in the column
            aod_qubit = self.find_suitable_aod_qubit(column)
            if aod_qubit is None:
                continue  # No suitable AOD qubit in this column, move to next
    
            # Attempt to move the AOD column
            if self.attempt_move(col_index, column, aod_qubit, slm_qubit):
                # If move successful, perform the swap
                slm_swaps.append([slm_qubit,aod_qubit])
                self.swap_qubits(slm_qubit, aod_qubit)
                unique_slm_qubits.pop(0)  # Remove the swapped SLM qubit

    def get_unique_x_slm_qubits(self):
        slm_qubits = self.find_slm_qubits_needing_swap()
        unique_x_qubits = []
        x_positions = set()
    
        for qubit in slm_qubits:
            atom = self.find_atom_by_id(self.qubit_atom_map[qubit])
            if atom.x not in x_positions:
                unique_x_qubits.append(qubit)
                x_positions.add(atom.x)
    
        return sorted(unique_x_qubits, key=lambda q: self.find_atom_by_id(self.qubit_atom_map[q]).x)


    def find_slm_qubits_needing_swap(self):
        slm_qubits = []
        for qubit_index, gate_index in enumerate(self.executable_gates):
            if gate_index is not None:
                atom = self.find_atom_by_id(self.qubit_atom_map[qubit_index])
                if atom.device == self.devices['SLM'] and self.needs_swap(qubit_index):
                    # Check that the qubit is not already in a partial swap
                    if self.partial_swaps[qubit_index][1] == 0:
                        # Check if both qubits are ready to perform the same gate
                        gate = self.qc.data[gate_index]
                        other_qubit_index = gate.qubits[0]._index if gate.qubits[1]._index == qubit_index else gate.qubits[1]._index
                        if self.executable_gates[other_qubit_index] == gate_index:
                            slm_qubits.append(qubit_index)
        return slm_qubits
        
    def find_suitable_aod_qubit(self, aod_column):
        for atom in aod_column:
            qubit = self.atom_qubit_map[atom.atom_id]
            if self.partial_swaps[qubit][1] == 0:  # Check if not in a partial swap
                return qubit
        return None
    
    def swap_qubits(self, qubit1, qubit2):
        # Swap the qubit-atom mappings
        atom1_id = self.qubit_atom_map[qubit1]
        atom2_id = self.qubit_atom_map[qubit2]
        
        self.qubit_atom_map[qubit1], self.qubit_atom_map[qubit2] = atom2_id, atom1_id
        self.atom_qubit_map[atom1_id], self.atom_qubit_map[atom2_id] = qubit2, qubit1

    
    def process_SLM_U3s(self, executed_gates):
        for atom_id in self.devices['SLM'].atoms:
            qubit_index = self.atom_qubit_map[atom_id]
            if self.executable_gates[qubit_index] is not None:
                next_gate = self.qc[self.executable_gates[qubit_index]]
                if next_gate.operation.name == 'u3':
                    executed_gates.append(self.executable_gates[qubit_index])
        return executed_gates

    def debug_print_atom_info(self):
        print("\n--- Atom Information Debug ---")
        print("Qubit Index | Atom ID | Device | Zone | Position")
        print("-" * 50)
    
        # Sort atoms by qubit index for consistent output
        sorted_qubit_indices = sorted(self.qubit_atom_map.keys())
    
        for qubit_index in sorted_qubit_indices:
            atom_id = self.qubit_atom_map[qubit_index]
            atom = self.find_atom_by_id(atom_id)
            
            if atom is None:
                print(f"{qubit_index:<12} | {atom_id:<8} | Not Found")
                continue
    
            device = "Unknown"
            if atom.device:
                device = atom.device.name
    
            zone = "Unknown"
            if atom.zone:
                zone = atom.zone.__class__.__name__
    
            print(f"{qubit_index:<12} | {atom_id:<8} | {device:<6} | {zone:<15} | ({atom.x:.2f}, {atom.y:.2f})")
    
        print("\n--- End of Atom Information Debug ---")
    
    def process_column_for_entanglement(self, col_index, column, entanglement_zone, executed_gates, swaps):
        for atom in column:
            qubit_index = self.atom_qubit_map.get(atom.atom_id)
            #Checks that there are still gates for this qubit (if not, continue to next qubit)
            if not self.is_valid_qubit_for_entanglement(qubit_index):
                continue
            
            next_gate = self.qc[self.executable_gates[qubit_index]]
            #Order of ops: check u3 (no overhead, no complexity), check partial SWAP (if in the middle should try to continue), check CZ, check new SWAP (no 
            if next_gate.operation.name == 'u3':
                #This should NOT move the column, but just pop the executable gate and that's it
                executed_gates.append(self.executable_gates[qubit_index])
                continue
            #Then check CZ
            def both_qubits_ready(gate):
                if gate.operation.name == 'cz':
                    # print(gate)
                    other_qubit_index = gate.qubits[1]._index if gate.qubits[0]._index == qubit_index else gate.qubits[0]._index
                    # print(gate)
                    return self.executable_gates[qubit_index] == self.executable_gates[other_qubit_index]
                    
            # self.debug_print_atom_info()
            if not both_qubits_ready(next_gate):
                continue
            
            #Try continuing partial swap if qubit is in the middle of one
            if self.try_partial_swap(col_index, column, qubit_index):
                self.execute_partial_swap_gate(atom, qubit_index, col_index)

                self.devices['AOD'].col_status[col_index] = Status.Entangling
                self.atom_entanglement_bools[atom.atom_id] = 1
                self.has_entang_flag = True
                swaps.append(qubit_index)
                
                continue

            elif next_gate.operation.name == 'cz': 
                other_qubit_index = next_gate.qubits[1]._index if next_gate.qubits[0]._index == qubit_index else next_gate.qubits[0]._index
                if self.zone_check(qubit_index, other_qubit_index) and self.try_cz_gate(col_index, column, next_gate, qubit_index, other_qubit_index):
                    executed_gates.append(self.executable_gates[qubit_index])
                    self.devices['AOD'].col_status[col_index] = Status.Entangling
                    self.atom_entanglement_bools[atom.atom_id] = 1
                    self.has_entang_flag = True
                    continue
                    

            if self.needs_swap(qubit_index):
                if self.find_slm_swap_qubit(col_index, column, next_gate, qubit_index):

                    flag = self.execute_partial_swap_gate(atom, qubit_index, col_index)

                    self.devices['AOD'].col_status[col_index] = Status.Entangling
                    self.atom_entanglement_bools[atom.atom_id] = 1
                    self.has_entang_flag = True
                    swaps.append(qubit_index)

        #Process if unused:
        if self.devices['AOD'].col_status[col_index] != Status.Entangling:
            self.devices['AOD'].col_status[col_index] = Status.Unused
            if self.iterate_left_to_right == 1:
                direction = 1
            else:
                direction = -1
            #If nothing has been entangled, move this to edge of the right/left storage
            if (col_index == 0 and direction == 1)  or (col_index == len(self.devices['AOD'].col_status) - 1 and direction == -1):
                if direction == 1:
                    left_most_storage_zone = min((zone for zone in self.zones if isinstance(zone, StorageZone) or isinstance(zone, ReadoutZone)),key=lambda z: z.bottom_left_x, default=None)
                    final_loc_x = left_most_storage_zone.bottom_left_x
                    final_loc_y = (left_most_storage_zone.bottom_left_y + left_most_storage_zone.height) / 2
                elif direction == -1:
                    right_most_storage_zone = max((zone for zone in self.zones if isinstance(zone, StorageZone) or isinstance(zone, ReadoutZone)),key=lambda z: z.bottom_left_x, default=None)
                    final_loc_x = right_most_storage_zone.bottom_left_x + right_most_storage_zone.width
                    final_loc_y = (right_most_storage_zone.bottom_left_y + right_most_storage_zone.height) / 2
                self.devices['AOD'].move_column(col_index, final_loc_x - column[0].x, final_loc_y - column[len(column)-1].y)
            #If nothing has been entangled, move this to the right/left storage
            elif self.has_entang_flag == False:
                if direction == 1:
                    cl = self.devices['AOD'].columns[col_index - 1]
                    final_loc_x = cl[0].x + STORAGE_SP
                    final_loc_y = cl[len(cl)-1].y
                elif direction == -1:
                    cl = self.devices['AOD'].columns[col_index + 1]
                    final_loc_x = cl[0].x - STORAGE_SP
                    final_loc_y = cl[len(cl)-1].y
                self.devices['AOD'].move_column(col_index, final_loc_x - column[0].x, final_loc_y - column[len(column)-1].y)
            #Else, move to bottom storage and as close as possible (STORAGE_SP) to the previous column
            else:
                bottom_storage_zone = min((zone for zone in self.zones if isinstance(zone, StorageZone) or isinstance(zone, ReadoutZone)),key=lambda z: z.bottom_left_y, default=None)
                if direction == 1:
                    final_loc_x = self.devices['AOD'].columns[col_index - 1][0].x + STORAGE_SP
                elif direction == -1:
                    final_loc_x = self.devices['AOD'].columns[col_index + 1][0].x - STORAGE_SP
                final_loc_y = bottom_storage_zone.bottom_left_y + bottom_storage_zone.height
                self.devices['AOD'].move_column(col_index, final_loc_x - column[0].x, final_loc_y - column[len(column)-1].y)
            self.spread_atoms(column, STORAGE_SP)
                
        return False

    def is_layer_complete(self, col_index):
        current_col_x = self.devices['AOD'].columns[col_index][0].x
        next_col_index = col_index + (1 if self.iterate_left_to_right else -1)
    
        # Check if there are no more columns to iterate
        if (self.iterate_left_to_right and next_col_index >= len(self.devices['AOD'].columns)) or \
           (not self.iterate_left_to_right and next_col_index < 0):
            return True
    
        next_col_x = self.devices['AOD'].columns[next_col_index][0].x
        next_col_status = self.devices['AOD'].col_status[next_col_index]
    
        if self.iterate_left_to_right:
            return (current_col_x >= self.rightmost_slm_x - 2*MIN_SP or 
                    (next_col_status == Status.Entangling and next_col_x >= self.rightmost_slm_x - 2*MIN_SP))
        else:
            return (current_col_x <= self.leftmost_slm_x + 2*MIN_SP or 
                    (next_col_status == Status.Entangling and next_col_x <= self.leftmost_slm_x + 2*MIN_SP))
    
    def is_valid_qubit_for_entanglement(self, qubit_index):
        if self.executable_gates[qubit_index] == None:
            return False
        elif qubit_index is None or qubit_index >= len(self.executable_gates):
            raise ValueError(f"Qubit index out of range: {qubit_index}")
        else:
            return True
           
    def try_partial_swap(self, col_index, column, qubit_index):
        #If not partial swap in prog, return 0
        if self.partial_swaps[qubit_index][1] == 0:
            return 0
        
        other_qubit_ind = self.partial_swaps[qubit_index][0]
        #has col already been moved? If NOT, try to move column to qubit
        if self.devices['AOD'].col_status[col_index] != Status.Entangling:
            return self.attempt_move(col_index, column, qubit_index, other_qubit_ind)
        #if it has alerady been moved, try to adjust other qubits in row to still be ENTANG_SP away from the rest 
        else:
            return self.attempt_row_adjustment(col_index, column, qubit_index, other_qubit_ind)

    
    def try_cz_gate(self, col_index, column, next_gate, qubit_index, other_qubit_index):
        if self.devices['AOD'].col_status[col_index] != Status.Entangling:
            return self.attempt_move(col_index, column, qubit_index, other_qubit_index)
        #if it has been moved, try to adjust other qubits in row to still be ENTANG_SP away from the rest 
        else:
            return self.attempt_row_adjustment(col_index, column, qubit_index, other_qubit_index)

    
    def attempt_move(self, col_index, column, qubit_index, other_qubit_index):
        # Store original positions
        original_positions = {atom.atom_id: (atom.x, atom.y) for atom in column}
        fl = False
        
        desired_x_distance = self.find_atom_by_id(self.qubit_atom_map[other_qubit_index]).x - self.find_atom_by_id(self.qubit_atom_map[qubit_index]).x - MIN_SP
        desired_y_distance = self.find_atom_by_id(self.qubit_atom_map[other_qubit_index]).y - self.find_atom_by_id(self.qubit_atom_map[qubit_index]).y
        
        #if this is the "first" col, we can move regardless
        if (self.iterate_left_to_right and col_index == 0) or (not self.iterate_left_to_right and col_index == len(self.devices['AOD'].columns)-1):
            self.devices['AOD'].move_column(col_index, desired_x_distance, desired_y_distance)
            fl = True
            
        #first try to move column (works only if no cols are blocking on the moved direction)
        elif self.iterate_left_to_right:
            #If the previously iterated col is in a location closer to the right than the slm qubit, it is blocked off
            if not (self.devices['AOD'].columns[col_index-1][0].x + MIN_SP >= self.find_atom_by_id(self.qubit_atom_map[other_qubit_index]).x):
                self.devices['AOD'].move_column(col_index, desired_x_distance, desired_y_distance)
                fl = True
        else:
            #If the previously iterated col is in a location closer to the left than the slm qubit, it is blocked off
            if not (self.devices['AOD'].columns[col_index+1][0].x - MIN_SP <= self.find_atom_by_id(self.qubit_atom_map[other_qubit_index]).x):
                self.devices['AOD'].move_column(col_index, desired_x_distance, desired_y_distance)
                fl = True

        #Col move failed; return.
        if not fl:
            return False
            
        #then adjust rows; reset atoms/return False if this fails
        if not self.attempt_row_adjustment(col_index, column, qubit_index, other_qubit_index):
            for atom in column:
                atom.x, atom.y = original_positions[atom.atom_id]
            return False
        
        return True

    def attempt_row_adjustment(self, col_index, column, qubit_index, other_qubit_ind):
        original_positions = {atom.atom_id: (atom.x, atom.y) for atom in column}
        
        target_atom = next(atom for atom in column if self.atom_qubit_map[atom.atom_id] == qubit_index)
        other_atom = self.find_atom_by_id(self.qubit_atom_map[other_qubit_ind])

        # Check X-axis reachability
        if abs(target_atom.x - other_atom.x) > 2 * MIN_SP:
            for atom in column:
                atom.x, atom.y = original_positions[atom.atom_id]
            return False  # The other atom is not reachable along the X-axis
        
        # Get the entanglement zone and storage zone
        ent_zone = next(zone for zone in self.zones if isinstance(zone, EntanglementZone))
        storage_zone = next(zone for zone in self.zones if isinstance(zone, StorageZone) and zone.bottom_left_y < ent_zone.bottom_left_y)
    
        # Identify already fixed atoms in the AOD column
        fixed_atoms = [atom for atom in column if self.atom_entanglement_bools[atom.atom_id] == 1]
    
        # Sort atoms in the column by y-coordinate (bottom-up)
        column.sort(key=lambda a: a.y)
    
        # Step 1: Attempt to move the target qubit to its target location
        target_y = other_atom.y
        target_index = column.index(target_atom)
        
        # Check if movement is possible considering fixed atoms
        prev_fixed = next((atom for atom in reversed(column[:target_index]) if atom in fixed_atoms), None)
        next_fixed = next((atom for atom in column[target_index+1:] if atom in fixed_atoms), None)
        
        if (prev_fixed and target_y < prev_fixed.y) or (next_fixed and target_y > next_fixed.y):
            for atom in column:
                atom.x, atom.y = original_positions[atom.atom_id]
            return False  # Can't move target atom due to fixed atoms
        
        target_atom.y = target_y
        fixed_atoms.append(target_atom)
    
        # Step 2: Move bottom-most non-entangling qubits to storage zone
        storage_top = storage_zone.bottom_left_y + storage_zone.height
        bottom_fixed_atom = min((atom for atom in fixed_atoms), key=lambda a: a.y)
        current_y = storage_top - STORAGE_SP / 2
        
        # Iterate from the atom just below the bottom fixed atom
        for atom in reversed([a for a in column if a.y < bottom_fixed_atom.y]):
            if current_y < storage_zone.bottom_left_y:
                for atom in column:
                    atom.x, atom.y = original_positions[atom.atom_id]
                return False  # Not enough space in storage zone
            atom.y = current_y
            current_y -= STORAGE_SP
        
        # New Step: Move top-most non-entangling qubits above the entangling zone
        ent_zone_top = ent_zone.bottom_left_y + ent_zone.height
        top_fixed_atom = max((atom for atom in fixed_atoms), key=lambda a: a.y) if fixed_atoms else None
        current_y = ent_zone_top + STORAGE_SP
        
        # Iterate from the atom just above the top fixed atom
        for atom in [a for a in column if top_fixed_atom is None or a.y > top_fixed_atom.y]:
            atom.y = current_y
            current_y += STORAGE_SP

        # Step 3: Space remaining qubits in the entanglement zone
        relevant_slm_atoms = self.get_relevant_slm_atoms(column[0].x, ent_zone)
        relevant_slm_atoms.sort(key=lambda a: a.y)  # Sort SLM atoms by y-coordinate
        
        # Gather groups of non-placed AOD atoms bounded by fixed atoms
        groups = []
        current_group = []
        lower_bound = None
        
        for atom in column:
            if ent_zone.bottom_left_y <= atom.y < ent_zone.bottom_left_y + ent_zone.height:
                if atom in fixed_atoms:
                    if lower_bound is not None:
                        if current_group:
                            groups.append((lower_bound, current_group, atom))
                            current_group = []
                        lower_bound = atom
                elif lower_bound is not None:
                    current_group.append(atom)
        
        # Find spaces and place non-placed atoms
        for lower_bound, group, upper_bound in groups:
            available_spaces = []
            current_y = lower_bound.y + ENTANG_SP
        
            # Find available spaces considering SLM atoms
            relevant_slm_in_range = [atom for atom in relevant_slm_atoms 
                                     if lower_bound.y < atom.y < upper_bound.y]
            
            for i in range(len(relevant_slm_in_range) + 1):
                if i == 0:
                    space_bottom = lower_bound.y + ENTANG_SP
                else:
                    space_bottom = relevant_slm_in_range[i-1].y + ENTANG_SP
        
                if i == len(relevant_slm_in_range):
                    space_top = upper_bound.y - ENTANG_SP
                else:
                    space_top = relevant_slm_in_range[i].y - ENTANG_SP
        
                space_size = space_top - space_bottom
                num_qubits_fit = space_size // (2 * ENTANG_SP)
                
                if num_qubits_fit > 0:
                    available_spaces.extend([space_bottom + j * (2 * ENTANG_SP) 
                                             for j in range(num_qubits_fit)])
        
            # Place atoms in available spaces
            if len(available_spaces) < len(group):
                for atom in column:
                    atom.x, atom.y = original_positions[atom.atom_id]
                return False  # Not enough space for all atoms
        
            for atom, space in zip(group, available_spaces):
                atom.y = space
        
        return True

    def move_atoms_to_storage(self, atoms, storage_zone):
        atoms.sort(key=lambda a: a.y, reverse=True)  # Sort atoms top to bottom
        
        if not atoms:
            return True # No atoms to move
    
        # Align the top-most atom with the top of the storage zone
        current_y = storage_zone.bottom_left_y + storage_zone.height
    
        for atom in atoms:
            if current_y < storage_zone.bottom_left_y:
                return False
            atom.y = current_y
            current_y -= STORAGE_SP

    def get_relevant_slm_atoms(self, aod_column_x, ent_zone):
        relevant_atoms = []
        for atom_id in self.devices['SLM'].atoms:
            atom = self.find_atom_by_id(atom_id)
            if (isinstance(atom.zone, EntanglementZone) and 
                abs(atom.x - aod_column_x) <= ENTANG_SP):
                relevant_atoms.append(atom)
        return relevant_atoms

    #Return True if the qubits do NOT share zones
    def zone_check(self, qubit_index, other_qubit_ind):
        atom =  self.find_atom_by_id(self.qubit_atom_map[qubit_index])
        other_atom = self.find_atom_by_id(self.qubit_atom_map[other_qubit_ind])

        if atom.zone != other_atom.zone:
            return True
        else:
            return False


    def debug_print_mappings(self):
        print("\n--- Qubit-Atom Mapping Debug ---")
        for qubit, atom_id in self.qubit_atom_map.items():
            atom = self.find_atom_by_id(atom_id)
            device = atom.device.name if atom.device else "Unknown"
            zone = atom.zone.__class__.__name__ if atom.zone else "Unknown"
            print(f"Qubit {qubit}: Atom {atom_id}, Device: {device}, Zone: {zone}")
        print("--- End of Mapping Debug ---\n")
    
    def execute_partial_swap_gate(self, atom, qubit_index, col_index):

        other_qubit_index = self.partial_swaps[qubit_index][0]
        other_atom = self.find_atom_by_id(self.qubit_atom_map[other_qubit_index])
        
        #First Increment the counter in partial_swaps for BOTH qubits.
        self.partial_swaps[qubit_index][1] += 1
        self.partial_swaps[other_qubit_index][1] += 1

        #Then, check if the counter hits 3, in which case we swap the mappings for the qubits and reset the partial swap values to [-1,0]
        if self.partial_swaps[qubit_index][1] == 3:
            
            self.atom_qubit_map[atom.atom_id] = other_qubit_index
            self.atom_qubit_map[other_atom.atom_id] = qubit_index
            self.qubit_atom_map[qubit_index] = other_atom.atom_id
            self.qubit_atom_map[other_qubit_index] = atom.atom_id
            self.partial_swaps[qubit_index] = [-1,0]
            self.partial_swaps[other_qubit_index] = [-1,0]

    def needs_swap(self, qubit_index):
        if self.executable_gates[qubit_index] is None:
            if qubit_index == 0:
                return False
        
        next_gate = self.qc[self.executable_gates[qubit_index]]
        if next_gate.operation.name != 'cz':
            return False
        
        other_qubit_ind = next_gate.qubits[0]._index if next_gate.qubits[1]._index == qubit_index else next_gate.qubits[1]._index

        #Check if THIS qubit is already in partial SWAP
        if self.partial_swaps[qubit_index][1] > 0:
            return False
        
        # Check if the other qubit is already involved in a swap
        if self.partial_swaps[other_qubit_ind][1] > 0:
            return False
        
        # Check if the qubits are in different zones
        return not self.zone_check(qubit_index, other_qubit_ind)
    
    #find slm qubit to SWAP with (ONLY no gates/also needs SWAP)
    #REMEMBER: ONLY start the SWAP if an accessible qubit is found (We might have moved this column already for this layer so rows might be an issue).
    #Moves col to location as well
    def find_slm_swap_qubit(self, col_index, column, next_gate, qubit_index):
        accessible_slm_atoms = self.get_accessible_slm_atoms(col_index)
        accessible_slm_atoms.sort(key=lambda atom: atom.x, reverse=not self.iterate_left_to_right)
        
        other_qubit_ind = None
        
        for slm_atom in accessible_slm_atoms:
            slm_qubit_index = self.atom_qubit_map[slm_atom.atom_id]
            
            # Can't use qubit already swapping
            if self.partial_swaps[slm_qubit_index][0] != -1:
                continue

            if self.executable_gates[slm_qubit_index] is None:
                other_qubit_ind = slm_qubit_index
                break
            
            # Check if the SLM qubit needs to execute a SWAP
            if self.needs_swap(slm_qubit_index):
                other_qubit_ind = slm_qubit_index
                break
            
            # Fallback: choose the first available SLM qubit not executing a U3 gate
            elif other_qubit_ind is None and self.executable_gates[slm_qubit_index] is not None:
                if self.qc[self.executable_gates[slm_qubit_index]].operation.name != 'u3':
                    other_qubit_ind = slm_qubit_index
                    break
        
        if other_qubit_ind is None:
            return False
        
        # Move the AOD column if a suitable SLM qubit is found
        if self.attempt_move(col_index, column, qubit_index, other_qubit_ind):
            # Set swap buddies and initialize swap stage
            self.partial_swaps[qubit_index] = [other_qubit_ind, 0]
            self.partial_swaps[other_qubit_ind] = [qubit_index, 0]
            return True
        else:
            return False
    
    def get_accessible_slm_atoms(self, col_index):
        ent_zone = next(zone for zone in self.zones if isinstance(zone, EntanglementZone))
        slm_atoms = [self.find_atom_by_id(atom_id) for atom_id in self.devices['SLM'].atoms 
                     if isinstance(self.find_atom_by_id(atom_id).zone, EntanglementZone)]
        
        if self.iterate_left_to_right:
            if col_index == 0:
                return slm_atoms  # All SLM atoms are accessible for the first column
            prev_column = self.devices['AOD'].columns[col_index - 1]
            return [atom for atom in slm_atoms if atom.x > prev_column[0].x + MIN_SP]
        else:
            if col_index == len(self.devices['AOD'].columns) - 1:
                return slm_atoms  # All SLM atoms are accessible for the last column when moving right to left
            next_column = self.devices['AOD'].columns[col_index + 1]
            return [atom for atom in slm_atoms if atom.x < next_column[0].x - MIN_SP]


    def validate_partial_swaps(self):
        errors = []
        for qubit_index, (partner, stage) in enumerate(self.partial_swaps):
            if partner != -1:  # This qubit is involved in a swap
                # if stage == 0:
                #     errors.append(f"Qubit {qubit_index} has a partner but stage is 0")
                # elif stage > 3:
                #     errors.append(f"Qubit {qubit_index} has an invalid stage: {stage}")
                
                # Check if partner qubit correctly references back
                if self.partial_swaps[partner][0] != qubit_index:
                    errors.append(f"Qubit {qubit_index}'s partner {partner} does not reference back")
                
                # Check if stages match
                if self.partial_swaps[partner][1] != stage:
                    errors.append(f"Swap stages don't match for qubits {qubit_index} and {partner}")
                
                # Check if qubits are in different devices
                qubit_device = self.find_atom_by_id(self.qubit_atom_map[qubit_index]).device
                partner_device = self.find_atom_by_id(self.qubit_atom_map[partner]).device
                if qubit_device == partner_device:
                    errors.append(f"Qubits {qubit_index} and {partner} are in the same device: {qubit_device.name}")
            
            elif stage != 0:
                errors.append(f"Qubit {qubit_index} has no partner but stage is {stage}")
    
        # Check for any qubit involved in multiple swaps
        partners = [p for p, _ in self.partial_swaps if p != -1]
        if len(partners) != len(set(partners)):
            errors.append("Some qubits are involved in multiple swaps")
    
        if errors:
            print("Errors found in partial_swaps state:")
            for error in errors:
                print(f"  - {error}")
            print("Current partial_swaps state:")
            print(self.partial_swaps)
            raise ValueError("Inconsistent partial_swaps state")
        else:
            print("partial_swaps state is valid")
    
    def is_swap_accessible(self, col_index, slm_atom):
        # Check if the SLM atom is accessible based on the current AOD column position
        aod_column = self.devices['AOD'].columns[col_index]
        if self.iterate_left_to_right:
            return slm_atom.x > aod_column[0].x
        else:
            return slm_atom.x < aod_column[0].x

    def spread_atoms(self, column, spacing, atom=None):
        if atom != None:
            for at in range(len(column)):
                if column[at] == atom:
                    atom_ind = at
                    break
            for at_ind in range(atom_ind+1,len(column)):
                at = column[at_ind]
                at.y = column[at_ind-1].y + spacing
            for at_ind in range(atom_ind-1,-1,-1):
                at = column[at_ind]
                at.y = column[at_ind+1].y - spacing
        else:
            for at_ind in range(len(column)-2,-1,-1):
                at = column[at_ind]
                at.y = column[at_ind+1].y - spacing

    def simulate_measurement(self):
        # Find the readout zone
        readout_zone = next((zone for zone in self.zones if isinstance(zone, ReadoutZone)), None)
        if not readout_zone:
            print("Readout zone not found.")
            return

        # Find the AOD column closest to the readout zone
        closest_column = min(self.devices['AOD'].columns, key=lambda col: abs(col[0].x - readout_zone.bottom_left_x))
        #Find the AOD column farthest to the readout zone
        farthest_column = max(self.devices['AOD'].columns, key=lambda col: abs(col[0].x - readout_zone.bottom_left_x))

        #Move furthest column to the readout zone
        lng_dist = self.devices['AOD'].move_column(self.devices['AOD'].columns.index(farthest_column), readout_zone.bottom_left_x - farthest_column[0].x, readout_zone.bottom_left_y - farthest_column[0].y)

        
        bottom_storage_zone = min((zone for zone in self.zones if isinstance(zone, StorageZone) or isinstance(zone, ReadoutZone)),key=lambda z: z.bottom_left_y, default=None)
        if not bottom_storage_zone:
            print("Error: No storage zone found.")
            return
        #Now move the furthest column to the rightside in bottom storage zone
        lng_dist_2 = self.devices['AOD'].move_column(self.devices['AOD'].columns.index(farthest_column), bottom_storage_zone.bottom_left_x - farthest_column[0].x, bottom_storage_zone.bottom_left_y - farthest_column[0].y)

        return lng_dist + lng_dist_2
    

algos = [
    # 'bv_n70',
    # 'cat_n65',
    # 'dnn_n51',
    # 'ghz_n78',
    # 'ising_n98',
    # 'knn_n129',
    # 'qft_n63',
    # 'qugan_n111',
    # 'swap_test_n115',
    'tfim_n128'
    # 'wstate_n76'
]

for algo in algos:
    print("Running "+algo+"...")
    qasm_file_path = './benchmarks/'+algo+'.qasm'
    #Double storage
    zone_specs = [
        {'type': 'StorageZone', 'bottom_left_x': 90, 'bottom_left_y': 0, 'width': 190, 'height': 50},
        {'type': 'EntanglementZone', 'bottom_left_x': 90, 'bottom_left_y': 60, 'width': 190, 'height': 130, 'col_size': 4},
        {'type': 'ReadoutZone', 'bottom_left_x': 0, 'bottom_left_y': 60, 'width': 80, 'height': 130},
        {'type': 'StorageZone', 'bottom_left_x': 290, 'bottom_left_y': 60, 'width': 80, 'height':130}
    ]
    grid = Grid(zone_specs, 0, qasm_file_path)
    # grid.print_grid_by_atoms()
    ret_vals = grid.return_vals()

    if not os.path.exists('results'):
        os.makedirs('results')
        
    with open(f'results/{algo}.pkl', 'wb') as f:
        pickle.dump(ret_vals, f)

    print(f"Results for {algo} saved.")
