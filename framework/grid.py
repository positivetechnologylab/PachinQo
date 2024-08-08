from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from enum import Enum
import time
import pickle
import os

from constants import STORAGE_SP, ENTANG_SP, MIN_SP
from devices import AOD, SLM
from zones import StorageZone, EntanglementZone, ReadoutZone
from atom import Atom
from utils import load_qasm_to_circuit, debug_print_atom_info, debug_print_mappings, validate_partial_swaps
"""
Status enum for each AOD column.
"""
class Status(Enum):
    Waiting = 0
    Unused = 1
    Entangling = 2

class Grid:
    def __init__(self, zone_specs, start_zone_ind, qasm_file_path, num_atoms=280):
        self.zones = []
        self.devices = {'AOD': AOD(), 'SLM': SLM()}
        self.initialize_zones(zone_specs)
        self.initialize_atoms(num_atoms,self.zones[start_zone_ind])
        
        self.qubit_atom_map = {} #qubit ID to atom ID
        self.atom_qubit_map = {} #atom ID to qubit ID
        self.qc = None
        self.iterate_left_to_right = True #direction used to swap direction for 2-cache architecture compilation
        self.executable_gates = None
        self.has_entang_flag = False

        self.same_zone_czs = []
        
        #list of counters to see stage SWAP is in (0,1,2 -> number of CZs that have occured; at 3, resets)
        #each element of form (qubit_id, stage), qubit_id is the other qubit being swapped with; stage is what stage in the partial swap it is in
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
        
        while any(gate is not None for gate in self.executable_gates):
            eg,swaps,u3s,layer_dist_moved,slm_swaps = self.compile_layer()
            all_egs.extend(eg)
            eg_layers.append(eg)
            swap_layers.append(swaps)
            self.all_layers.append([eg,swaps,u3s,slm_swaps,layer_dist_moved])
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
        
    def print_grid_by_atoms(self):
        zone_background_colors = {
            'StorageZone': 'skyblue',
            'EntanglementZone': 'lightgreen',
            'ReadoutZone': 'salmon',
        }
        
        fig, ax = plt.subplots(figsize=(10, 8))
    
        for zone in self.zones:
            background_color = zone_background_colors.get(zone.__class__.__name__, 'grey')
            zone_rect = patches.Rectangle((zone.bottom_left_x, zone.bottom_left_y), zone.width, zone.height, linewidth=1, edgecolor='r', facecolor=background_color, alpha=0.3)
            ax.add_patch(zone_rect)
    
        for qubit, atom_id in self.qubit_atom_map.items():
            atom = self.find_atom_by_id(atom_id)
            if atom and atom.device != self.devices['AOD']:
                color = 'blue' if atom.device and atom.device.name == 'SLM' else 'black'
                ax.scatter(atom.x, atom.y, color=color, s=4)
                qubit_id = self.atom_qubit_map[atom.atom_id]
                ax.text(atom.x, atom.y + 3, str(qubit_id), color=color, ha='center', va='bottom')
    
        if 'AOD' in self.devices:
            for column in self.devices['AOD'].columns:
                for atom in column:
                    if atom:
                        ax.scatter(atom.x, atom.y, color='red', s=8)
                        qubit_id = self.atom_qubit_map[atom.atom_id]
                        ax.text(atom.x, atom.y - 9, str(qubit_id), color='red', ha='center', va='bottom')
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.axis('equal')
        plt.show()

    
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

    def update_slm_bounds(self):
        ent_zone = next(zone for zone in self.zones if isinstance(zone, EntanglementZone))
        slm_atoms = [self.find_atom_by_id(atom_id) for atom_id in self.devices['SLM'].atoms 
                     if isinstance(self.find_atom_by_id(atom_id).zone, EntanglementZone)]
        if slm_atoms:
            self.leftmost_slm_x = min(atom.x for atom in slm_atoms)
            self.rightmost_slm_x = max(atom.x for atom in slm_atoms)
    
    #Maps qubits defined from qasm file to atoms in the grid.
    def qubit_to_atom(self, quantum_circuit):
        self.qubit_atom_map = {}
        num_qubits = quantum_circuit.num_qubits

        for qubit_index in range(num_qubits):
            atom_id = qubit_index
            self.qubit_atom_map[qubit_index] = atom_id

        self.same_zone_czs = [0] * len(self.qubit_atom_map)
        self.partial_swaps = [[-1, 0] for _ in range(len(self.qubit_atom_map))]

        unmapped_atoms = set(atom_id for zone in self.zones for atom_id in zone.atoms if atom_id not in self.qubit_atom_map.values())

        for zone in self.zones:
            zone.atoms = {atom_id: atom for atom_id, atom in zone.atoms.items() if atom_id not in unmapped_atoms}
        self.atom_qubit_map = {atom_id: qubit for qubit, atom_id in self.qubit_atom_map.items()}

    #Finds the nearest available site in the target zone based on grid coords
    def get_nearest_available_site(self, current_zone, target_zone, atom_x, atom_y):
        available_sites = [site for site in target_zone.sites if not site['occupied']]
        if not available_sites:
            return None

        nearest_site = min(
            available_sites,
            key=lambda site: math.dist(
                [atom_x, atom_y],
                site['coords'])
            )
        return nearest_site

    #move atom to entanglement zone slm's nearest available site
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

        distance_moved = math.sqrt((nearest_site['coords'][0] - atom.x) ** 2 + (nearest_site['coords'][1] - atom.y) ** 2)

        current_zone.remove_atom(atom_id)
        atom.x, atom.y = nearest_site['coords']
        atom.zone = target_zone
        atom.device = self.devices['SLM']
        target_zone.add_atom(atom, nearest_site)

        return distance_moved

    #initialize list of gate index pointers that point to the next gate a qubit needs to execute in the quantum circuit 
    def initialize_executable_gates(self):
        self.executable_gates = [None] * self.qc.num_qubits
        for qubit_index in range(self.qc.num_qubits):
            for gate_index, (instruction, qargs, _) in enumerate(self.qc.data):
                if instruction.name == 'u3' and any(qarg._index == qubit_index for qarg in qargs):
                    self.executable_gates[qubit_index] = gate_index
                    self.same_zone_czs[qubit_index] = 0
                    break
                elif instruction.name == 'cz' and any(qarg._index == qubit_index for qarg in qargs):
                    
                    self.executable_gates[qubit_index] = gate_index
                    atoms = []
                    for qarg in qargs:
                        atoms.append(self.find_atom_by_id(self.qubit_atom_map[qarg._index]))
                    if atoms[0].device == atoms[1].device:
                        self.same_zone_czs[qubit_index] = 1
                    break

    #move atoms to appropriate zones
    def reorder_atoms(self, qasm_file_path, czs):
        quantum_circuit = load_qasm_to_circuit(qasm_file_path)
        self.qc = quantum_circuit
        observed_qubits = set()
        compiled_cz_gates = []
        final_states = {}
        max_distance_moved = 0 

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
            available_sites = any(zn.has_unoccupied_sites() for zn in self.zones if isinstance(zn, EntanglementZone))
            
            if not any(observed_status) and available_sites:
                dist = self.move_to_ent(atom_objects[0].atom_id)
            elif observed_status[0] != observed_status[1] and available_sites:
                unobserved_atom = atom_objects[0] if not observed_status[0] else atom_objects[1]
                observed_atom = atom_objects[0] if observed_status[0] else atom_objects[1]
                if observed_atom.zone == ent_zone:
                    pass
                else:
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

    #Locate the atom object by its ID
    def find_atom_by_id(self, atom_id):
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

        observed_atoms_in_storage = [self.qubit_atom_map[qubit] for qubit in observed_qubits if qubit in self.qubit_atom_map and self.find_atom_by_id(self.qubit_atom_map[qubit]).zone == storage_zone]

        unoccupied_sites = [site['coords'] for site in storage_zone.sites if not site['occupied']]
        unoccupied_sites.sort(key=lambda x: (x[1], x[0]))

        if not unoccupied_sites:
            print("Error: No unoccupied sites in storage.")
            return

        entanglement_columns_x = [entanglement_zone.bottom_left_x + i * entanglement_zone.spacing for i in range(entanglement_zone.width // entanglement_zone.spacing)]

        for atom_id in observed_atoms_in_storage:
            atom = self.find_atom_by_id(atom_id)
            if atom:
                suitable_sites = [site for site in unoccupied_sites if site[0] in entanglement_columns_x]
                if not suitable_sites:
                    suitable_sites = unoccupied_sites

                chosen_site = suitable_sites[0]
                unoccupied_sites.remove(chosen_site)

                storage_zone.remove_atom(atom.atom_id)
                storage_zone.add_atom(atom, chosen_site)

    def move_storage_atoms_to_AOD(self):
        storage_zone = None
        for zone in self.zones:
            if isinstance(zone, StorageZone) and zone.atoms != {}:
                storage_zone = zone
                break

        if not storage_zone:
            print("No storage zone found.")
            return

        column_width = ENTANG_SP
        num_columns = storage_zone.width // column_width

        columns = [[] for _ in range(num_columns)]
        
        for atom_id, atom in storage_zone.atoms.items():
            column_index = int((atom.x - storage_zone.bottom_left_x) / column_width)
            columns[column_index].append(atom)

        columns = [col for col in columns if col]

        self.devices['SLM'].atoms = [atom for atom in self.devices['SLM'].atoms if atom in self.qubit_atom_map.values()]

        # move each column of atoms to the AOD device
        for column_index, column_atoms in enumerate(columns):
            for atom in column_atoms:
                storage_zone.remove_atom(atom.atom_id)
                self.devices['SLM'].remove_atom(atom)
                self.devices['AOD'].add_atom_to_column(atom, column_index)
                atom.device = self.devices['AOD']
            self.devices['AOD'].col_status.append(Status.Unused)
        self.initialize_executable_gates()
        
    def move_AOD_columns_to_right_storage(self):
        
        right_most_storage_zone = max(
            (zone for zone in self.zones if isinstance(zone, StorageZone) or isinstance(zone, ReadoutZone)), 
            key=lambda z: z.bottom_left_x, default=None
        )
        if not right_most_storage_zone:
            print("Error: No storage zone found.")
            return 0
    
        left_edge_x = right_most_storage_zone.bottom_left_x
        start_y = (right_most_storage_zone.bottom_left_y + right_most_storage_zone.height) / 2
    
        max_distance_moved = 0
        current_x = left_edge_x + STORAGE_SP
    
        for i, column in enumerate(self.devices['AOD'].columns):
            if not column: 
                continue

            new_x = current_x
            new_y = start_y
    
            if new_x + STORAGE_SP > right_most_storage_zone.bottom_left_x + right_most_storage_zone.width:
                print(f"Warning: Column {i} cannot fit in the storage zone. Stopping movement.")
                break
    
            dx = new_x - column[0].x
            dy = new_y - column[0].y
    
            distance_moved = self.devices['AOD'].move_column(i, dx, dy)
            max_distance_moved = max(max_distance_moved, distance_moved)
    
            current_x += STORAGE_SP
    
        return max_distance_moved

    def move_AOD_columns_to_left_storage(self):
        left_most_storage_zone = min(
            (zone for zone in self.zones if isinstance(zone, StorageZone) or isinstance(zone, ReadoutZone)), 
            key=lambda z: z.bottom_left_x, default=None
        )
        if not left_most_storage_zone:
            print("Error: No storage zone found.")
            return 0
    
        right_edge_x = left_most_storage_zone.bottom_left_x + left_most_storage_zone.width
        start_y = (left_most_storage_zone.bottom_left_y + left_most_storage_zone.height) / 2
    
        max_distance_moved = 0
        current_x = right_edge_x - STORAGE_SP
    
        for i in range(len(self.devices['AOD'].columns) - 1, -1, -1):
            column = self.devices['AOD'].columns[i]
            if not column:
                continue
    
            new_x = current_x
            new_y = start_y
    
            if new_x < left_most_storage_zone.bottom_left_x:
                print(f"Warning: Column {i} cannot fit in the storage zone. Stopping movement.")
                break
    
            dx = new_x - column[0].x
            dy = new_y - column[0].y
    
            distance_moved = self.devices['AOD'].move_column(i, dx, dy)
            max_distance_moved = max(max_distance_moved, distance_moved)
    
            current_x -= STORAGE_SP
    
        return max_distance_moved
        
    #Given list of executable gates, move pointers after execution
    def update_executable_gates(self, executed_gate_indices):
        for executed_gate_index in executed_gate_indices:
            instruction, qargs, _ = self.qc.data[executed_gate_index]

            for qarg in qargs:
                qubit_index = qarg._index
                next_gate_index = self.find_next_gate_for_qubit(qubit_index, executed_gate_index + 1)
                self.executable_gates[qubit_index] = next_gate_index
            
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
            

    #get next gate given a qubit
    def find_next_gate_for_qubit(self, qubit_index, start_index):
        for index in range(start_index, len(self.qc.data)):
            instruction, qargs, _ = self.qc.data[index]
            if any(qarg._index == qubit_index for qarg in qargs):
                return index
        return None

    #Compiles a single parallelized set of instructions for all qubits in the grid and executes them
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

        has_entang_flag = False
        swap_czs = []
        slm_full_swaps = []
        
        initial_positions = [(column[0].x, column[0].y) if column else (None, None) 
                                    for column in self.devices['AOD'].columns]

        if not self.devices['AOD'].col_status or len(self.devices['AOD'].col_status) != len(self.devices['AOD'].columns):
            self.devices['AOD'].col_status = [Status.Unused] * len(self.devices['AOD'].columns)
        
        entanglement_zone = next((zone for zone in self.zones if isinstance(zone, EntanglementZone)), None)
        if not entanglement_zone:
            print("Entanglement zone not found.")
            return

        # iterate through AOD columns according to direction
        column_indices = range(len(self.devices['AOD'].columns)) if self.iterate_left_to_right else reversed(range(len(self.devices['AOD'].columns)))

        #Execute U3s for qubits in the SLM
        self.process_SLM_U3s(executed_gates)

        col_used_w_slm = False
        
        swaps = []
        #Logic to handle compilation for each column in the AOD
        for col_index in column_indices:
            column = self.devices['AOD'].columns[col_index]
    
            self.process_column_for_entanglement(col_index, column, entanglement_zone, executed_gates, swaps)

            if self.is_layer_complete(col_index):
                break
        
        # if no gates were executed (i.e. no AOD ops possible), perform as many full swaps between SLM and AOD as possible
        slm_swaps = []
        if not executed_gates:
            if not has_entang_flag:
                self.perform_full_swaps(slm_swaps)


        final_positions = [(column[0].x, column[0].y) if column else (None, None) 
                           for column in self.devices['AOD'].columns]

        # get the maximum distance any column has moved (use this to compute move time for the layer)
        max_distance_compiled = 0
        for init_pos, final_pos in zip(initial_positions, final_positions):
            if init_pos != (None, None) and final_pos != (None, None):
                dx = final_pos[0] - init_pos[0]
                dy = final_pos[1] - init_pos[1]
                distance = math.sqrt(dx**2 + dy**2)
                if distance > max_distance_compiled:
                    max_distance_compiled = distance

        self.update_executable_gates(executed_gates)

        # Reset statuses/direction for the next layer compilation
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

    #Performs layer of SLM swaps (i.e. if only SLM qubits need CZ gates but can't execute because they are in the same device)
    def perform_full_swaps(self,slm_swaps):
        unique_slm_qubits = self.get_unique_x_slm_qubits()
        column_indices = range(len(self.devices['AOD'].columns)) if self.iterate_left_to_right else reversed(range(len(self.devices['AOD'].columns)))
    
        for col_index in column_indices:
            if not unique_slm_qubits:
                break
    
            column = self.devices['AOD'].columns[col_index]
            slm_qubit = unique_slm_qubits[0]
            
            # find a suitable AOD qubit in the column
            aod_qubit = self.find_suitable_aod_qubit(column)
            if aod_qubit is None:
                continue
    
            # move the AOD column
            if self.attempt_move(col_index, column, aod_qubit, slm_qubit):
                # if move successful, perform swap
                slm_swaps.append([slm_qubit,aod_qubit])
                self.swap_qubits(slm_qubit, aod_qubit)
                unique_slm_qubits.pop(0)

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
                    # check that the qubit is not already in a partial swap
                    if self.partial_swaps[qubit_index][1] == 0:
                        # check if both qubits are ready to perform the same gate
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
    
    # swap the qubit-atom mappings
    def swap_qubits(self, qubit1, qubit2):
        
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



    #Logic to process qubits in a single AOD col
    def process_column_for_entanglement(self, col_index, column, entanglement_zone, executed_gates, swaps):
        for atom in column:
            qubit_index = self.atom_qubit_map.get(atom.atom_id)
            if not self.is_valid_qubit_for_entanglement(qubit_index):
                continue
            
            next_gate = self.qc[self.executable_gates[qubit_index]]
            #Order of ops: check u3 (no overhead, no complexity), check partial SWAP (if in the middle should try to continue), check CZ, check new SWAP 
            if next_gate.operation.name == 'u3':
                executed_gates.append(self.executable_gates[qubit_index])
                continue

            def both_qubits_ready(gate):
                if gate.operation.name == 'cz':
                    other_qubit_index = gate.qubits[1]._index if gate.qubits[0]._index == qubit_index else gate.qubits[0]._index
                    return self.executable_gates[qubit_index] == self.executable_gates[other_qubit_index]
                    
            if not both_qubits_ready(next_gate):
                continue
            
            #try continuing partial swap if qubit is in the middle of one
            if self.try_partial_swap(col_index, column, qubit_index):
                self.execute_partial_swap_gate(atom, qubit_index, col_index)

                self.devices['AOD'].col_status[col_index] = Status.Entangling
                self.atom_entanglement_bools[atom.atom_id] = 1
                self.has_entang_flag = True
                swaps.append(qubit_index)
                
                continue
            #try executing CZ gate
            elif next_gate.operation.name == 'cz': 
                other_qubit_index = next_gate.qubits[1]._index if next_gate.qubits[0]._index == qubit_index else next_gate.qubits[0]._index
                if self.zone_check(qubit_index, other_qubit_index) and self.try_cz_gate(col_index, column, next_gate, qubit_index, other_qubit_index):
                    executed_gates.append(self.executable_gates[qubit_index])
                    self.devices['AOD'].col_status[col_index] = Status.Entangling
                    self.atom_entanglement_bools[atom.atom_id] = 1
                    self.has_entang_flag = True
                    continue
                    
            #else if it needs swap, find qubit to swap with in SLM
            if self.needs_swap(qubit_index):
                if self.find_slm_swap_qubit(col_index, column, next_gate, qubit_index):

                    flag = self.execute_partial_swap_gate(atom, qubit_index, col_index)

                    self.devices['AOD'].col_status[col_index] = Status.Entangling
                    self.atom_entanglement_bools[atom.atom_id] = 1
                    self.has_entang_flag = True
                    swaps.append(qubit_index)

        #Col processing if no entangling operations occur (moves the column into either cache or storage zone for remainder of layer)
        if self.devices['AOD'].col_status[col_index] != Status.Entangling:
            self.devices['AOD'].col_status[col_index] = Status.Unused
            if self.iterate_left_to_right == 1:
                direction = 1
            else:
                direction = -1
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

    #checks if any additional operations can be performed at all in layer
    def is_layer_complete(self, col_index):
        current_col_x = self.devices['AOD'].columns[col_index][0].x
        next_col_index = col_index + (1 if self.iterate_left_to_right else -1)
    
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
        #if it has alerady been moved, try to adjust rows of qubits in the column without moving column itself
        else:
            return self.attempt_row_adjustment(col_index, column, qubit_index, other_qubit_ind)

    def try_cz_gate(self, col_index, column, next_gate, qubit_index, other_qubit_index):
        if self.devices['AOD'].col_status[col_index] != Status.Entangling:
            return self.attempt_move(col_index, column, qubit_index, other_qubit_index)
        else:
            return self.attempt_row_adjustment(col_index, column, qubit_index, other_qubit_index)

    def attempt_move(self, col_index, column, qubit_index, other_qubit_index):
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

        # check x-axis reachability
        if abs(target_atom.x - other_atom.x) > 2 * MIN_SP:
            for atom in column:
                atom.x, atom.y = original_positions[atom.atom_id]
            return False
        
        ent_zone = next(zone for zone in self.zones if isinstance(zone, EntanglementZone))
        storage_zone = next(zone for zone in self.zones if isinstance(zone, StorageZone) and zone.bottom_left_y < ent_zone.bottom_left_y)
    
        fixed_atoms = [atom for atom in column if self.atom_entanglement_bools[atom.atom_id] == 1]
        column.sort(key=lambda a: a.y)
    
        # attempt to move the target qubit to its target location
        target_y = other_atom.y
        target_index = column.index(target_atom)
        
        # check if movement is possible considering fixed atoms (AOD atoms that are already entangling)
        prev_fixed = next((atom for atom in reversed(column[:target_index]) if atom in fixed_atoms), None)
        next_fixed = next((atom for atom in column[target_index+1:] if atom in fixed_atoms), None)
        
        if (prev_fixed and target_y < prev_fixed.y) or (next_fixed and target_y > next_fixed.y):
            for atom in column:
                atom.x, atom.y = original_positions[atom.atom_id]
            return False
        
        target_atom.y = target_y
        fixed_atoms.append(target_atom)
    
        # move bottom-most non-entangling qubits to storage zone
        storage_top = storage_zone.bottom_left_y + storage_zone.height
        bottom_fixed_atom = min((atom for atom in fixed_atoms), key=lambda a: a.y)
        current_y = storage_top - STORAGE_SP / 2
        
        for atom in reversed([a for a in column if a.y < bottom_fixed_atom.y]):
            if current_y < storage_zone.bottom_left_y:
                for atom in column:
                    atom.x, atom.y = original_positions[atom.atom_id]
                return False
            atom.y = current_y
            current_y -= STORAGE_SP
        
        # move top-most non-entangling qubits above the entangling zone
        ent_zone_top = ent_zone.bottom_left_y + ent_zone.height
        top_fixed_atom = max((atom for atom in fixed_atoms), key=lambda a: a.y) if fixed_atoms else None
        current_y = ent_zone_top + STORAGE_SP
        
        for atom in [a for a in column if top_fixed_atom is None or a.y > top_fixed_atom.y]:
            atom.y = current_y
            current_y += STORAGE_SP

        relevant_slm_atoms = self.get_relevant_slm_atoms(column[0].x, ent_zone)
        relevant_slm_atoms.sort(key=lambda a: a.y)  # Sort SLM atoms by y-coordinate
        
        # gather groups of non-placed AOD atoms bounded by fixed atoms
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
        
        # find available spaces considering SLM atoms and place non-placed atoms
        for lower_bound, group, upper_bound in groups:
            available_spaces = []
            current_y = lower_bound.y + ENTANG_SP
        
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
        
            if len(available_spaces) < len(group):
                for atom in column:
                    atom.x, atom.y = original_positions[atom.atom_id]
                return False
        
            for atom, space in zip(group, available_spaces):
                atom.y = space
        
        return True

    def get_relevant_slm_atoms(self, aod_column_x, ent_zone):
        relevant_atoms = []
        for atom_id in self.devices['SLM'].atoms:
            atom = self.find_atom_by_id(atom_id)
            if (isinstance(atom.zone, EntanglementZone) and 
                abs(atom.x - aod_column_x) <= ENTANG_SP):
                relevant_atoms.append(atom)
        return relevant_atoms

    #returns True if the qubits do NOT share zones
    def zone_check(self, qubit_index, other_qubit_ind):
        atom =  self.find_atom_by_id(self.qubit_atom_map[qubit_index])
        other_atom = self.find_atom_by_id(self.qubit_atom_map[other_qubit_ind])

        if atom.zone != other_atom.zone:
            return True
        else:
            return False
    
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
    
    #find slm qubit to SWAP with
    def find_slm_swap_qubit(self, col_index, column, next_gate, qubit_index):
        accessible_slm_atoms = self.get_accessible_slm_atoms(col_index)
        accessible_slm_atoms.sort(key=lambda atom: atom.x, reverse=not self.iterate_left_to_right)
        
        other_qubit_ind = None
        
        for slm_atom in accessible_slm_atoms:
            slm_qubit_index = self.atom_qubit_map[slm_atom.atom_id]
            
            # Can't use qubit already swapping
            if self.partial_swaps[slm_qubit_index][0] != -1:
                continue

            #prioritize if the SLM qubit is done executing gates
            if self.executable_gates[slm_qubit_index] is None:
                other_qubit_ind = slm_qubit_index
                break
            
            # prioritize if the SLM qubit needs to execute a SWAP
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
                return slm_atoms
            prev_column = self.devices['AOD'].columns[col_index - 1]
            return [atom for atom in slm_atoms if atom.x > prev_column[0].x + MIN_SP]
        else:
            if col_index == len(self.devices['AOD'].columns) - 1:
                return slm_atoms
            next_column = self.devices['AOD'].columns[col_index + 1]
            return [atom for atom in slm_atoms if atom.x < next_column[0].x - MIN_SP]

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
        readout_zone = next((zone for zone in self.zones if isinstance(zone, ReadoutZone)), None)
        if not readout_zone:
            print("Readout zone not found.")
            return

        closest_column = min(self.devices['AOD'].columns, key=lambda col: abs(col[0].x - readout_zone.bottom_left_x))
        farthest_column = max(self.devices['AOD'].columns, key=lambda col: abs(col[0].x - readout_zone.bottom_left_x))

        lng_dist = self.devices['AOD'].move_column(self.devices['AOD'].columns.index(farthest_column), readout_zone.bottom_left_x - farthest_column[0].x, readout_zone.bottom_left_y - farthest_column[0].y)

        bottom_storage_zone = min((zone for zone in self.zones if isinstance(zone, StorageZone) or isinstance(zone, ReadoutZone)),key=lambda z: z.bottom_left_y, default=None)
        if not bottom_storage_zone:
            print("Error: No storage zone found.")
            return

        lng_dist_2 = self.devices['AOD'].move_column(self.devices['AOD'].columns.index(farthest_column), bottom_storage_zone.bottom_left_x - farthest_column[0].x, bottom_storage_zone.bottom_left_y - farthest_column[0].y)

        return lng_dist + lng_dist_2