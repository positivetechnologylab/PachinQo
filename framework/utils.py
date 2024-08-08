from qiskit import QuantumCircuit
from zones import StorageZone, EntanglementZone, ReadoutZone
from devices import AOD, SLM
from atom import Atom

def load_qasm_to_circuit(qasm_file_path):
    quantum_circuit = QuantumCircuit.from_qasm_file(qasm_file_path)
    return quantum_circuit

def debug_print_atom_info(qubit_atom_map, find_atom_by_id_func):
    print("\n--- Atom Information Debug ---")
    print("Qubit Index | Atom ID | Device | Zone | Position")
    print("-" * 50)
    sorted_qubit_indices = sorted(qubit_atom_map.keys())
    for qubit_index in sorted_qubit_indices:
        atom_id = qubit_atom_map[qubit_index]
        atom = find_atom_by_id_func(atom_id)
        
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

def debug_print_mappings(qubit_atom_map, find_atom_by_id_func):
    print("\n--- Qubit-Atom Mapping Debug ---")
    for qubit, atom_id in qubit_atom_map.items():
        atom = find_atom_by_id_func(atom_id)
        if atom is None:
            print(f"Qubit {qubit}: Atom {atom_id}, Device: Not Found, Zone: Not Found")
            continue
        device = atom.device.name if atom.device else "Unknown"
        zone = atom.zone.__class__.__name__ if atom.zone else "Unknown"
        print(f"Qubit {qubit}: Atom {atom_id}, Device: {device}, Zone: {zone}")
    print("--- End of Mapping Debug ---\n")


def validate_partial_swaps(qubit_atom_map, partial_swaps, find_atom_by_id_func):
    errors = []
    for qubit_index, (partner, stage) in enumerate(partial_swaps):
        if partner != -1:           
            # Check if partner qubit correctly references back
            if partial_swaps[partner][0] != qubit_index:
                errors.append(f"Qubit {qubit_index}'s partner {partner} does not reference back")
            
            # Check if stages match
            if partial_swaps[partner][1] != stage:
                errors.append(f"Swap stages don't match for qubits {qubit_index} and {partner}")
            
            # Check if qubits are in different devices
            qubit_atom = find_atom_by_id_func(qubit_atom_map[qubit_index])
            partner_atom = find_atom_by_id_func(qubit_atom_map[partner])
            if qubit_atom and partner_atom:
                qubit_device = qubit_atom.device
                partner_device = partner_atom.device
                if qubit_device == partner_device:
                    errors.append(f"Qubits {qubit_index} and {partner} are in the same device: {qubit_device.name}")
            else:
                errors.append(f"Could not find atoms for qubits {qubit_index} and/or {partner}")
        
        elif stage != 0:
            errors.append(f"Qubit {qubit_index} has no partner but stage is {stage}")

    # Check for any qubit involved in multiple swaps
    partners = [p for p, _ in partial_swaps if p != -1]
    if len(partners) != len(set(partners)):
        errors.append("Some qubits are involved in multiple swaps")

    if errors:
        print("Errors found in partial_swaps state:")
        for error in errors:
            print(f"  - {error}")
        print("Current partial_swaps state:")
        print(partial_swaps)
        raise ValueError("Inconsistent partial_swaps state")
    else:
        print("partial_swaps state is valid")