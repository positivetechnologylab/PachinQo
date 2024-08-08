# PachinQo

This project simulates a compilation technique for Rydberg atom quantum computing operations.

## Project Structure

The project consists of the following Python files in the framework directory:

- `main.py`: The entry point for the code.
- `grid.py`: Contains the core logic for the atom grid simulation.
- `atom.py`: Defines the Atom class.
- `devices.py`: Contains classes for optical devices (AOD, SLM).
- `zones.py`: Defines different zone classes (Entanglement, Storage, Readout).
- `utils.py`: Utility functions for debugging and validation.
- `constants.py`: Defines spacing constants for atoms in each zone.

## File Descriptions

### main.py

The main entry point for running the simulation. It initializes the grid and executes the simulation.

### grid.py

Contains the `Grid` class, which is the core of the simulation. It manages the atoms, zones, and devices, and implements the main simulation logic.

### atom.py

Defines the `Atom` class, representing individual atoms in the grid with properties such as position, zone, and associated device.

### devices.py

Contains classes for optical devices:
- `OpticalDevice`: Base class for devices.
- `AOD` (Acousto-Optic Deflector): Manages mobile columns of atoms.
- `SLM` (Spatial Light Modulator): Manages fixed atom traps in the grid.

### zones.py

Defines different types of zones in the grid:
- `Zone`: Base class for all zones.
- `StorageZone`: For storing atoms.
- `EntanglementZone`: Where atom entanglement occurs.
- `ReadoutZone`: For atom readout operations.

### utils.py

Contains utility functions for debugging and validating the grid state:
- `debug_print_atom_info`: Prints detailed information about atoms.
- `debug_print_mappings`: Displays qubit-to-atom mappings.
- `validate_partial_swaps`: Validates the state of partial swap operations.

### constants.py

Defines constants used throughout the project, particularly spacing values for atoms in different zones.

## Usage

To run the simulation:

```bash
python main.py
