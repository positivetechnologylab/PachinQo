from grid import Grid
import os
import pickle

algos = [
    'bv_n70',
    'cat_n65',
    'dnn_n51',
    'ghz_n78',
    'ising_n98',
    'knn_n129',
    'qft_n63',
    'qugan_n111',
    'swap_test_n115',
    'tfim_n128',
    'wstate_n76'
]

for algo in algos:
    print("Running "+algo+"...")
    qasm_file_path = '../benchmarks/'+algo+'.qasm'
    zone_specs = [
        {'type': 'StorageZone', 'bottom_left_x': 90, 'bottom_left_y': 0, 'width': 190, 'height': 50},
        {'type': 'EntanglementZone', 'bottom_left_x': 90, 'bottom_left_y': 60, 'width': 190, 'height': 130, 'col_size': 4},
        {'type': 'ReadoutZone', 'bottom_left_x': 0, 'bottom_left_y': 60, 'width': 80, 'height': 130},
        {'type': 'StorageZone', 'bottom_left_x': 290, 'bottom_left_y': 60, 'width': 80, 'height':130}
    ]
    grid = Grid(zone_specs, 0, qasm_file_path)
    ret_vals = grid.return_vals()

    if not os.path.exists('../results'):
        os.makedirs('../results')
        
    with open(f'../results/{algo}.pkl', 'wb') as f:
        pickle.dump(ret_vals, f)

    print(f"Results for {algo} saved.")