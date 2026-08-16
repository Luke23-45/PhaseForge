import h5py
import numpy as np

DATASET = r"data\raw\robomimic\transport\low_dim_v15.hdf5"

def analyze_case(idx: int):
    with h5py.File(DATASET, "r") as f:
        demo = f["data"][f"demo_{idx}"]
        obj = demo["obs"]["object"][:]
        eef1 = demo["obs"]["robot1_eef_pos"][:]
        
    trash = obj[:, 7:10]
    
    print(f"=== Demo {idx} ===")
    
    trash_z0 = trash[0, 2]
    lift_idx = None
    for i in range(len(trash)):
        if trash[i, 2] > trash_z0 + 0.02:
            lift_idx = i
            break
            
    if lift_idx is None:
        print("Trash never lifted?")
        return
        
    print(f"Trash lifted at step {lift_idx}.")
    grasp_idx = lift_idx - 10
    
    eef = eef1[grasp_idx]
    tr = trash[grasp_idx]
    
    print(f"At grasp time (step {grasp_idx}):")
    print(f"  Trash pos: {tr}")
    print(f"  EEF1 pos:  {eef}")
    print(f"  Offset (eef - trash): {eef - tr}")
    print(f"  Controller TRASH_GRASP_Z_OFFSET = 0.005")

if __name__ == "__main__":
    analyze_case(0)
    analyze_case(1)
