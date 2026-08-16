import h5py
import numpy as np

DATASET = r"data\raw\robomimic\transport\low_dim_v15.hdf5"

def analyze_case(idx: int):
    with h5py.File(DATASET, "r") as f:
        demo = f["data"][f"demo_{idx}"]
        obj = demo["obs"]["object"][:]
        eef0 = demo["obs"]["robot0_eef_pos"][:]
        
    lid = obj[:, 14:17]
    
    print(f"=== Demo {idx} LID ===")
    
    lid_z0 = lid[0, 2]
    lift_idx = None
    for i in range(len(lid)):
        if lid[i, 2] > lid_z0 + 0.02:
            lift_idx = i
            break
            
    if lift_idx is None:
        print("Lid never lifted?")
        return
        
    print(f"Lid lifted at step {lift_idx}.")
    grasp_idx = lift_idx - 10
    
    eef = eef0[grasp_idx]
    li = lid[grasp_idx]
    
    print(f"At grasp time (step {grasp_idx}):")
    print(f"  Lid pos: {li}")
    print(f"  EEF0 pos:  {eef}")
    print(f"  Offset (eef - lid): {eef - li}")

if __name__ == "__main__":
    analyze_case(0)
    analyze_case(1)
