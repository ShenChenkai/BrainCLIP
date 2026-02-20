import numpy as np
import os
import sys

def verify_data():
    here = os.path.dirname(os.path.realpath(__file__))
    fc_path = os.path.join(here, 'FC.npy')
    y_path = os.path.join(here, 'Y_positive.npy')
    merged_path = os.path.join(here, 'FC_Y.npy')

    print(f"Checking data in: {here}")
    
    # 1. Load Raw Files
    if not os.path.exists(fc_path) or not os.path.exists(y_path):
        print("❌ Missing FC.npy or Y_positive.npy")
        return

    print("Loading FC.npy and Y_positive.npy...")
    try:
        fc = np.load(fc_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)
        
        print(f"\n[Raw Data Stats]")
        print(f"FC shape: {fc.shape}")
        print(f"Y shape:  {y.shape}")
        
        # Check Length
        if len(fc) != len(y):
            print(f"❌ SIZE MISMATCH: FC has {len(fc)} samples, Y has {len(y)} samples.")
        else:
            print(f"✅ Sample count matches: {len(fc)}")

        # Check Label Distribution
        y_flat = y.reshape(-1)
        unique, counts = np.unique(y_flat, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")
        
        # Check for NaNs
        if np.isnan(fc).any():
            print("⚠️ Warning: FC contains NaN values!")
        else:
            print("✅ FC has no NaNs")

    except Exception as e:
        print(f"❌ Error loading raw files: {e}")
        return

    # 2. Check Merged File
    if os.path.exists(merged_path):
        print(f"\nLoading merged file: {merged_path}...")
        try:
            merged = np.load(merged_path, allow_pickle=True).item()
            
            m_fc = merged['corr']
            m_y = merged['label']
            
            print(f"[Merged Data Stats]")
            print(f"Merged FC shape: {m_fc.shape}")
            print(f"Merged Y shape:  {m_y.shape}")
            
            # Consistency Check
            if np.array_equal(fc, m_fc):
                print("✅ Merged 'corr' matches original FC.npy")
            else:
                print("⚠️ Merged 'corr' DOES NOT match original FC.npy (Check logic in merge_data.py)")
                
                
            # Label Consistency Check
            # Note: merge_data.py might transform labels to 0/1
            if np.array_equal(y_flat, m_y):
                 print("✅ Merged 'label' matches original Y_positive.npy exactly")
            else:
                print("ℹ️ Merged 'label' differs from original. Checking if it's just 0/1 mapping...")
                # Simulate the mapping done in merge_data.py
                y_mapped = (y_flat != 0).astype(np.int64)
                if np.array_equal(y_mapped, m_y):
                     print("✅ Merged 'label' matches after (label != 0) transformation.")
                else:
                     print("❌ Merged 'label' has unexplained differences!")

        except Exception as e:
             print(f"❌ Error inspecting merged file: {e}")
    else:
        print("\nℹ️ Merged file 'FC_Y.npy' not found, skipping consistency check.")

if __name__ == "__main__":
    verify_data()
