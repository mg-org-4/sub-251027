#!/usr/bin/env python3
"""
Check what tensors are in the 5D fix files
"""
import os
import sys
from safetensors.torch import load_file

def check_fix_file(path):
    """Check what tensors are in a fix file"""
    if not os.path.isfile(path):
        print(f"Error: File not found: {path}")
        return
    
    print(f"Checking fix file: {path}")
    print("=" * 60)
    
    try:
        tensors = load_file(path)
        print(f"Total tensors: {len(tensors)}")
        print()
        
        # Check for patch_embedding specifically
        patch_tensors = [name for name in tensors.keys() if "patch" in name.lower()]
        if patch_tensors:
            print("Found patch_embedding tensors:")
            for name in patch_tensors:
                tensor = tensors[name]
                print(f"  {name}: {tensor.shape} ({tensor.dtype})")
        else:
            print("No patch_embedding tensors found in this fix file.")
        
        print("\nAll tensors in fix file:")
        for name, tensor in tensors.items():
            print(f"  {name}: {tensor.shape} ({tensor.dtype})")
            
    except Exception as e:
        print(f"Error reading fix file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check all WAN fix files
    fix_files = [
        "fix_5d_tensors_wan.safetensors",
        "fix_5d_tensors_wan_a7a8fac1.safetensors"
    ]
    
    for fix_file in fix_files:
        fix_path = os.path.join(script_dir, fix_file)
        if os.path.exists(fix_path):
            check_fix_file(fix_path)
            print("\n" + "="*60 + "\n")
        else:
            print(f"Fix file not found: {fix_path}")
