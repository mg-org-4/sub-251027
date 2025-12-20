#!/usr/bin/env python3
"""
Debug script to examine GGUF files and check for missing tensors
"""
import os
import sys
import gguf

def debug_gguf_file(path):
    """Debug a GGUF file and show tensor information"""
    if not os.path.isfile(path):
        print(f"Error: File not found: {path}")
        return
    
    print(f"Debugging GGUF file: {path}")
    print("=" * 60)
    
    try:
        reader = gguf.GGUFReader(path)
        
        # Show basic info
        print(f"Architecture: {reader.get_architecture()}")
        print(f"Total tensors: {len(reader.tensors)}")
        print()
        
        # Check for specific tensors that WAN models need
        required_tensors = [
            "patch_embedding.weight",
            "blocks.0.self_attn.norm_q.weight", 
            "text_embedding.2.weight",
            "head.modulation"
        ]
        
        print("Checking for required WAN tensors:")
        found_tensors = {}
        for tensor in reader.tensors:
            found_tensors[tensor.name] = tensor
            
        for req_tensor in required_tensors:
            if req_tensor in found_tensors:
                tensor = found_tensors[req_tensor]
                print(f"  ✓ {req_tensor}: {tensor.shape} ({tensor.tensor_type})")
            else:
                print(f"  ✗ {req_tensor}: MISSING")
        
        print()
        print("All tensors in file:")
        print("-" * 40)
        
        # Group tensors by type
        tensor_types = {}
        for tensor in reader.tensors:
            ttype = str(tensor.tensor_type)
            if ttype not in tensor_types:
                tensor_types[ttype] = []
            tensor_types[ttype].append(tensor)
        
        for ttype, tensors in tensor_types.items():
            print(f"\n{ttype} tensors ({len(tensors)}):")
            for tensor in tensors[:10]:  # Show first 10 of each type
                print(f"  {tensor.name}: {tensor.shape}")
            if len(tensors) > 10:
                print(f"  ... and {len(tensors) - 10} more")
        
        # Look for patch_embedding specifically
        print("\nSearching for patch_embedding related tensors:")
        patch_tensors = [t for t in reader.tensors if "patch" in t.name.lower()]
        if patch_tensors:
            for tensor in patch_tensors:
                print(f"  {tensor.name}: {tensor.shape} ({tensor.tensor_type})")
        else:
            print("  No patch_embedding tensors found!")
            
    except Exception as e:
        print(f"Error reading GGUF file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_gguf.py <path_to_gguf_file>")
        sys.exit(1)
    
    debug_gguf_file(sys.argv[1])
