import os
import netCDF4 as nc
import numpy as np
from pathlib import Path
from collections import defaultdict

def read_labels_from_nc(filepath):
    """
    Read labels from a NetCDF4 file.
    Adjust the variable name based on your actual file structure.
    """
    try:
        with nc.Dataset(filepath, 'r') as dataset:
            # Common label variable names - adjust as needed
            if 'label' in dataset.variables:
                labels = dataset.variables['label'][:]
            elif 'labels' in dataset.variables:
                labels = dataset.variables['labels'][:]
            elif 'tornado' in dataset.variables:
                labels = dataset.variables['tornado'][:]
            else:
                # If labels are stored differently, modify this
                print(f"Warning: Could not find label variable in {filepath}")
                return None
            return np.array(labels)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_expected_subset_labels(original_labels):
    """
    Given original labels, determine what the subset should look like.
    
    Rules:
    1. Find all tornadic time steps (label = 1)
    2. For consecutive tornadic steps, mark only ONE pre-tornadic (step before sequence)
    3. Remove all tornadic time steps
    4. Return the expected subset labels after removal
    
    Returns:
        expected_labels: What the subset labels should be (after removal)
        mapping: Dictionary mapping original indices to subset indices
        pretornadic_original_indices: Which original indices should be pre-tornadic
    """
    labels = np.array(original_labels)
    n = len(labels)
    
    # Find all tornadic indices
    tornadic_indices = set(np.where(labels == 1)[0].tolist())
    
    pretornadic_original_indices = set()
    
    if tornadic_indices:
        # Group consecutive tornadic indices
        sorted_tornadic = sorted(tornadic_indices)
        tornadic_groups = []
        current_group = [sorted_tornadic[0]]
        
        for i in range(1, len(sorted_tornadic)):
            if sorted_tornadic[i] == sorted_tornadic[i-1] + 1:
                current_group.append(sorted_tornadic[i])
            else:
                tornadic_groups.append(current_group)
                current_group = [sorted_tornadic[i]]
        tornadic_groups.append(current_group)
        
        # For each group, mark the time step before as pre-tornadic
        for group in tornadic_groups:
            first_tornadic_idx = group[0]
            if first_tornadic_idx > 0:  # Can only have pre-tornadic if not first time step
                pretornadic_original_indices.add(first_tornadic_idx - 1)
    
    # Build expected subset (after removing tornadic indices)
    expected_labels = []
    original_to_subset = {}  # Maps original index to subset index
    subset_idx = 0
    
    for orig_idx in range(n):
        if orig_idx not in tornadic_indices:
            # This index remains in subset
            original_to_subset[orig_idx] = subset_idx
            
            # Determine label: 1 if pre-tornadic, 0 otherwise
            if orig_idx in pretornadic_original_indices:
                expected_labels.append(1)
            else:
                expected_labels.append(original_labels[orig_idx])
            
            subset_idx += 1
    
    return np.array(expected_labels), original_to_subset, pretornadic_original_indices

def verify_file(original_path, subset_path, filename):
    """
    Verify that a single file's pre-tornadic labels are correct.
    """
    errors = []
    warnings = []
    
    # Read original labels
    original_labels = read_labels_from_nc(original_path)
    if original_labels is None:
        return [f"Could not read original file: {filename}"], []
    
    # Read subset labels
    subset_labels = read_labels_from_nc(subset_path)
    if subset_labels is None:
        return [f"Could not read subset file: {filename}"], []
    
    # Get expected subset labels
    expected_labels, original_to_subset, pretornadic_original = get_expected_subset_labels(original_labels)
    
    # Check 1: Length should match (tornadic removed)
    if len(subset_labels) != len(expected_labels):
        errors.append(f"  Length mismatch: subset has {len(subset_labels)} time steps, expected {len(expected_labels)}")
        errors.append(f"    Original: {len(original_labels)} time steps, {np.sum(original_labels)} tornadic")
        errors.append(f"    Expected to remove {np.sum(original_labels)} tornadic steps")
    
    # Check 2: Compare each label
    min_len = min(len(subset_labels), len(expected_labels))
    for subset_idx in range(min_len):
        if subset_labels[subset_idx] != expected_labels[subset_idx]:
            # Find which original index this corresponds to
            orig_idx = None
            for o_idx, s_idx in original_to_subset.items():
                if s_idx == subset_idx:
                    orig_idx = o_idx
                    break
            
            error_msg = f"  Subset index {subset_idx}"
            if orig_idx is not None:
                error_msg += f" (original index {orig_idx})"
            error_msg += f": expected {expected_labels[subset_idx]}, got {subset_labels[subset_idx]}"
            
            if orig_idx in pretornadic_original:
                error_msg += " [should be pre-tornadic]"
            
            errors.append(error_msg)
    
    # Check 3: Verify pre-tornadic assignments
    pretornadic_count_expected = len(pretornadic_original)
    pretornadic_count_actual = np.sum(expected_labels == 1)
    
    if pretornadic_count_expected > 0:
        warnings.append(f"  Pre-tornadic time steps identified: {pretornadic_count_expected}")
        warnings.append(f"    Original indices: {sorted(pretornadic_original)}")
    
    return errors, warnings

def verify_datasets(original_dir, subset_dir, verbose=True):
    """
    Verify all files in both datasets.
    """
    original_dir = Path(original_dir)
    subset_dir = Path(subset_dir)
    
    # Get all NetCDF files from both directories
    original_files = set([f.name for f in original_dir.glob('*.nc*')])
    subset_files = set([f.name for f in subset_dir.glob('*.nc*')])
    
    print(f"Original dataset: {len(original_files)} files")
    print(f"Subset dataset: {len(subset_files)} files")
    
    # Files only in subset might be unexpected
    only_in_subset = subset_files - original_files
    if only_in_subset:
        print(f"⚠ Warning: {len(only_in_subset)} files in subset but not in original")
    
    # Files only in original are expected (files with no pre-tornadic signatures)
    only_in_original = original_files - subset_files
    print(f"Files only in original: {len(only_in_original)} (may not have pre-tornadic signatures)")
    
    print("\n" + "="*80 + "\n")
    
    total_errors = 0
    files_with_errors = 0
    files_verified = 0
    
    # Verify files that exist in both datasets
    common_files = original_files & subset_files
    
    for filename in sorted(common_files):
        original_path = original_dir / filename
        subset_path = subset_dir / filename
        
        errors, warnings = verify_file(original_path, subset_path, filename)
        files_verified += 1
        
        if errors:
            files_with_errors += 1
            total_errors += len(errors)
            print(f"❌ {filename}:")
            for error in errors:
                print(error)
            if verbose and warnings:
                for warning in warnings:
                    print(warning)
            print()
        elif verbose and warnings:
            print(f"✓ {filename}: OK")
            for warning in warnings:
                print(warning)
            print()
        else:
            print(f"✓ {filename}: OK")
    
    print("\n" + "="*80)
    print(f"\nSummary:")
    print(f"  Total files verified: {files_verified}")
    print(f"  Files with errors: {files_with_errors}")
    print(f"  Total errors found: {total_errors}")
    
    if total_errors == 0:
        print("\n✓ All pre-tornadic labels are correctly assigned!")
    else:
        print(f"\n❌ Found {total_errors} labeling errors across {files_with_errors} files")
    
    return total_errors == 0

if __name__ == "__main__":
    # Update these paths to your dataset locations
    ORIGINAL_DATASET_DIR = "/path/to/original/dataset"
    SUBSET_DATASET_DIR = "/path/to/subset/dataset"
    
    print("Pre-tornadic Label Verification")
    print("="*80)
    print()
    
    # Set verbose=False to only show errors
    verify_datasets(ORIGINAL_DATASET_DIR, SUBSET_DATASET_DIR, verbose=True)