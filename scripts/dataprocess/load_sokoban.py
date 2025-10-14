#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sudoku Environment Generator
Generate predefined environments for different Sudoku sizes and save them as JSON files.
Supported sizes: 4x4, 6x6, 8x8, 10x10, 16x16
Generate 400 different environments for each size.
"""

import json
import random
import numpy as np
from typing import List, Dict, Any
import os

def generate_simple_sudoku_template(size: int, seed: int) -> List[List[int]]:
    """
    Generate a simple Sudoku template using a more direct method to avoid backtracking performance issues.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create an empty grid
    grid = [[0 for _ in range(size)] for _ in range(size)]
    box_size = int(size ** 0.5)
    
    # Fill each subgrid with some base numbers without violating rules
    for box_row in range(0, size, box_size):
        for box_col in range(0, size, box_size):
            # Randomly place a few numbers in each subgrid
            available_numbers = list(range(1, size + 1))
            random.shuffle(available_numbers)
            
            # Randomly choose some positions within the subgrid
            positions = [(r, c) for r in range(box_row, box_row + box_size) 
                        for c in range(box_col, box_col + box_size)]
            random.shuffle(positions)
            
            # Decide how many numbers to fill based on a simple difficulty heuristic
            fill_count = random.randint(1, min(3, len(available_numbers)))
            
            for i in range(fill_count):
                if i >= len(positions) or i >= len(available_numbers):
                    break
                    
                r, c = positions[i]
                num = available_numbers[i]
                
                # Simple check to avoid obvious conflicts
                if is_safe_placement(grid, r, c, num, size):
                    grid[r][c] = num
    
    # Randomly add some numbers in other cells
    empty_positions = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == 0]
    random.shuffle(empty_positions)
    
    # Add random numbers while ensuring basic rules are not violated
    additional_count = random.randint(size // 4, size // 2)
    for i, (r, c) in enumerate(empty_positions[:additional_count]):
        for num in random.sample(range(1, size + 1), size):
            if is_safe_placement(grid, r, c, num, size):
                grid[r][c] = num
                break
    
    return grid

def is_safe_placement(grid: List[List[int]], row: int, col: int, num: int, size: int) -> bool:
    """Check whether placing a number at the specified position is safe (does not violate Sudoku rules)."""
    box_size = int(size ** 0.5)
    
    # Check row
    for c in range(size):
        if grid[row][c] == num:
            return False
    
    # Check column
    for r in range(size):
        if grid[r][col] == num:
            return False
    
    # Check subgrid
    start_row = (row // box_size) * box_size
    start_col = (col // box_size) * box_size
    
    for r in range(start_row, start_row + box_size):
        for c in range(start_col, start_col + box_size):
            if grid[r][c] == num:
                return False
    
    return True

def generate_sudoku_environments_for_size(size: int, count: int = 400) -> List[Dict[str, Any]]:
    """Generate a specified number of Sudoku environments for a given size."""
    print(f"Generating {size}x{size} Sudoku environments, total {count} ...")
    
    environments = []
    
    for i in range(count):
        # Use different seeds to ensure diversity
        seed = i * 1000 + size * 100
        
        try:
            puzzle = generate_simple_sudoku_template(size, seed)
            
            # Compute some statistics
            filled_cells = sum(1 for row in puzzle for cell in row if cell != 0)
            total_cells = size * size
            fill_ratio = filled_cells / total_cells
            
            env_data = {
                "id": i,
                "size": size,
                "seed": seed,
                "puzzle": puzzle,
                "filled_cells": filled_cells,
                "total_cells": total_cells,
                "fill_ratio": round(fill_ratio, 3),
                "difficulty": "easy" if fill_ratio > 0.6 else "medium" if fill_ratio > 0.4 else "hard"
            }
            
            environments.append(env_data)
            
            # Progress every 100 items
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{count} for {size}x{size}")
                
        except Exception as e:
            print(f"  Error when generating index {i} for {size}x{size}: {e}")
            # Minimal fallback environment
            puzzle = [[0 for _ in range(size)] for _ in range(size)]
            # Put a few numbers on the diagonal
            for j in range(min(3, size)):
                if j < size:
                    puzzle[j][j] = (j % size) + 1
            
            env_data = {
                "id": i,
                "size": size,
                "seed": seed,
                "puzzle": puzzle,
                "filled_cells": min(3, size),
                "total_cells": size * size,
                "fill_ratio": min(3, size) / (size * size),
                "difficulty": "minimal"
            }
            environments.append(env_data)
    
    print(f"✅ Finished generating {size}x{size} Sudoku environments, total {len(environments)}")
    return environments

def main():
    """Main: generate Sudoku environments for all sizes."""
    print("🎯 Start generating Sudoku environments ...")
    
    # Supported sizes (all perfect squares or commonly used variants)
    sizes = [4, 6, 9, 12, 16]  # Note: 6x6 and 8x8 are not perfect squares; this script uses sqrt(size) subgrids.
    count_per_size = 400
    
    # Create output directory
    output_dir = "data/sudoku_environments"
    os.makedirs(output_dir, exist_ok=True)
    
    all_environments = {}
    
    for size in sizes:
        print(f"\n📋 Processing {size}x{size} ...")
        
        # Generate environments
        environments = generate_sudoku_environments_for_size(size, count_per_size)
        
        # Save to a separate file
        filename = f"{output_dir}/sudoku_{size}x{size}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(environments, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved to {filename}")
        
        # Add to the combined collection
        all_environments[f"{size}x{size}"] = environments
        
        # Show statistics
        fill_ratios = [env["fill_ratio"] for env in environments]
        avg_fill_ratio = sum(fill_ratios) / len(fill_ratios)
        print(f"📊 {size}x{size} stats: average fill ratio {avg_fill_ratio:.3f}")
    
    # Save all environments to a single file
    all_filename = f"{output_dir}/all_sudoku_environments.json"
    with open(all_filename, 'w', encoding='utf-8') as f:
        json.dump(all_environments, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 All Sudoku environments have been generated!")
    print(f"📁 Output directory: {output_dir}/")
    print(f"📄 Combined file: {all_filename}")
    
    # Show file sizes
    for size in sizes:
        filename = f"{output_dir}/sudoku_{size}x{size}.json"
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"   - sudoku_{size}x{size}.json: {size_mb:.2f} MB")

    """Test a generated environment."""
    print("\n🧪 Testing a generated environment ...")
    
    # Small-size test
    test_env = generate_simple_sudoku_template(4, 42)
    print("4x4 sample:")
    for row in test_env:
        print("  " + " ".join(f"{x:2d}" if x != 0 else " ." for x in row))
    
    # Basic rule validation
    size = 4
    box_size = 2
    valid = True
    
    # Check rows
    for r in range(size):
        row_nums = [test_env[r][c] for c in range(size) if test_env[r][c] != 0]
        if len(row_nums) != len(set(row_nums)):
            valid = False
            print(f"❌ Duplicate in row {r}")
    
    # Check columns
    for c in range(size):
        col_nums = [test_env[r][c] for r in range(size) if test_env[r][c] != 0]
        if len(col_nums) != len(set(col_nums)):
            valid = False
            print(f"❌ Duplicate in column {c}")
    
    if valid:
        print("✅ Basic validation passed")
    else:
        print("❌ Basic validation failed")

if __name__ == "__main__":
    
    # Generate all environments
    main()
