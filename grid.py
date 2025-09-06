import numpy as np
import random


class Grid:
    def __init__(self, n: int, m: int):
        """
        Initialize a grid of size N x M filled with zeros.
        
        Args:
            n (int): Number of rows
            m (int): Number of columns
        """
        self.n = n
        self.m = m
        self.grid = np.zeros((n, m), dtype=int)
        # Store intervals for efficient operations
        self.row_intervals = [[] for _ in range(n)]
    
    def populate(self, r: int):
        """
        Populate the grid with R random horizontal blocks of ones.
        Each row gets R blocks that don't touch each other and have size > 1.
        
        Args:
            r (int): Number of blocks per row
        """
        # Reset grid to zeros
        self.grid = np.zeros((self.n, self.m), dtype=int)
        # Reset intervals
        self.row_intervals = [[] for _ in range(self.n)]
        
        for row in range(self.n):
            # Generate R random block positions and sizes for this row
            blocks = self._generate_blocks_for_row(r)
            
            # Apply blocks to the row and store intervals
            for start, size in blocks:
                self.grid[row, start:start + size] = 1
                self.row_intervals[row].append((start, start + size))
    
    def _generate_blocks_for_row(self, r: int):
        """
        Generate R non-overlapping blocks for a single row.
        
        Args:
            r (int): Number of blocks to generate
            
        Returns:
            list: List of tuples (start_position, block_size)
        """
        if r == 0:
            return []
        
        # Calculate minimum spacing needed between blocks
        min_block_size = 2  # Blocks must be bigger than 1
        min_spacing = 1     # Minimum gap between blocks
        
        # Estimate maximum block size to ensure we can fit R blocks
        max_block_size = max(min_block_size, (self.m - (r - 1) * min_spacing) // r)
        
        blocks = []
        available_positions = list(range(self.m))
        
        for _ in range(r):
            if len(available_positions) < min_block_size:
                break
                
            # Randomly choose block size (between min_block_size and max_block_size)
            block_size = random.randint(min_block_size, max_block_size)
            
            # Find valid starting positions for this block
            valid_starts = []
            for start in available_positions:
                if start + block_size <= self.m:
                    # Check if this position doesn't overlap with existing blocks
                    valid = True
                    for existing_start, existing_size in blocks:
                        if (start < existing_start + existing_size + min_spacing and 
                            start + block_size + min_spacing > existing_start):
                            valid = False
                            break
                    if valid:
                        valid_starts.append(start)
            
            if valid_starts:
                start_pos = random.choice(valid_starts)
                blocks.append((start_pos, block_size))
                
                # Remove used positions and nearby positions
                for i in range(max(0, start_pos - min_spacing), 
                             min(self.m, start_pos + block_size + min_spacing)):
                    if i in available_positions:
                        available_positions.remove(i)
        
        return blocks
    
    def __add__(self, other):
        """
        Add two grids of the same size together.
        
        Args:
            other (Grid): Another grid to add
            
        Returns:
            tuple: (success: bool, result_grid: Grid)
        """
        if not isinstance(other, Grid):
            raise TypeError("Can only add Grid objects together")
        
        if self.n != other.n or self.m != other.m:
            raise ValueError("Grids must have the same dimensions")
        
        # Add the grids
        result_array = self.grid + other.grid
        result_grid = Grid(self.n, self.m)
        result_grid.grid = result_array
        
        # Populate the row_intervals for the result grid
        result_grid._update_intervals_from_grid()
        
        # Check if addition is successful (max value <= 1)
        max_value = np.max(result_array)
        success = max_value <= 1
        
        return success, result_grid
    
    def __str__(self):
        """String representation of the grid."""
        return str(self.grid)
    
    def __repr__(self):
        """Detailed representation of the grid."""
        return f"Grid({self.n}, {self.m})\n{self.grid}"
    
    def get_grid(self):
        """Get the numpy array representation of the grid."""
        return self.grid.copy()
    
    def set_grid(self, grid_array):
        """Set the grid from a numpy array."""
        if grid_array.shape != (self.n, self.m):
            raise ValueError(f"Grid array must have shape ({self.n}, {self.m})")
        self.grid = grid_array.copy()
        # Update intervals to match the new grid
        self._update_intervals_from_grid()
    
    def get_grid_stats(self):
        """Get efficient statistics about the grid without full array operations."""
        total_ones = sum(len(intervals) for intervals in self.row_intervals)
        total_zeros = self.n * self.m - total_ones
        
        # Count overlapping positions (where addition would create values > 1)
        overlapping_positions = 0
        for row in range(self.n):
            # Create a set of all positions that have ones
            ones_positions = set()
            for start, end in self.row_intervals[row]:
                ones_positions.update(range(start, end))
            overlapping_positions += len(ones_positions)
        
        return {
            'total_ones': total_ones,
            'total_zeros': total_zeros,
            'ones_positions': overlapping_positions,
            'density': overlapping_positions / (self.n * self.m)
        }
    
    def get_addition_summary(self, other):
        """Get a summary of what happens when adding this grid with another."""
        if not isinstance(other, Grid):
            raise TypeError("Can only analyze addition with Grid objects")
        
        if self.n != other.n or self.m != other.m:
            raise ValueError("Grids must have the same dimensions")
        
        # Count overlaps (positions where both grids have ones)
        overlaps = 0
        for row in range(self.n):
            # Get positions with ones in both grids
            self_ones = set()
            other_ones = set()
            
            for start, end in self.row_intervals[row]:
                self_ones.update(range(start, end))
            for start, end in other.row_intervals[row]:
                other_ones.update(range(start, end))
            
            # Count overlaps
            overlaps += len(self_ones.intersection(other_ones))
        
        total_ones_self = sum(len(intervals) for intervals in self.row_intervals)
        total_ones_other = sum(len(intervals) for intervals in other.row_intervals)
        
        return {
            'total_ones_self': total_ones_self,
            'total_ones_other': total_ones_other,
            'overlapping_positions': overlaps,
            'will_succeed': overlaps == 0,  # Addition succeeds if no overlaps
            'max_value_after_addition': 2 if overlaps > 0 else 1
        }
    

    
    def _update_intervals_from_grid(self):
        """Update row_intervals based on the current grid array."""
        self.row_intervals = [[] for _ in range(self.n)]
        
        for row in range(self.n):
            # Find consecutive blocks of ones in this row
            start = None
            for col in range(self.m):
                if self.grid[row, col] == 1 and start is None:
                    start = col
                elif self.grid[row, col] == 0 and start is not None:
                    # End of a block
                    self.row_intervals[row].append((start, col))
                    start = None
            
            # Handle case where block extends to end of row
            if start is not None:
                self.row_intervals[row].append((start, self.m))
    
    def _visualize_compact_gantt(self, title, cmap):
        """Compact Gantt chart for very large grids (100+ rows)."""
        # Create compact figure
        plt.figure(figsize=(12, 8))
        
        # Sample rows for visualization (show every nth row)
        sample_step = max(1, self.n // 50)  # Show max 50 rows
        sample_rows = list(range(0, self.n, sample_step))
        
        # Create compact Gantt chart
        for i, row in enumerate(sample_rows):
            # Draw row label
            plt.text(-self.m * 0.02, i, f'Row {row}', ha='right', va='center', fontsize=8)
            
            # Draw timeline
            plt.plot([0, self.m], [i, i], 'k-', alpha=0.3, linewidth=0.5)
            
            # Draw blocks as horizontal bars
            for start, end in self.row_intervals[row]:
                plt.barh(i, end - start, left=start, height=0.6, 
                         color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        plt.title(f"{title} (showing {len(sample_rows)} of {self.n} rows)", fontsize=14, fontweight='bold')
        plt.xlabel('Column Position', fontsize=12)
        plt.ylabel('Row (sampled)', fontsize=12)
        
        # Set axis limits and properties
        plt.xlim(-self.m * 0.05, self.m)
        plt.ylim(-0.5, len(sample_rows) - 0.5)
        
        # Add grid lines
        plt.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Set tick spacing
        step = max(1, self.m // 20)
        plt.xticks(range(0, self.m, step))
        plt.yticks(range(len(sample_rows)))
        
        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    
    def _visualize_small_grid(self, title, cmap, show_values):
        """Standard visualization for small grids."""
        # Create heatmap
        im = plt.imshow(self.grid, cmap=cmap, aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, label='Value')
        
        # Set title and labels
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        # Add grid lines
        plt.grid(True, which='both', color='black', linewidth=0.5, alpha=0.3)
        
        # Set tick labels
        plt.xticks(range(self.m))
        plt.yticks(range(self.n))
        
        # Add value annotations only if requested and grid is small
        if show_values and self.n <= 20 and self.m <= 20:
            for i in range(self.n):
                for j in range(self.m):
                    plt.text(j, i, str(self.grid[i, j]), 
                            ha='center', va='center', 
                            color='black' if self.grid[i, j] == 0 else 'white',
                            fontweight='bold')



