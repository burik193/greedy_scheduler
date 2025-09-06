from typing import Dict, List, Tuple, Optional
from grid import Grid
from order import Order, Activity
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


class Solver:
    """
    Greedy solver for placing order activities on a grid.
    Places activities as far left as possible while respecting:
    - Grid blockers (populated areas)
    - Activity dependencies and pause constraints
    - Row assignments
    - Multiple orders (no conflicts between orders)
    """
    
    def __init__(self, grid: Grid, orders: List[Order] = None, verbose: bool = True):
        """
        Initialize the solver with a grid and orders.
        
        Args:
            grid (Grid): Grid with populated areas (blockers)
            orders (List[Order]): List of orders with activities to place
            verbose (bool): If True, print detailed progress. If False, use tqdm progress bars
        """
        self.grid = grid
        self.orders = orders if orders else []
        self.row_timelines = {}  # row -> list of (start, end, activity_id, order_id)
        self.blocker_map = {}    # row -> list of (start, end) blocked intervals
        self.solution = {}       # activity_id -> (start_time, end_time, row, order_id)
        self.order_solutions = {}  # order_id -> Dict[activity_id, (start, end, row)]
        self.order_by_id = {}    # order_id -> Order reference
        self.infeasible = False
        self.infeasibility_reason = ""
        self.next_order_id = 0
        self.verbose = verbose
    
    def pprint(self, *args, **kwargs):
        """Conditional print method that only prints when verbose is True."""
        if self.verbose:
            print(*args, **kwargs)
    
    def solve(self) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Solve the placement problem for all orders.
        
        Returns:
            Dict[int, Tuple[int, int, int, int]]: activity_id -> (start_time, end_time, row, order_id)
            Empty dict if no activities could be placed
        """
        # Reset solution state
        self.solution.clear()
        self.order_solutions.clear()
        self.infeasible = False
        self.infeasibility_reason = ""
        
        # Extract blockers from grid
        self._extract_blockers()
        
        # Initialize row timelines
        self._initialize_timelines()
        
        # Track which orders succeeded and failed
        successful_orders = []
        failed_orders = []
        
        # Solve each order
        if self.verbose:
            for i, order in enumerate(self.orders):
                self.pprint(f"\nAttempting to place Order {i}...")
                if self._solve_order(order):
                    successful_orders.append(i)
                    self.pprint(f"✅ Order {i} placed successfully with {len(self.order_solutions[self.next_order_id - 1])} activities")
                else:
                    failed_orders.append(i)
                    self.pprint(f"❌ Order {i} failed: {self.infeasibility_reason}")
                    # Reset infeasibility for next order
                    self.infeasible = False
                    self.infeasibility_reason = ""
        else:
            # Use tqdm for progress when not verbose
            for i, order in tqdm(enumerate(self.orders), desc="Placing orders", unit="order"):
                if self._solve_order(order):
                    successful_orders.append(i)
                else:
                    failed_orders.append(i)
                    # Reset infeasibility for next order
                    self.infeasible = False
                    self.infeasibility_reason = ""
        
        # Report results
        if self.verbose:
            self.pprint(f"\n=== PLACEMENT SUMMARY ===")
            self.pprint(f"Successful orders: {successful_orders}")
            self.pprint(f"Failed orders: {failed_orders}")
            self.pprint(f"Total activities placed: {len(self.solution)}")
        
        if not self.solution:
            self.infeasible = True
            self.infeasibility_reason = "No activities could be placed from any order"
            return {}
        
        return self.solution.copy()
    
    def _solve_order(self, order: Order) -> bool:
        """
        Solve placement for a single order.
        
        Args:
            order (Order): Order to solve
            
        Returns:
            bool: True if successful, False if infeasible
        """
        order_id = self.next_order_id
        self.next_order_id += 1
        
        self.pprint(f"  Processing order with internal ID {order_id}")
        
        # Store the order reference for this order_id
        self.order_by_id[order_id] = order
        
        # Get execution order for this order
        execution_order = order.get_execution_order()
        if not execution_order:
            self.pprint(f"    Order {order_id} has no activities, skipping")
            return True  # Empty order, skip
        
        self.pprint(f"    Order {order_id} has {len(execution_order)} activities")
        
        # Initialize order solution
        self.order_solutions[order_id] = {}
        
        # Place activities sequentially
        for activity in execution_order:
            if not self._place_activity(activity, order_id):
                self.pprint(f"    Failed to place activity {activity.id} for order {order_id}")
                return False  # Infeasible
        
        self.pprint(f"    Successfully placed all activities for order {order_id}")
        self.pprint(f"    Solution now contains {len(self.solution)} activities:")
        for act_id, (start, end, row, ord_id) in self.solution.items():
            self.pprint(f"      Activity {act_id}: Row {row}, Time {start}-{end}, Order {ord_id}")
        return True
    
    def _extract_blockers(self):
        """Extract blocked intervals from the grid for each row."""
        self.blocker_map.clear()
        
        for row in range(self.grid.n):
            self.blocker_map[row] = []
            
            # Find consecutive blocks of ones (blockers) in this row
            start = None
            for col in range(self.grid.m):
                if self.grid.grid[row, col] == 1 and start is None:
                    start = col
                elif self.grid.grid[row, col] == 0 and start is not None:
                    # End of a blocker
                    self.blocker_map[row].append((start, col))
                    start = None
            
            # Handle case where blocker extends to end of row
            if start is not None:
                self.blocker_map[row].append((start, self.grid.m))
    
    def _initialize_timelines(self):
        """Initialize timeline for each row."""
        self.row_timelines.clear()
        
        for row in range(self.grid.n):
            self.row_timelines[row] = []
    
    def _place_activity(self, activity: Activity, order_id: int) -> bool:
        """
        Place a single activity on its assigned row or find alternative row.
        
        Args:
            activity (Activity): Activity to place
            order_id (int): ID of the order this activity belongs to
            
        Returns:
            bool: True if placement successful, False if infeasible
        """
        original_row = activity.row
        duration = activity.duration
        
        self.pprint(f"      Placing Activity {activity.id} (duration: {duration}, preferred row: {original_row}) for Order {order_id}")
        
        # Find earliest and latest start times based on dependencies
        earliest_start = self._get_earliest_start_time(activity, order_id)
        latest_start = self._get_latest_start_time(activity, order_id)
        self.pprint(f"        Earliest start time: {earliest_start}")
        if latest_start is not None:
            self.pprint(f"        Latest start time: {latest_start}")
        
        # Try original row first
        start_time = self._find_leftmost_position(original_row, earliest_start, duration, latest_start)
        if start_time is not None:
            self.pprint(f"        ✅ Found position on preferred row {original_row}: time {start_time}-{start_time + duration}")
            self._place_activity_at_position(activity, order_id, start_time, start_time + duration, original_row)
            return True
        
        self.pprint(f"        ❌ No valid position on preferred row {original_row}, trying alternative rows...")
        
        # Try alternative rows
        for alt_row in range(self.grid.n):
            if alt_row == original_row:
                continue
                
            start_time = self._find_leftmost_position(alt_row, earliest_start, duration, latest_start)
            if start_time is not None:
                self.pprint(f"        ✅ Found position on alternative row {alt_row}: time {start_time}-{start_time + duration}")
                self._place_activity_at_position(activity, order_id, start_time, start_time + duration, alt_row)
                return True
        
        self.pprint(f"        ❌ Failed to find valid position for Activity {activity.id} on any row")
        self.infeasible = True
        self.infeasibility_reason = f"Activity {activity.id} cannot be placed on any row"
        return False
    
    def _place_activity_at_position(self, activity: Activity, order_id: int, start_time: int, end_time: int, row: int):
        """Helper method to place an activity at a specific position."""
        # Create unique activity ID by combining order_id and activity.id
        unique_activity_id = f"{order_id}_{activity.id}"
        
        # Place the activity with unique ID
        self.solution[unique_activity_id] = (start_time, end_time, row, order_id)
        self.order_solutions[order_id][unique_activity_id] = (start_time, end_time, row)
        
        self.pprint(f"        Activity {activity.id} placed in solution as {unique_activity_id}: {self.solution[unique_activity_id]}")
        
        # Update row timeline with unique ID
        self._update_timeline(row, start_time, end_time, unique_activity_id, order_id)
    
    def _get_earliest_start_time(self, activity: Activity, order_id: int) -> int:
        """
        Calculate earliest start time based on dependencies.
        
        Args:
            activity (Activity): Activity to find start time for
            order_id (int): ID of the order this activity belongs to
            
        Returns:
            int: Earliest possible start time
        """
        # Find the order this activity belongs to
        order = self.order_by_id[order_id]
        
        # Find the activity that comes before this one
        for prev_activity in order.activities:
            if prev_activity.next_activity and prev_activity.next_activity.id == activity.id:
                # This activity depends on prev_activity
                prev_unique_id = f"{order_id}_{prev_activity.id}"
                if prev_unique_id in self.solution:
                    prev_start, prev_end, _, _ = self.solution[prev_unique_id]
                    # Add pause constraint - use min_pause as earliest start
                    return prev_end + prev_activity.min_pause
                else:
                    # Previous activity not placed yet - this shouldn't happen in sequential placement
                    self.infeasible = True
                    self.infeasibility_reason = f"Previous activity {prev_activity.id} not placed"
                    return 0
        
        # No dependencies - can start at time 0
        return 0
    
    def _get_latest_start_time(self, activity: Activity, order_id: int) -> int:
        """
        Calculate latest start time based on dependencies to respect max_pause.
        
        Args:
            activity (Activity): Activity to find start time for
            order_id (int): ID of the order this activity belongs to
            
        Returns:
            int: Latest possible start time (or None if no constraint)
        """
        # Find the order this activity belongs to
        order = self.order_by_id[order_id]
        
        # Find the activity that comes before this one
        for prev_activity in order.activities:
            if prev_activity.next_activity and prev_activity.next_activity.id == activity.id:
                # This activity depends on prev_activity
                prev_unique_id = f"{order_id}_{prev_activity.id}"
                if prev_unique_id in self.solution:
                    prev_start, prev_end, _, _ = self.solution[prev_unique_id]
                    # Add max pause constraint
                    return prev_end + prev_activity.max_pause
                else:
                    return None
        
        # No dependencies - no latest start time constraint
        return None
    
    def _find_leftmost_position(self, row: int, earliest_start: int, duration: int, latest_start: Optional[int] = None) -> Optional[int]:
        """
        Find the leftmost valid position for an activity on a given row.
        
        Args:
            row (int): Row to place activity on
            earliest_start (int): Earliest possible start time
            duration (int): Duration of the activity
            latest_start (Optional[int]): Latest possible start time (for max_pause constraint)
            
        Returns:
            Optional[int]: Start time if placement possible, None otherwise
        """
        # Check if row exists
        if row not in self.row_timelines:
            return None
        
        # Try to place as far left as possible, but respect latest_start if provided
        start_time = earliest_start
        
        # If latest_start is provided, we can't start after it
        max_start = latest_start if latest_start is not None else self.grid.m
        
        while start_time + duration <= self.grid.m and start_time <= max_start:
            # Check if this position is valid
            if self._is_position_valid(row, start_time, start_time + duration):
                return start_time
            
            start_time += 1
        
        return None
    
    def _is_position_valid(self, row: int, start: int, end: int) -> bool:
        """
        Check if a position is valid (no conflicts with blockers or other activities).
        
        Args:
            row (int): Row to check
            start (int): Start time
            end (int): End time
            
        Returns:
            bool: True if position is valid
        """
        # Check for blocker conflicts
        for blocker_start, blocker_end in self.blocker_map.get(row, []):
            if start < blocker_end and end > blocker_start:
                return False  # Overlaps with blocker
        
        # Check for activity conflicts
        for activity_start, activity_end, _, _ in self.row_timelines.get(row, []):
            if start < activity_end and end > activity_start:
                return False  # Overlaps with another activity
        
        return True
    
    def _update_timeline(self, row: int, start: int, end: int, activity_id: int, order_id: int):
        """
        Update the timeline for a row after placing an activity.
        
        Args:
            row (int): Row to update
            start (int): Start time of activity
            end (int): End time of activity
            activity_id (int): ID of the placed activity
            order_id (int): ID of the order this activity belongs to
        """
        if row not in self.row_timelines:
            self.row_timelines[row] = []
        
        # Insert activity in sorted order by start time
        timeline = self.row_timelines[row]
        
        # Find insertion position
        insert_pos = 0
        for i, (existing_start, _, _, _) in enumerate(timeline):
            if start < existing_start:
                insert_pos = i
                break
            insert_pos = i + 1
        
        # Insert the activity
        timeline.insert(insert_pos, (start, end, activity_id, order_id))
    
    def add_order(self, order: Order):
        """
        Add an order to the solver.
        
        Args:
            order (Order): Order to add
        """
        self.orders.append(order)
    
    def clear_orders(self):
        """Clear all orders from the solver."""
        self.orders.clear()
        self.order_solutions.clear()
        self.solution.clear()
        self.order_by_id.clear()
        self.next_order_id = 0
    
    def get_order_solution(self, order_index: int) -> Dict[int, Tuple[int, int, int]]:
        """
        Get solution for a specific order.
        
        Args:
            order_index (int): Index of the order in the orders list
        
        Returns:
            Dict[int, Tuple[int, int, int]]: activity_id -> (start_time, end_time, row)
        """
        if order_index < len(self.orders):
            return self.order_solutions.get(order_index, {}).copy()
        return {}
    
    def get_order_makespan(self, order_index: int) -> int:
        """
        Get makespan for a specific order.
        
        Args:
            order_index (int): Index of the order in the orders list
        
        Returns:
            int: Makespan of the order
        """
        order_solution = self.get_order_solution(order_index)
        if not order_solution:
            return 0
        
        max_end_time = max(end_time for _, end_time, _ in order_solution.values())
        return max_end_time
    
    def is_feasible(self) -> bool:
        """Check if the problem is feasible."""
        return not self.infeasible
    
    def get_infeasibility_reason(self) -> str:
        """Get the reason for infeasibility if any."""
        return self.infeasibility_reason
    
    def get_makespan(self) -> int:
        """Calculate the total makespan of the solution."""
        if not self.solution:
            return 0
        
        max_end_time = max(end_time for _, end_time, _, _ in self.solution.values())
        return max_end_time
    
    def get_solution_summary(self) -> Dict:
        """Get a summary of the solution."""
        if not self.solution:
            return {"status": "infeasible", "reason": self.infeasibility_reason}
        
        return {
            "status": "feasible",
            "total_activities": len(self.solution),
            "total_orders": len(self.orders),
            "makespan": self.get_makespan(),
            "orders": {
                order_index: {
                    "activities": len(self.get_order_solution(order_index)),
                    "makespan": self.get_order_makespan(order_index)
                }
                for order_index in range(len(self.orders))
            },
            "activities": {
                activity_id: {
                    "start": start_time,
                    "end": end_time,
                    "row": row,
                    "order_id": order_id,
                    "duration": end_time - start_time
                }
                for activity_id, (start_time, end_time, row, order_id) in self.solution.items()
            }
        }
    
    def visualize_solution(self, title="Activity Placement Solution"):
        """
        Visualize the solution using a simple Plotly gantt chart.
        
        Args:
            title (str): Title for the visualization
        """
        if not self.solution:
            print("No solution to visualize")
            return
        
        self.pprint(f"Visualizing solution with {len(self.solution)} activities:")
        for activity_id, (start, end, row, order_id) in self.solution.items():
            self.pprint(f"  Activity {activity_id}: Row {row}, Time {start}-{end}, Order {order_id}")
        
        # Create data for timeline
        timeline_data = []
        
        # Add activities
        for activity_id, (start, end, row, order_id) in self.solution.items():
            timeline_data.append({
                'Task': f'Row {row}',
                'Start': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=start),
                'Finish': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=end),
                'Resource': f'Order {order_id}',
                'Activity': f'A{activity_id}'
            })
        
        # Add blockers
        for row in range(self.grid.n):
            for start, end in self.blocker_map.get(row, []):
                timeline_data.append({
                    'Task': f'Row {row}',
                    'Start': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=start),
                    'Finish': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=end),
                    'Resource': 'Blockers',
                    'Activity': 'BLOCKED'
                })
        
        # Create DataFrame
        df = pd.DataFrame(timeline_data)
        
        self.pprint(f"Timeline data created with {len(timeline_data)} entries")
        self.pprint(f"Unique resources (orders): {sorted(df['Resource'].unique())}")
        
        # Create simple gantt chart using px.timeline
        fig = px.timeline(
            df, 
            x_start="Start", 
            x_end="Finish", 
            y="Task",
            color="Resource",
            title=title,
            hover_data=["Activity"]
        )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            height=600,
            showlegend=True
        )
        
        # Show the plot
        fig.show()
    
    def check_consistency(self) -> Dict:
        """
        Check if the solution violates any constraints.
        
        Returns:
            Dict: JSON report of all violations found
        """
        violations = {
            "dependency_violations": [],
            "pause_violations": [],
            "overlap_violations": [],
            "blocker_violations": [],
            "summary": {
                "total_violations": 0,
                "dependency_violations": 0,
                "pause_violations": 0,
                "overlap_violations": 0,
                "blocker_violations": 0
            }
        }
        
        # Check dependency violations
        for order_id, order in self.order_by_id.items():
            for activity in order.activities:
                if activity.next_activity:
                    # Check if next activity is placed after this one
                    current_id = f"{order_id}_{activity.id}"
                    next_id = f"{order_id}_{activity.next_activity.id}"
                    
                    if current_id in self.solution and next_id in self.solution:
                        current_start, current_end, _, _ = self.solution[current_id]
                        next_start, next_end, _, _ = self.solution[next_id]
                        
                        if next_start < current_end:
                            violations["dependency_violations"].append({
                                "activity": current_id,
                                "where": [
                                    f"2024-01-01T{next_start:02d}:00:00",
                                    f"2024-01-01T{current_end:02d}:00:00"
                                ],
                                "why": f"Next activity {next_id} starts before current activity ends"
                            })
        
        # Check pause violations
        for order_id, order in self.order_by_id.items():
            for activity in order.activities:
                if activity.next_activity:
                    current_id = f"{order_id}_{activity.id}"
                    next_id = f"{order_id}_{activity.next_activity.id}"
                    
                    if current_id in self.solution and next_id in self.solution:
                        current_start, current_end, _, _ = self.solution[current_id]
                        next_start, next_end, _, _ = self.solution[next_id]
                        
                        pause = next_start - current_end
                        if pause < activity.min_pause:
                            violations["pause_violations"].append({
                                "activity": current_id,
                                "where": [
                                    f"2024-01-01T{current_end:02d}:00:00",
                                    f"2024-01-01T{next_start:02d}:00:00"
                                ],
                                "why": f"Pause {pause} is less than minimum required {activity.min_pause}"
                            })
                        elif pause > activity.max_pause:
                            violations["pause_violations"].append({
                                "activity": current_id,
                                "where": [
                                    f"2024-01-01T{current_end:02d}:00:00",
                                    f"2024-01-01T{next_start:02d}:00:00"
                                ],
                                "why": f"Pause {pause} is greater than maximum allowed {activity.max_pause}"
                            })
        
        # Check activity overlaps within each row
        for row in range(self.grid.n):
            row_activities = []
            for activity_id, (start, end, activity_row, _) in self.solution.items():
                if activity_row == row:
                    row_activities.append((start, end, activity_id))
            
            # Sort by start time
            row_activities.sort(key=lambda x: x[0])
            
            # Check for overlaps
            for i in range(len(row_activities) - 1):
                current_start, current_end, current_id = row_activities[i]
                next_start, next_end, next_id = row_activities[i + 1]
                
                if current_end > next_start:
                    violations["overlap_violations"].append({
                        "activity": current_id,
                        "where": [
                            f"2024-01-01T{next_start:02d}:00:00",
                            f"2024-01-01T{current_end:02d}:00:00"
                        ],
                        "why": f"Overlaps with activity {next_id}"
                    })
        
        # Check blocker violations
        for activity_id, (start, end, row, _) in self.solution.items():
            for blocker_start, blocker_end in self.blocker_map.get(row, []):
                if start < blocker_end and end > blocker_start:
                    violations["blocker_violations"].append({
                        "activity": activity_id,
                        "where": [
                            f"2024-01-01T{max(start, blocker_start):02d}:00:00",
                            f"2024-01-01T{min(end, blocker_end):02d}:00:00"
                        ],
                        "why": "blocker"
                    })
        
        # Update summary counts
        violations["summary"]["dependency_violations"] = len(violations["dependency_violations"])
        violations["summary"]["pause_violations"] = len(violations["pause_violations"])
        violations["summary"]["overlap_violations"] = len(violations["overlap_violations"])
        violations["summary"]["blocker_violations"] = len(violations["blocker_violations"])
        violations["summary"]["total_violations"] = sum([
            violations["summary"]["dependency_violations"],
            violations["summary"]["pause_violations"],
            violations["summary"]["overlap_violations"],
            violations["summary"]["blocker_violations"]
        ])
        
        return violations
    
    def save_consistency_report(self, filename: str = "consistency_report.json"):
        """
        Save the consistency report to a JSON file.
        
        Args:
            filename (str): Name of the file to save the report to
        """
        import json
        
        report = self.check_consistency()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Consistency report saved to {filename}")
        print(f"Total violations found: {report['summary']['total_violations']}")


# Example usage and testing
if __name__ == "__main__":
    from grid import Grid
    from order import Order
    
    # 1. Create 5 orders
    print("Creating 5 test orders...")
    orders = []
    
    for i in range(100):
        order = Order()
        order.populate(num_activities=100, max_time=8, max_row=5, min_pause=2, max_pause=5)
        orders.append(order)
        print(f"Order {i}: {100} activities")
    
    # 2. Create grid and solve
    print("\nCreating grid and solving...")
    grid = Grid(5, 10000)
    grid.populate(2)  # Add some random blockers
    
    
    # Test non-verbose mode with tqdm
    print("\n=== NON-VERBOSE MODE (with tqdm) ===")
    solver2 = Solver(grid, orders, verbose=False)
    solution2 = solver2.solve()
    
    if solver2.is_feasible():
        print(f"✅ Solution found! Total activities: {len(solution2)}")
        solver2.visualize_solution("5 Orders Placement Solution (Non-verbose)")
        
        # Check consistency and save report
        print("\n=== CHECKING SOLUTION CONSISTENCY ===")
        consistency_report = solver2.check_consistency()
        print(f"Consistency check complete. Total violations: {consistency_report['summary']['total_violations']}")
        
        if consistency_report['summary']['total_violations'] > 0:
            print("Violations found:")
            for violation_type, violations in consistency_report.items():
                if violation_type != 'summary' and violations:
                    print(f"  {violation_type}: {len(violations)} violations")
        else:
            print("✅ No violations found - solution is consistent!")
        
        # Save detailed report to JSON file
        solver2.save_consistency_report("consistency_report.json")
        
    else:
        print(f"❌ Problem is infeasible: {solver2.get_infeasibility_reason()}")



