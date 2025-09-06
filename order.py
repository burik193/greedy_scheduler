import random
from typing import List, Dict, Tuple, Optional
import networkx as nx


class Activity:
    """Represents a single activity in an order."""
    
    def __init__(self, id: int, duration: int, row: int):
        """
        Initialize an activity.
        
        Args:
            id (int): Unique identifier for the activity
            duration (int): Duration of the activity (must be > 0)
            row (int): Row where the activity will be executed
        """
        if duration <= 0:
            raise ValueError("Activity duration must be positive")
        
        self.id = id
        self.duration = duration
        self.row = row
        self.next_activity: Optional[Activity] = None
        self.min_pause: int = 0
        self.max_pause: int = 0
    
    def set_next_activity(self, next_activity: 'Activity', min_pause: int, max_pause: int):
        """
        Set the next activity with pause constraints.
        
        Args:
            next_activity (Activity): The next activity to execute
            min_pause (int): Minimum pause between activities
            max_pause (int): Maximum pause between activities
        """
        if min_pause < 0 or max_pause < min_pause:
            raise ValueError("Invalid pause constraints")
        
        self.next_activity = next_activity
        self.min_pause = min_pause
        self.max_pause = max_pause
    
    def __str__(self):
        """String representation of the activity."""
        next_info = f" -> Activity {self.next_activity.id}" if self.next_activity else " (END)"
        pause_info = f" [pause: {self.min_pause}-{self.max_pause}]" if self.next_activity else ""
        return f"Activity {self.id} (duration: {self.duration}, row: {self.row}){next_info}{pause_info}"
    
    def __repr__(self):
        """Detailed representation of the activity."""
        return self.__str__()


class Order:
    """Represents an order with multiple activities and dependencies."""
    
    def __init__(self):
        """Initialize an empty order."""
        self.activities: List[Activity] = []
        self.activity_map: Dict[int, Activity] = {}  # id -> Activity mapping
    
    def add_activity(self, activity: Activity):
        """Add an activity to the order."""
        if activity.id in self.activity_map:
            raise ValueError(f"Activity with ID {activity.id} already exists")
        
        self.activities.append(activity)
        self.activity_map[activity.id] = activity
    
    def populate(self, num_activities: int, max_time: int, max_row: int, 
                min_pause: int = 5, max_pause: int = 60):
        """
        Populate the order with random activities and dependencies.
        
        Args:
            num_activities (int): Number of activities to create
            max_time (int): Maximum duration for any activity
            max_row (int): Maximum row number (exclusive)
            min_pause (int): Minimum pause between activities
            max_pause (int): Maximum pause between activities
        """
        if num_activities <= 0:
            raise ValueError("Number of activities must be positive")
        if max_time <= 0:
            raise ValueError("Max time must be positive")
        if max_row <= 0:
            raise ValueError("Max row must be positive")
        if min_pause < 0 or max_pause < min_pause:
            raise ValueError("Invalid pause constraints")
        
        # Clear existing activities
        self.activities.clear()
        self.activity_map.clear()
        
        # Create activities with random durations and rows
        for i in range(num_activities):
            duration = random.randint(1, max_time)
            row = random.randint(0, max_row - 1)
            
            activity = Activity(i, duration, row)
            self.add_activity(activity)
        
        # Create a valid spanning tree (no cycles)
        self._create_spanning_tree(min_pause, max_pause)
    
    def _create_spanning_tree(self, min_pause: int, max_pause: int):
        """
        Create a valid spanning tree of activities.
        Uses a modified depth-first approach to ensure no cycles.
        """
        if len(self.activities) <= 1:
            return
        
        # Start with the first activity as root
        root = self.activities[0]
        visited = {root.id}
        stack = [root]
        
        # Keep track of which activities still need connections
        unconnected = set(activity.id for activity in self.activities[1:])
        
        while unconnected and stack:
            current = stack[-1]
            
            # Find unconnected activities that can be connected to current
            candidates = []
            for activity_id in unconnected:
                activity = self.activity_map[activity_id]
                
                # Check if connecting would create a cycle
                if not self._would_create_cycle(current, activity):
                    candidates.append(activity)
            
            if candidates:
                # Choose a random candidate
                next_activity = random.choice(candidates)
                
                # Set up the connection with random pause
                pause = random.randint(min_pause, max_pause)
                current.set_next_activity(next_activity, min_pause, pause)
                
                # Mark as connected and add to stack
                unconnected.remove(next_activity.id)
                visited.add(next_activity.id)
                stack.append(next_activity)
            else:
                # Backtrack if no candidates found
                stack.pop()
        
        # If there are still unconnected activities, connect them to random connected ones
        while unconnected:
            activity_id = unconnected.pop()
            activity = self.activity_map[activity_id]
            
            # Find a random connected activity to connect to
            connected_activities = [a for a in self.activities if a.id in visited]
            if connected_activities:
                target = random.choice(connected_activities)
                pause = random.randint(min_pause, max_pause)
                target.set_next_activity(activity, min_pause, pause)
                visited.add(activity.id)
    
    def _would_create_cycle(self, source: Activity, target: Activity) -> bool:
        """
        Check if connecting source to target would create a cycle.
        
        Args:
            source (Activity): Source activity
            target (Activity): Target activity
            
        Returns:
            bool: True if connection would create a cycle
        """
        # Follow the chain from target to see if we reach source
        current = target
        visited = {target.id}
        
        while current.next_activity:
            current = current.next_activity
            if current.id == source.id:
                return True  # Would create a cycle
            if current.id in visited:
                break  # Already visited this activity
            visited.add(current.id)
        
        return False
    
    def get_execution_order(self) -> List[Activity]:
        """
        Get the activities in their execution order.
        
        Returns:
            List[Activity]: Activities in execution order
        """
        if not self.activities:
            return []
        
        # Find the root activity (one that has no incoming connections)
        incoming = set()
        for activity in self.activities:
            if activity.next_activity:
                incoming.add(activity.next_activity.id)
        
        # Root is the activity with no incoming connections
        root_id = None
        for activity in self.activities:
            if activity.id not in incoming:
                root_id = activity.id
                break
        
        if root_id is None:
            # If no root found, return activities as-is
            return self.activities.copy()
        
        # Traverse the tree in order
        execution_order = []
        current = self.activity_map[root_id]
        
        while current:
            execution_order.append(current)
            current = current.next_activity
        
        return execution_order
    
    def get_total_duration(self) -> int:
        """Calculate the total duration of all activities."""
        return sum(activity.duration for activity in self.activities)
    
    def get_critical_path(self) -> Tuple[List[Activity], int]:
        """
        Find the critical path (longest path) through the activities.
        
        Returns:
            Tuple[List[Activity], int]: Critical path activities and total duration
        """
        if not self.activities:
            return [], 0
        
        # Use dynamic programming to find longest path
        durations = {activity.id: activity.duration for activity in self.activities}
        max_durations = {}
        predecessors = {}
        
        # Initialize
        for activity in self.activities:
            max_durations[activity.id] = durations[activity.id]
            predecessors[activity.id] = None
        
        # Calculate longest paths
        for _ in range(len(self.activities)):
            for activity in self.activities:
                if activity.next_activity:
                    next_id = activity.next_activity.id
                    new_duration = max_durations[activity.id] + durations[next_id]
                    
                    if new_duration > max_durations[next_id]:
                        max_durations[next_id] = new_duration
                        predecessors[next_id] = activity.id
        
        # Find the activity with maximum duration
        max_activity_id = max(max_durations.keys(), key=lambda x: max_durations[x])
        
        # Reconstruct the path
        path = []
        current_id = max_activity_id
        while current_id is not None:
            path.insert(0, self.activity_map[current_id])
            current_id = predecessors[current_id]
        
        return path, max_durations[max_activity_id]
    
    def validate_dependencies(self) -> bool:
        """
        Validate that the order has no circular dependencies.
        
        Returns:
            bool: True if valid, False if cycles detected
        """
        visited = set()
        rec_stack = set()
        
        def has_cycle(activity_id: int) -> bool:
            if activity_id in rec_stack:
                return True  # Back edge found - cycle detected
            if activity_id in visited:
                return False  # Already processed
            
            visited.add(activity_id)
            rec_stack.add(activity_id)
            
            activity = self.activity_map[activity_id]
            if activity.next_activity:
                if has_cycle(activity.next_activity.id):
                    return True
            
            rec_stack.remove(activity_id)
            return False
        
        # Check each activity for cycles
        for activity in self.activities:
            if has_cycle(activity.id):
                return False
        
        return True
    
    def visualize(self, title="Order Dependencies Graph", figsize=(12, 8), 
                  node_size=2000, font_size=10, show_pause_values=True):
        """
        Visualize the order as a directed graph showing dependencies between activities.
        
        Args:
            title (str): Title for the plot
            figsize (tuple): Figure size as (width, height)
            node_size (int): Size of nodes in the graph
            font_size (int): Font size for labels
            show_pause_values (bool): Whether to show pause values on edges
        """
        if not self.activities:
            print("No activities to visualize")
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes (activities)
        for activity in self.activities:
            G.add_node(activity.id, 
                       duration=activity.duration, 
                       row=activity.row)
        
        # Add edges (dependencies) with weights for thickness
        edge_weights = []
        for activity in self.activities:
            if activity.next_activity:
                G.add_edge(activity.id, activity.next_activity.id, 
                           min_pause=activity.min_pause,
                           max_pause=activity.max_pause)
                edge_weights.append(activity.max_pause)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Use hierarchical layout for better dependency visualization
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes with different colors based on row
        node_colors = [G.nodes[node]['row'] for node in G.nodes()]
        
        # Create a colormap for rows
        cmap = plt.cm.tab10
        norm = plt.Normalize(min(node_colors), max(node_colors))
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              cmap=cmap,
                              node_size=node_size,
                              alpha=0.8,
                              edgecolors='black',
                              linewidths=2)
        
        # Draw edges with thickness based on max pause
        if edge_weights:
            # Normalize edge weights for better visualization
            min_weight = min(edge_weights)
            max_weight = max(edge_weights)
            if max_weight > min_weight:
                normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 3 + 1 
                                    for w in edge_weights]
            else:
                normalized_weights = [2] * len(edge_weights)
        else:
            normalized_weights = [1] * len(edge_weights)
        
        nx.draw_networkx_edges(G, pos, 
                              edge_color='blue',
                              width=normalized_weights,
                              alpha=0.7,
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->')
        
        # Add node labels with activity information
        labels = {}
        for node in G.nodes():
            duration = G.nodes[node]['duration']
            row = G.nodes[node]['row']
            labels[node] = f'A{node}\n{duration}t\nR{row}'
        
        nx.draw_networkx_labels(G, pos, labels, 
                               font_size=font_size, 
                               font_weight='bold',
                               font_color='white')
        
        # Add edge labels for pause values
        if show_pause_values:
            edge_labels = {}
            for activity in self.activities:
                if activity.next_activity:
                    edge_labels[(activity.id, activity.next_activity.id)] = \
                        f'{activity.min_pause}-{activity.max_pause}'
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                        font_size=font_size-2,
                                        font_color='red',
                                        bbox=dict(boxstyle='round,pad=0.3', 
                                                facecolor='white', 
                                                alpha=0.8))
        
        # Customize the plot
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        # Add legend for row colors
        legend_elements = []
        unique_rows = sorted(set(node_colors))
        for row in unique_rows:
            color = cmap(norm(row))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10,
                                            label=f'Row {row}'))
        
        plt.legend(handles=legend_elements, title='Activity Rows', 
                  loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add information box
        info_text = f'Total Activities: {len(self.activities)}\n'
        info_text += f'Total Duration: {self.get_total_duration()}\n'
        critical_path, crit_duration = self.get_critical_path()
        info_text += f'Critical Path: {crit_duration}'
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self):
        """String representation of the order."""
        if not self.activities:
            return "Order: Empty"
        
        execution_order = self.get_execution_order()
        result = f"Order with {len(self.activities)} activities:\n"
        
        for i, activity in enumerate(execution_order):
            result += f"  {i+1}. {activity}\n"
        
        return result
    
    def __repr__(self):
        """Detailed representation of the order."""
        return self.__str__()


# Example usage and testing
if __name__ == "__main__":
    # Create an order
    order = Order()
    
    # Populate with activities
    print("Creating order with 8 activities...")
    order.populate(num_activities=8, max_time=20, max_row=5, min_pause=3, max_pause=15)
    
    # Display the order
    print(order)
    
    # Validate dependencies
    is_valid = order.validate_dependencies()
    print(f"\nDependencies valid: {is_valid}")
    
    # Show execution order
    execution_order = order.get_execution_order()
    print(f"\nExecution order: {' -> '.join(f'A{act.id}' for act in execution_order)}")
    
    # Show critical path
    critical_path, total_duration = order.get_critical_path()
    print(f"\nCritical path: {' -> '.join(f'A{act.id}' for act in critical_path)}")
    print(f"Critical path duration: {total_duration}")
    
    # Show total duration
    total = order.get_total_duration()
    print(f"Total activity duration: {total}")
    
    # Test with different parameters
    print("\n" + "="*50)
    print("Testing with different parameters...")
    
    order2 = Order()
    order2.populate(num_activities=5, max_time=10, max_row=3, min_pause=1, max_pause=8)
    print(order2)
    
    is_valid2 = order2.validate_dependencies()
    print(f"Dependencies valid: {is_valid2}")
    
    # Visualize the orders
    print("\n" + "="*50)
    print("Visualizing orders as dependency graphs...")
    
    # Visualize the first order
    print("Visualizing Order 1 (8 activities)...")
    order.visualize(title="Order 1 - 8 Activities with Dependencies")
    
    # Visualize the second order
    print("Visualizing Order 2 (5 activities)...")
    order2.visualize(title="Order 2 - 5 Activities with Dependencies", 
                     figsize=(10, 6), show_pause_values=False)
