import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import tube_search
import time
from queue import Queue
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Patch

# Constants
STYLE = {
    'BACKGROUND': '#1C1C1C',
    'BUTTON_COLOR': '#3C3C3C',
    'BUTTON_HOVER': '#4C4C4C',
    'TEXT_COLOR': 'white',
    'NODE_SIZE': 1200,
    'FONT_SIZE': 10,
    'EDGE_WIDTH': 2.0,
}

# Define London Underground line colors
LINE_COLORS = {
    'Bakerloo': '#B36305',
    'Central': '#E32017',
    'Circle': '#FFD300',
    'District': '#00782A',
    'Hammersmith & City': '#F3A9BB',
    'Jubilee': '#A0A5A9',
    'Metropolitan': '#9B0056',
    'Northern': '#000000',
    'Piccadilly': '#003688',
    'Victoria': '#0098D4',
    'Waterloo & City': '#95CDBA'
}

NODE_COLORS = {
    'start': '#32CD32',    # Lime green
    'end': '#4169E1',      # Royal blue
    'current': '#FF4500',  # Orange-red
    'default': '#808080'   # Gray
}

def create_tube_graph(csv_file):
    """Create both C++ and NetworkX graphs from CSV data."""
    data = tube_search.readCSV(csv_file)
    cpp_graph = tube_search.Graph()
    nx_graph = nx.Graph()
    
    for row in data:
        if len(row) >= 6:
            src, dest, line, time = row[0], row[1], row[2], int(row[3])
            cpp_graph.addEdge(src, dest, time, line, int(row[4]), int(row[5]))
            nx_graph.add_edge(src, dest, weight=time, line=line)
    
    return cpp_graph, nx_graph

def adjust_layout(pos, min_dist=1.5, max_iterations=20):
    """Adjust node positions to maintain minimum distance."""
    for _ in range(max_iterations):
        moved = False
        for n1 in pos:
            for n2 in pos:
                if n1 < n2:
                    x1, y1 = pos[n1]
                    x2, y2 = pos[n2]
                    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if dist < min_dist:
                        angle = np.arctan2(y2 - y1, x2 - x1)
                        push = (min_dist - dist) / 2
                        pos[n1] = (x1 - push * np.cos(angle), y1 - push * np.sin(angle))
                        pos[n2] = (x2 + push * np.cos(angle), y2 + push * np.sin(angle))
                        moved = True
            if not moved:
                break
    return pos

def get_available_stations(G):
    """Return sorted list of all available stations."""
    return sorted(list(G.nodes()))

def print_available_stations(stations):
    """Print available stations in a formatted way."""
    print("\nAvailable Stations:")
    print("==================")
    col_width = 25
    num_cols = 3
    for i in range(0, len(stations), num_cols):
        row = stations[i:i + num_cols]
        print("".join(station.ljust(col_width) for station in row))
    print("\n")

def get_valid_station_input(prompt, available_stations):
    """Get and validate station input from user."""
    while True:
        station = input(prompt).strip()
        if station in available_stations:
            return station
        print(f"\nError: '{station}' is not a valid station.")
        print("Please choose from the available stations listed above.")

def print_search_results(result, algo_name):
    """Print formatted search results."""
    print(f"\n=== {algo_name} Route ===")
    print("Path taken:", " -> ".join(result.path))
    print(f"Total time: {result.total_time} minutes")
    print(f"Nodes explored: {result.explored_nodes}")
    print("Tube lines used:", ", ".join(result.tube_lines))

class InteractiveVisualization:
    def __init__(self, G, start, end, comparison_result):
        self.setup_visualization(G, start, end, comparison_result)
        self.create_controls()
        self.animate()

    def setup_visualization(self, G, start, end, comparison_result):
        """Initialize visualization parameters and setup."""
        plt.style.use('dark_background')
        self.G = G
        self.start = start
        self.end = end
        self.comparison_result = comparison_result
        self.results = {
            'DFS': comparison_result.dfs_result,
            'BFS': comparison_result.bfs_result,
            'Improved UCS': comparison_result.improved_ucs_result,
            'Best FS': comparison_result.best_fs_result
        }
        
        self.current_algo = 'DFS'
        self.paused = True
        self.step = 0
        self.visible_nodes = {start}
        
        self.fig = plt.figure(figsize=(24, 16), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor(STYLE['BACKGROUND'])
        
        self.pos = nx.spring_layout(G, k=10.0, iterations=100, scale=4.0, seed=42)
        self.pos = adjust_layout(self.pos)

    def create_controls(self):
        """Create and setup control buttons and radio buttons."""
        # Control frame
        control_frame = plt.axes([0.85, 0.1, 0.13, 0.8])
        control_frame.set_facecolor('#2C2C2C')
        control_frame.set_alpha(0.8)
        control_frame.axis('off')
        
        # Algorithm selection
        rax = plt.axes([0.87, 0.7, 0.1, 0.15])
        self.radio = RadioButtons(rax, list(self.results.keys()), active=0)
        self._style_radio_buttons()
        
        # Store buttons as instance variables
        self.buttons = {
            'pause': self._create_button('Pause/Play', 0.5, self.toggle_pause),
            'reset': self._create_button('Reset', 0.4, self.reset),
            'step': self._create_button('Step', 0.3, self.single_step),
            'exit': self._create_button('Exit', 0.2, self.exit_program, color='#8B0000', hover='#FF0000'),
            'route': self._create_button('Show Route', 0.1, self.show_route, color='#006400', hover='#008000'),
            'compare': self._create_button('Compare', 0.6, self.show_comparison, color='#4B0082', hover='#8A2BE2')
        }

    def _style_radio_buttons(self):
        """Style radio buttons consistently."""
        for circle in self.radio.circles:
            circle.set_facecolor('#4A4A4A')
            circle.set_edgecolor('white')
        for text in self.radio.labels:
            text.set_color('white')
        self.radio.on_clicked(self.algo_changed)

    def _create_button(self, label, position, callback, color=None, hover=None):
        """Create a styled button."""
        btn = Button(plt.axes([0.87, position, 0.1, 0.05]), 
                    label,
                    color=color or STYLE['BUTTON_COLOR'],
                    hovercolor=hover or STYLE['BUTTON_HOVER'])
        btn.label.set_color(STYLE['TEXT_COLOR'])
        btn.on_clicked(callback)
        return btn

    def draw_graph(self):
        """Draw the graph with current state."""
        self.ax.clear()
        result = self.results[self.current_algo]
        
        self._draw_edges()
        self._draw_nodes(result)
        
        if self.step >= len(result.exploration_history) - 1:
            self._draw_final_path(result)
        
        self._add_stats(result)
        self._add_legend()
        
        self.ax.set_facecolor(STYLE['BACKGROUND'])
        plt.tight_layout(rect=[0, 0, 0.85, 1])

    def _draw_edges(self):
        """Draw graph edges with appropriate styling."""
        visible_edges = [(u, v) for (u, v) in self.G.edges() 
                        if u in self.visible_nodes and v in self.visible_nodes]
        
        nx.draw_networkx_edges(self.G, self.pos,
                             edge_color='#2C2C2C',
                             width=STYLE['EDGE_WIDTH'],
                             alpha=0.3,
                             ax=self.ax)
        
        for u, v in visible_edges:
            line = self.G[u][v].get('line', '')
            color = LINE_COLORS.get(line, '#808080')
            nx.draw_networkx_edges(self.G, self.pos,
                                 edgelist=[(u, v)],
                                 edge_color=color,
                                 width=STYLE['EDGE_WIDTH'],
                                 alpha=0.9,
                                 ax=self.ax)

    def _draw_nodes(self, result):
        """Draw nodes with appropriate colors and labels."""
        current_node = (result.exploration_history[min(self.step, 
                       len(result.exploration_history)-1)]
                       if result.exploration_history else None)
        
        for node in self.visible_nodes:
            if node == self.start:
                color = NODE_COLORS['start']
            elif node == self.end:
                color = NODE_COLORS['end']
            elif node == current_node:
                color = NODE_COLORS['current']
            else:
                color = NODE_COLORS['default']
            
            nx.draw_networkx_nodes(self.G, self.pos,
                                 nodelist=[node],
                                 node_color=color,
                                 node_size=STYLE['NODE_SIZE'] * 0.6,
                                 edgecolors='white',
                                 linewidths=1.5,
                                 ax=self.ax)
            
            x, y = self.pos[node]
            self.ax.text(x, y,
                        node,
                        bbox=dict(facecolor=STYLE['BACKGROUND'],
                                edgecolor='white',
                                alpha=0.8,
                                pad=3,
                                boxstyle='round,pad=0.3'),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=STYLE['FONT_SIZE'] * 0.9,
                        color='white')

    def _draw_final_path(self, result):
        """Draw the final path with highlighting."""
        path_edges = list(zip(result.path[:-1], result.path[1:]))
        nx.draw_networkx_edges(self.G, self.pos,
                             edgelist=path_edges,
                             edge_color='#FFD700',
                             width=STYLE['EDGE_WIDTH'] * 2,
                             alpha=0.8,
                             ax=self.ax)

    def _add_stats(self, result):
        """Add statistics box to visualization."""
        stats_text = (
            f"{self.current_algo} Search\n"
            f"Explored: {len(self.visible_nodes)} nodes\n"
            f"Time: {result.total_time} minutes"
        )
        self.ax.text(0.02, 0.98, stats_text,
                    transform=self.ax.transAxes,
                    bbox=dict(facecolor='#2C2C2C',
                            edgecolor='white',
                            alpha=0.8,
                            pad=5,
                            boxstyle='round,pad=0.3'),
                    fontsize=10,
                    color='white',
                    verticalalignment='top')

    def _add_legend(self):
        """Add legend to visualization."""
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, markersize=15,
                      label=label, linestyle='None')
            for label, color in [
                ('Start Station', NODE_COLORS['start']),
                ('End Station', NODE_COLORS['end']),
                ('Current Station', NODE_COLORS['current'])
            ]
        ]
        legend_elements.append(
            plt.Line2D([0], [0], color='#FFD700',
                      linewidth=3, label='Final Route')
        )
        
        self.ax.legend(handles=legend_elements,
                      loc='upper right',
                      bbox_to_anchor=(0.84, 0.98),
                      facecolor='#2C2C2C',
                      edgecolor='white',
                      fontsize=10)

    def algo_changed(self, label):
        """Handle algorithm change."""
        self.current_algo = label
        self.reset(None)

    def toggle_pause(self, event):
        """Toggle pause state."""
        self.paused = not self.paused

    def reset(self, event):
        """Reset visualization state."""
        self.step = 0
        self.visible_nodes = {self.start}
        self.paused = True
        self.draw_graph()

    def single_step(self, event):
        """Advance one step in visualization."""
        if self.step < len(self.results[self.current_algo].exploration_history):
            self.step += 1
            self.visible_nodes.add(
                self.results[self.current_algo].exploration_history[self.step-1]
            )
            self.draw_graph()

    def exit_program(self, event):
        """Exit the visualization."""
        plt.close('all')

    def show_route(self, event):
        """Show route details in popup window."""
        result = self.results[self.current_algo]
        route_str = " → ".join(result.path)
        time_taken = result.total_time
        
        tube_lines_info = ""
        if hasattr(result, 'tube_lines'):
            tube_lines_info = "\nTube Lines Used: " + ", ".join(result.tube_lines)
        
        fig = plt.figure(figsize=(12, 4))
        fig.patch.set_facecolor(STYLE['BACKGROUND'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(STYLE['BACKGROUND'])
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        route_text = f"Route: {route_str}\nTotal Time: {time_taken} minutes{tube_lines_info}"
        ax.text(0.5, 0.5, route_text,
               horizontalalignment='center',
               verticalalignment='center',
               color='white',
               fontsize=12,
               bbox=dict(facecolor='#2C2C2C',
                       edgecolor='white',
                       alpha=0.8,
                       pad=10,
                       boxstyle='round'))
        
        plt.show()

    def show_comparison(self, event):
        """Show comparison of all algorithms in a new window."""
        # Create new figure without affecting main window
        comparison_fig = plt.figure(figsize=(15, 10))
        comparison_fig.patch.set_facecolor(STYLE['BACKGROUND'])
        
        # Create a 2x2 grid for the four algorithms
        grid_size = (2, 2)
        
        for idx, (algo, result) in enumerate(self.results.items()):
            ax = plt.subplot(grid_size[0], grid_size[1], idx + 1)
            ax.set_facecolor(STYLE['BACKGROUND'])
            
            # Draw the route for this algorithm
            self._draw_comparison_route(ax, result, algo)
            
            # Remove axis
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        # Create a new window manager
        comparison_manager = plt.get_current_fig_manager()
        comparison_manager.window.attributes('-topmost', 1)  # Keep window on top
        
        # Show comparison window
        plt.show(block=False)
        
        # Keep reference to prevent garbage collection
        self.comparison_fig = comparison_fig

    def _draw_comparison_route(self, ax, result, algo_name):
        """Draw a single route comparison."""
        # Create a subgraph with only the path nodes
        path_nodes = result.path
        route_graph = self.G.subgraph(path_nodes)
        
        # Use the same layout positions as main graph
        pos = {node: self.pos[node] for node in path_nodes}
        
        # Draw edges with tube line colors
        path_edges = list(zip(result.path[:-1], result.path[1:]))
        for u, v in path_edges:
            line = self.G[u][v].get('line', '')
            color = LINE_COLORS.get(line, '#808080')
            nx.draw_networkx_edges(route_graph, pos,
                                 edgelist=[(u, v)],
                                 edge_color=color,
                                 width=STYLE['EDGE_WIDTH'],
                                 alpha=0.9,
                                 ax=ax)
        
        # Draw nodes
        for node in path_nodes:
            if node == self.start:
                color = NODE_COLORS['start']
            elif node == self.end:
                color = NODE_COLORS['end']
            else:
                color = NODE_COLORS['default']
            
            nx.draw_networkx_nodes(route_graph, pos,
                                 nodelist=[node],
                                 node_color=color,
                                 node_size=STYLE['NODE_SIZE'] * 0.4,
                                 edgecolors='white',
                                 linewidths=1,
                                 ax=ax)
            
            # Add station labels
            x, y = pos[node]
            ax.text(x, y,
                   node,
                   bbox=dict(facecolor=STYLE['BACKGROUND'],
                           edgecolor='white',
                           alpha=0.8,
                           pad=2,
                           boxstyle='round,pad=0.2'),
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=STYLE['FONT_SIZE'] * 0.7,
                   color='white')
        
        # Add algorithm info
        stats_text = (
            f"{algo_name}\n"
            f"Time: {result.total_time}min\n"
            f"Nodes: {result.explored_nodes}"
        )
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='#2C2C2C',
                        edgecolor='white',
                        alpha=0.8,
                        pad=3,
                        boxstyle='round,pad=0.2'),
                fontsize=8,
                color='white',
                verticalalignment='top')

    def animate(self):
        """Main animation loop."""
        plt.ion()  # Turn on interactive mode
        self.draw_graph()  # Initial draw
        
        try:
            while plt.fignum_exists(self.fig.number):
                if not self.paused:
                    if self.step < len(self.results[self.current_algo].exploration_history):
                        self.step += 1
                        self.visible_nodes.add(
                            self.results[self.current_algo].exploration_history[self.step-1]
                        )
                        if self.step == len(self.results[self.current_algo].exploration_history):
                            self.visible_nodes.add(self.end)
                        self.draw_graph()
                
                plt.pause(0.1)
                
        except Exception as e:
            print(f"Animation stopped: {e}")
        finally:
            plt.ioff()  # Turn off interactive mode

def main():
    """Main program function."""
    cpp_graph, nx_graph = create_tube_graph("tubedata_test.csv")
    available_stations = get_available_stations(nx_graph)
    
    print("\nWelcome to London Underground Journey Planner!")
    print_available_stations(available_stations)
    
    start = get_valid_station_input("Enter starting station: ", available_stations)
    end = get_valid_station_input("Enter destination station: ", available_stations)
    
    print("Please wait while we calculate routes...\n")
    
    comparison_result = tube_search.parallel_search(cpp_graph, start, end)
    
    # Print results for each algorithm
    for algo, result in [
        ('DFS', comparison_result.dfs_result),
        ('BFS', comparison_result.bfs_result),
        ('Improved UCS', comparison_result.improved_ucs_result),
        ('Best First Search', comparison_result.best_fs_result)
    ]:
        print_search_results(result, algo)
    
    print("\nDisplaying visual comparison...")
    
    InteractiveVisualization(nx_graph, start, end, comparison_result)

if __name__ == "__main__":
    main() 