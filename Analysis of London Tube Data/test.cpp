#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <queue>
#include <stack>
#include <set>
#include <list>
#include <thread>
#include <future>
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <algorithm>

namespace py = pybind11;
using namespace std;


// Function to read CSV file into a 2D vector
vector<vector<string>> readCSV(const string& filename) {
    vector<vector<string>> data;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<string> row;

        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

// Structure to represent an edge in the graph
struct Edge {
    string destination;
    int time;
    string line_of_tube;
    string zone1;
    string zone2;
    
    Edge(const string& dest, int t, const string& line, const string& z1, const string& z2="0") 
        : destination(dest), time(t), line_of_tube(line), zone1(z1), zone2(z2) {}
};

// Graph representation using an adjacency list
class Graph {
public:
    unordered_map<string, list<Edge>> adjList;

    void addEdge(const string& src, const string& dest, int time, 
                 const string& line, const string& z1, const string& z2="0") {
        adjList[src].push_back(Edge(dest, time, line, z1, z2));
        adjList[dest].push_back(Edge(src, time, line, z1, z2));
    }

    // Add overload for when zones are passed as integers
    void addEdge(const string& src, const string& dest, int time,
                 const string& line, int z1, int z2=0) {
        // Convert integers to strings
        string zone1 = to_string(z1);
        string zone2 = to_string(z2);
        addEdge(src, dest, time, line, zone1, zone2);
    }
};

// Structure to represent a node for pathfinding
struct Node {
    string state;
    Node* parent;
    int time;
    string tube_line;
    string m_zone;    
    string s_zone;    
    double heuristic; 
    double total_cost;

    // Basic constructor for DFS, BFS, UCS
    Node(const string& s, Node* p, int t, const string& line = "") 
        : state(s), parent(p), time(t), tube_line(line),
          m_zone("1"), s_zone("1"), heuristic(0.0),total_cost(0.0) {}
    
    // Extended constructor for Best First Search
    Node(const string& s, Node* p, int t, const string& line,
         const string& mz, const string& sz, double h,double c) 
        : state(s), parent(p), time(t), tube_line(line),
          m_zone(mz), s_zone(sz), heuristic(h), total_cost(c) {}
    
    virtual ~Node() = default;
};

// Internal unified result structure
struct SearchResultCommon {
    vector<string> path;
    int total_time;
    int explored_nodes;
    vector<string> exploration_history;
    set<string> tube_lines;
    
    SearchResultCommon() : total_time(0), explored_nodes(0) {}
    
    SearchResultCommon(vector<string> p, int t, int n, vector<string> h, set<string> lines) 
        : path(p), total_time(t), explored_nodes(n), 
          exploration_history(h), tube_lines(lines) {}
};

// Helper function for all search algorithms
SearchResultCommon Path_Time_taken(Node* curr_node) {
    vector<string> final_route;
    int total_time = 0;
    set<string> tube_lines;

    while (curr_node->parent != nullptr) {
        final_route.push_back(curr_node->state);
        total_time += curr_node->time;
        if (!curr_node->tube_line.empty()) {
            tube_lines.insert(curr_node->tube_line);
        }
        curr_node = curr_node->parent;
    }

    final_route.push_back(curr_node->state);
    reverse(final_route.begin(), final_route.end());

    return SearchResultCommon(final_route, total_time, 0, vector<string>(), tube_lines);
}

// Different types for Python bindings
struct DFSResult : public SearchResultCommon {
    using SearchResultCommon::SearchResultCommon;  // Inherit constructors
};

struct BFSResult : public SearchResultCommon {
    using SearchResultCommon::SearchResultCommon;
};

struct UCSResult : public SearchResultCommon {
    using SearchResultCommon::SearchResultCommon;
};

struct BestFSResult : public SearchResultCommon {
    using SearchResultCommon::SearchResultCommon;
};


DFSResult depth_first_search(
    Graph& graph, const string& initial, const string& goal, bool check_reverse = false) {
    
    stack<Node*> frontier;
    frontier.push(new Node(initial, nullptr, 0));
    
    set<string> explored;
    explored.insert(initial);
    
    vector<string> exploration_history;  // Track exploration order
    int number_of_explored_nodes = 0;

    while (!frontier.empty()) {
        Node* curr_node = frontier.top();
        frontier.pop();
        number_of_explored_nodes++;
        
        // Add current node to exploration history
        exploration_history.push_back(curr_node->state);

        if (curr_node->state == goal) {
            auto result = Path_Time_taken(curr_node);
            return DFSResult(result.path, result.total_time, 
                              number_of_explored_nodes, exploration_history, result.tube_lines);
        }

        // Get neighbors of the current node
        list<Edge> neighbors = graph.adjList[curr_node->state];
        vector<Edge> children(neighbors.begin(), neighbors.end());

        // Reverse the neighbors if check_reverse is true
        if (check_reverse) {
            reverse(children.begin(), children.end());
        }

        // Iterate through the neighbors
        for (const auto& child : children) {
            if (explored.find(child.destination) == explored.end()) {
                // Create a new node and add it to the frontier
                Node* child_node = new Node(child.destination, curr_node, child.time, child.line_of_tube);
                frontier.push(child_node);
                explored.insert(child.destination);
            }
        }
    }

    return DFSResult(vector<string>(), 0, number_of_explored_nodes, exploration_history, set<string>());
}

BFSResult breadth_first_search(Graph& graph, const string& initial, const string& goal, bool check_reverse = false) {
    queue<Node*> frontier;
    frontier.push(new Node(initial, nullptr, 0));
    
    set<string> explored;
    explored.insert(initial);
    
    int number_of_explored_nodes = 0;
    vector<string> exploration_history;
    
    while (!frontier.empty()) {
        Node* curr_node = frontier.front();
        frontier.pop();
        number_of_explored_nodes++;
        exploration_history.push_back(curr_node->state);
        
        if (curr_node->state == goal) {
            auto result = Path_Time_taken(curr_node);
            return BFSResult(result.path, result.total_time, 
                           number_of_explored_nodes, exploration_history, result.tube_lines);
        }
        
        vector<Edge> children(graph.adjList[curr_node->state].begin(),
                            graph.adjList[curr_node->state].end());
        if (check_reverse) {
            reverse(children.begin(), children.end());
        }
        
        for (const Edge& child : children) {
            if (explored.find(child.destination) == explored.end()) {
                Node* child_node = new Node(child.destination, curr_node, child.time, child.line_of_tube);
                frontier.push(child_node);
                explored.insert(child.destination);
            }
        }
    }
    
    return BFSResult(vector<string>(), 0, number_of_explored_nodes, exploration_history, set<string>());
}


// Add improved UCS implementation
UCSResult improved_uniform_cost_search(Graph& graph, const string& start, const string& end) {
    // Use priority queue ordered by time (cost)
    auto compare = [](Node* a, Node* b) { return a->time > b->time; };
    priority_queue<Node*, vector<Node*>, decltype(compare)> frontier(compare);
    
    Node* start_node = new Node(start, nullptr, 0, "");
    frontier.push(start_node);
    
    // Track explored stations and their minimum times
    map<string, int> explored;  // station -> minimum time seen
    explored[start] = 0;
    
    int number_of_explored_nodes = 0;
    vector<string> exploration_history;
    
    while (!frontier.empty()) {
        Node* curr_node = frontier.top();
        frontier.pop();
        
        // Skip if we've found a better path to this station
        if (explored.count(curr_node->state) && explored[curr_node->state] < curr_node->time) {
            continue;
        }
        
        number_of_explored_nodes++;
        exploration_history.push_back(curr_node->state);
        
        if (curr_node->state == end) {
            auto result = Path_Time_taken(curr_node);
            return UCSResult(result.path, curr_node->time, 
                                   number_of_explored_nodes, exploration_history,
                                   result.tube_lines);
        }
        
        // Process all neighbors
        for (const Edge& child : graph.adjList[curr_node->state]) {
            int new_time = curr_node->time + child.time;
            
            // Add line change penalty if changing lines
            if (!curr_node->tube_line.empty() && curr_node->tube_line != child.line_of_tube) {
                new_time += 2;  // 2-minute penalty for changing lines
            }
            
            // Only add if we haven't seen this station or if this is a better path
            if (!explored.count(child.destination) || new_time < explored[child.destination]) {
                Node* child_node = new Node(child.destination, curr_node, new_time, child.line_of_tube);
                frontier.push(child_node);
                explored[child.destination] = new_time;
            }
        }
    }
    
    return UCSResult(vector<string>(), 0, number_of_explored_nodes, 
                            exploration_history, set<string>());
}


pair<string, string> calculate_zone(Graph& graph, const string& aim) {
    string main_zone = "1";  // Default value
    string secon_zone = "1"; // Default value
    bool res = false;
    string temp;

    // Iterate through neighbors until we find one with a non-zero secondary zone
    for (const auto& edge : graph.adjList[aim]) {
        temp = edge.destination;
        if (edge.zone2 != "0") {
            res = true;
            break;
        }
    }

    // If we found a node with secondary zone
    if (res) {
        // Find the edge that matches temp
        for (const auto& edge : graph.adjList[aim]) {
            if (edge.destination == temp) {
                main_zone = edge.zone1;
                secon_zone = edge.zone2;
                break;
            }
        }
    } else {
        // If no secondary zone found, both zones are the same as the main zone
        for (const auto& edge : graph.adjList[aim]) {
            if (edge.destination == temp) {
                main_zone = edge.zone1;
                secon_zone = main_zone;  // Set secondary zone same as main zone
                break;
            }
        }
    }

    return {main_zone, secon_zone};
}

double calc_weight_zone(Graph& graph, const string& node, 
                       const string& goal_m_zone, const string& goal_s_zone) {
    // Map for zone values
    map<string, int> zone_data = {
        {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5},
        {"6", 6}, {"7", 7}, {"8", 8}, {"9", 9}, {"10", 10}
    };

    // Get the zones for the current node
    auto [child_m_zone, child_s_zone] = calculate_zone(graph, node);

    // Map goal zones to numerical values
    int g_m_zone = zone_data[goal_m_zone];
    int g_s_zone = zone_data[goal_s_zone];

    // Map child zones to numerical values
    int c_m_zone = zone_data[child_m_zone];
    int c_s_zone = zone_data[child_s_zone];

    double cost = 0.0;

    // If the child node and goal node are in the same zone, no penalty
    if (c_m_zone == g_m_zone || c_s_zone == g_s_zone || 
        c_m_zone == g_s_zone || c_s_zone == g_m_zone) {
        return cost;  // No penalty if in the same zone
    }

    // If the child node is moving towards the goal, apply a smaller penalty
    if (g_m_zone > c_m_zone || g_s_zone > c_s_zone || 
        g_m_zone > c_s_zone || g_s_zone > c_m_zone) {
        if (g_m_zone == c_m_zone + 1 || g_s_zone == c_s_zone + 1 || 
            g_m_zone == c_s_zone + 1 || g_s_zone == c_m_zone + 1) {
            cost += 0.5;
        } else {
            cost += 1.0;
        }
    } else {
        // If the child node is moving away from the goal, apply a higher penalty
        if (g_m_zone == c_m_zone - 1 || g_s_zone == c_s_zone - 1 || 
            g_m_zone == c_s_zone - 1 || g_s_zone == c_m_zone - 1) {
            cost += 0.5;
        } else {
            cost += 1.0;
        }
    }

    return cost;
}

BestFSResult best_first_search(Graph& graph, const string& start, const string& end) {
    // Calculate goal zones for heuristic calculation
    auto [goal_m_zone, goal_s_zone] = calculate_zone(graph, end);
    auto [initial_m_zone, initial_s_zone] = calculate_zone(graph, start);

    // Initialize with start node
    double start_heuristic = calc_weight_zone(graph, start, goal_m_zone, goal_s_zone);
    vector<Node*> frontier;
    frontier.push_back(new Node(start, nullptr, 0,"", 
                                  initial_m_zone, initial_s_zone,start_heuristic,0.0));

    set<string> explored;
    explored.insert(start);

    int number_of_explored_nodes = 0;
    vector<string> exploration_history;

    while (!frontier.empty()) {
        sort(frontier.begin(), frontier.end(),
             [](Node* a, Node* b) { 
                 if (a->heuristic == b->heuristic)
                     return a->total_cost < b->total_cost;
                 return a->heuristic < b->heuristic;
             });

        Node* curr_node = frontier.front();
        frontier.erase(frontier.begin());
        number_of_explored_nodes++;
        exploration_history.push_back(curr_node->state);

        if (curr_node->state == end) {
            auto result = Path_Time_taken(curr_node);
            
            return BestFSResult(result.path, curr_node->time, number_of_explored_nodes, 
                              exploration_history, result.tube_lines);
        }

        vector<tuple<string, double, int, double, string, string, string>> cost_list;  // Added string for tube line
        
        // Only process unexplored neighbors
        for (const Edge& edge : graph.adjList[curr_node->state]) {
            // Check explored set before processing
            if (explored.find(edge.destination) == explored.end()) {
                auto [child_m_zone, child_s_zone] = calculate_zone(graph, edge.destination);
                double heuristic = calc_weight_zone(graph, edge.destination, goal_m_zone, goal_s_zone);
                int new_time = curr_node->time + edge.time;
                double total_cost = heuristic + new_time;
                
                cost_list.push_back({edge.destination, heuristic, new_time, total_cost,
                                   child_m_zone, child_s_zone, edge.line_of_tube});  // Store tube line
                
                // Mark as explored immediately after processing
                explored.insert(edge.destination);
            }
        }
        
        // Sort cost list
        sort(cost_list.begin(), cost_list.end(),
             [](const auto& a, const auto& b) { 
                 if (get<1>(a) == get<1>(b))
                     return get<3>(a) < get<3>(b);
                 return get<1>(a) < get<1>(b);
             });

        // Add to frontier with tube line information
        for (const auto& [state, h, t, total, mz, sz, line] : cost_list) {
            frontier.push_back(new Node(state, curr_node, t,line, mz, sz, h,h+t));
        }
    }

    return BestFSResult(vector<string>(), 0, number_of_explored_nodes, 
                       exploration_history, set<string>());
}


// Update ComparisonResult
struct ComparisonResult {
    DFSResult dfs_result;
    BFSResult bfs_result;
    UCSResult improved_ucs_result;
    BestFSResult best_fs_result;
    
    ComparisonResult(DFSResult dfs, BFSResult bfs, 
                    UCSResult imp_ucs, BestFSResult best_fs) 
        : dfs_result(dfs), bfs_result(bfs),
          improved_ucs_result(imp_ucs), best_fs_result(best_fs) {}
};

// Mutex for thread-safe console output
std::mutex console_mutex;


// Thread-safe console output function
void thread_safe_print(const std::string& message) {
    std::lock_guard<std::mutex> lock(console_mutex);
    std::cout << message << std::endl;
}

// Update parallel_search to include UCS
ComparisonResult parallel_search(Graph& graph, const string& start, const string& end) {
    promise<DFSResult> dfs_promise;
    promise<BFSResult> bfs_promise;
    promise<UCSResult> ucs_promise;
    promise<BestFSResult> best_fs_promise;
    
    auto dfs_future = dfs_promise.get_future();
    auto bfs_future = bfs_promise.get_future();
    auto ucs_future = ucs_promise.get_future();
    auto best_fs_future = best_fs_promise.get_future();
    
    thread dfs_thread([&]() {
        try {
            thread_safe_print("DFS search started...");
            auto result = depth_first_search(graph, start, end, false);
            dfs_promise.set_value(result);
            thread_safe_print("DFS search completed.");
        } catch (...) {
            dfs_promise.set_exception(current_exception());
        }
    });
    
    thread bfs_thread([&]() {
        try {
            thread_safe_print("BFS search started...");
            auto result = breadth_first_search(graph, start, end, false);
            bfs_promise.set_value(result);
            thread_safe_print("BFS search completed.");
        } catch (...) {
            bfs_promise.set_exception(current_exception());
        }
    });
    
    thread ucs_thread([&]() {
        try {
            thread_safe_print("Improved UCS search started...");
            auto result = improved_uniform_cost_search(graph, start, end);
            ucs_promise.set_value(result);
            thread_safe_print("Improved UCS search completed.");
        } catch (...) {
            ucs_promise.set_exception(current_exception());
        }
    });
    
    thread best_fs_thread([&]() {
        try {
            thread_safe_print("Best First Search started...");
            auto result = best_first_search(graph, start, end);
            best_fs_promise.set_value(result);
            thread_safe_print("Best First Search completed.");
        } catch (...) {
            best_fs_promise.set_exception(current_exception());
        }
    });
    
    dfs_thread.join();
    bfs_thread.join();
    ucs_thread.join();
    best_fs_thread.join();
    
    return ComparisonResult(
        dfs_future.get(),
        bfs_future.get(),
        ucs_future.get(),
        best_fs_future.get()
    );
}

/************************
 * Python Bindings
 ************************/

PYBIND11_MODULE(tube_search, m) {
    m.doc() = "London Underground Path Finding Module";

     // Make sure readCSV is properly bound
    m.def("readCSV", &readCSV, "Read tube network data from CSV file",
          py::arg("filename"));

    // Bind Edge class
    py::class_<Edge>(m, "Edge")
        .def(py::init<const string&, int, const string&, const string&, const string&>())
        .def_readwrite("destination", &Edge::destination)
        .def_readwrite("time", &Edge::time)
        .def_readwrite("line_of_tube", &Edge::line_of_tube)
        .def_readwrite("zone1", &Edge::zone1)
        .def_readwrite("zone2", &Edge::zone2);

    // Bind Graph class
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("addEdge", static_cast<void (Graph::*)(const string&, const string&, int, const string&, const string&, const string&)>(&Graph::addEdge))
        .def("addEdge", static_cast<void (Graph::*)(const string&, const string&, int, const string&, int, int)>(&Graph::addEdge));

    // Bind each result type separately
    py::class_<DFSResult>(m, "DFSResult")
        .def(py::init<vector<string>, int, int, vector<string>, set<string>>())
        .def_readwrite("path", &DFSResult::path)
        .def_readwrite("total_time", &DFSResult::total_time)
        .def_readwrite("explored_nodes", &DFSResult::explored_nodes)
        .def_readwrite("exploration_history", &DFSResult::exploration_history)
        .def_readwrite("tube_lines", &DFSResult::tube_lines);

    py::class_<BFSResult>(m, "BFSResult")
        .def(py::init<vector<string>, int, int, vector<string>, set<string>>())
        .def_readwrite("path", &BFSResult::path)
        .def_readwrite("total_time", &BFSResult::total_time)
        .def_readwrite("explored_nodes", &BFSResult::explored_nodes)
        .def_readwrite("exploration_history", &BFSResult::exploration_history)
        .def_readwrite("tube_lines", &BFSResult::tube_lines);

    py::class_<UCSResult>(m, "UCSResult")
        .def(py::init<vector<string>, int, int, vector<string>, set<string>>())
        .def_readwrite("path", &UCSResult::path)
        .def_readwrite("total_time", &UCSResult::total_time)
        .def_readwrite("explored_nodes", &UCSResult::explored_nodes)
        .def_readwrite("exploration_history", &UCSResult::exploration_history)
        .def_readwrite("tube_lines", &UCSResult::tube_lines);

    py::class_<BestFSResult>(m, "BestFSResult")
        .def(py::init<vector<string>, int, int, vector<string>, set<string>>())
        .def_readwrite("path", &BestFSResult::path)
        .def_readwrite("total_time", &BestFSResult::total_time)
        .def_readwrite("explored_nodes", &BestFSResult::explored_nodes)
        .def_readwrite("exploration_history", &BestFSResult::exploration_history)
        .def_readwrite("tube_lines", &BestFSResult::tube_lines);

    py::class_<ComparisonResult>(m, "ComparisonResult")
        .def(py::init<DFSResult, BFSResult, UCSResult, BestFSResult>())
        .def_readwrite("dfs_result", &ComparisonResult::dfs_result)
        .def_readwrite("bfs_result", &ComparisonResult::bfs_result)
        .def_readwrite("improved_ucs_result", &ComparisonResult::improved_ucs_result)
        .def_readwrite("best_fs_result", &ComparisonResult::best_fs_result);

    // Bind search functions
    m.def("depth_first_search", &depth_first_search, 
          "Perform DFS search",
          py::arg("graph"), py::arg("start"), py::arg("end"), py::arg("verbose") = false);
    
    m.def("breadth_first_search", &breadth_first_search,
          "Perform BFS search",
          py::arg("graph"), py::arg("start"), py::arg("end"), py::arg("verbose") = false);
    
    m.def("improved_uniform_cost_search", &improved_uniform_cost_search,
          "Perform improved UCS search with tube line changes",
          py::arg("graph"), py::arg("start"), py::arg("end"));
    
    m.def("best_first_search", &best_first_search,
          "Perform Best First Search with zone-based heuristics",
          py::arg("graph"), py::arg("start"), py::arg("end"));
    
    m.def("parallel_search", &parallel_search,
          "Perform all searches in parallel",
          py::arg("graph"), py::arg("start"), py::arg("end"));

   
}
