///
/// author: grainme
///
/// subject: Implementing A*
/// idea: balancing what we know with what we estimate to make optimal decisions.
/// date: 17 april 2025
///
///
use ndarray::Array2;
use ordered_float::OrderedFloat;
use plotters::prelude::*;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    fn new(x: i32, y: i32) -> Self {
        Position { x, y }
    }

    fn validate_position(&self, grid: &Array2<u32>) -> bool {
        self.x >= 0 && self.y >= 0 && *grid.get((self.x as usize, self.y as usize)).unwrap() == 0
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct Node {
    position: Position,
    from_start: OrderedFloat<f32>,
    to_target: OrderedFloat<f32>,
    parent: Box<Option<Node>>,
}

/// Create a node for the A* algorithm.
///    Args:
///     position: (x, y) coordinates of the node
///     g: Cost from start to this node (default: infinity)
///     h: Estimated cost from this node to goal (default: 0)
///     parent: Parent node (default: None)
impl Node {
    // we could have used lifetimes here but...
    // i am quite lazy!
    fn new(position: Position, g: Option<f32>, h: Option<f32>, parent: Option<Node>) -> Self {
        Node {
            position,
            from_start: OrderedFloat(g.unwrap_or(f32::INFINITY)),
            to_target: OrderedFloat(h.unwrap_or(0.0)),
            parent: Box::new(parent),
        }
    }

    // Get all valid neighboring positions in the grid.
    // Args:
    //    grid: 2D array where 0 represents walkable cells and 1 represents obstacles
    //    position: Current position (x, y)
    // Returns:
    //    List of valid neighboring positions
    fn get_neighbors(&self) -> Vec<Position> {
        let position = &self.position;
        vec![
            Position::new(position.x - 1, position.y),
            Position::new(position.x, position.y - 1),
            Position::new(position.x - 1, position.y - 1),
            Position::new(position.x + 1, position.y),
            Position::new(position.x + 1, position.y + 1),
            Position::new(position.x, position.y + 1),
            Position::new(position.x - 1, position.y + 1),
            Position::new(position.x + 1, position.y - 1),
        ]
    }

    fn get_valid_neighbors(&self, grid: &Array2<u32>) -> Vec<Position> {
        let mut neighbors = self.get_neighbors();
        neighbors.retain(|e| e.validate_position(&grid));

        neighbors
    }
}

/// calculate the estimated distance between two points using the manhattan distance.
fn calculate_heuristic(pos1: &Position, pos2: &Position) -> f32 {
    // we can use other metrics than Manhattan.
    ((pos1.x - pos2.x).abs() + (pos1.y - pos2.y).abs()) as f32
}

// Reconstruct the path from goal to start by following parent pointers.
fn reconstruct_path(node: Option<Node>) -> Vec<Position> {
    let mut path: Vec<Position> = Vec::new();
    let mut current: Option<Node> = node;
    while current.is_some() {
        let current_node = current.unwrap();
        path.push(current_node.position);
        current = *current_node.parent;
    }
    path.reverse();

    path
}

// Find the optimal path using A* algorithm.
//
// Args:
//    grid: 2D numpy array (0 = free space, 1 = obstacle)
//    start: Starting position (x, y)
//    goal: Goal position (x, y)
//
// Returns:
//    List of positions representing the optimal path
#[allow(unused)]
fn find_path(grid: &Array2<u32>, start: Position, target: Position) -> Option<Vec<Position>> {
    // init starting node
    let start_node = Node::new(
        start,
        Some(0.0),
        Some(calculate_heuristic(&start, &target)),
        None, // it has no parent, it's the starting node
    );

    // priority_queue, sorted by (already_traveled + need_to_travel)
    // Using a min-heap by negating the f-score
    let mut open_queue: BinaryHeap<(OrderedFloat<f32>, Position)> = BinaryHeap::new();

    // for fast lookup we define a hashmap where the key is the node position
    // and the value is the node
    let mut open_map: HashMap<Position, Node> = HashMap::new();

    // keeping track of the already explored nodes
    let mut closed_set: HashSet<Position> = HashSet::new();

    open_queue.push((
        OrderedFloat(-(*start_node.from_start + *start_node.to_target)),
        start,
    ));
    open_map.insert(start, start_node);

    while !open_queue.is_empty() {
        // Get the node with lowest f_score (g + h)
        let (_, current_position) = open_queue.pop().unwrap();

        // Get the node from the map
        if !open_map.contains_key(&current_position) {
            continue; // This can happen if we've found a better path to this node already
        }

        let current_node = open_map.remove(&current_position).unwrap();

        // Check if we reached the goal
        if current_node.position == target {
            return Some(reconstruct_path(Some(current_node)));
        }
        closed_set.insert(current_position);

        // Explore neighbors
        let neighbors = current_node.get_valid_neighbors(grid);

        for neighbor_pos in neighbors {
            // Skip if already in closed set
            if closed_set.contains(&neighbor_pos) {
                continue;
            }

            let tentative_g =
                *current_node.from_start + calculate_heuristic(&current_position, &neighbor_pos);

            let mut is_better = false;

            // Check if this is a new node or if we found a better path
            if !open_map.contains_key(&neighbor_pos) {
                is_better = true;
            } else if tentative_g < *open_map.get(&neighbor_pos).unwrap().from_start {
                is_better = true;
            }

            if is_better {
                let h_value = calculate_heuristic(&neighbor_pos, &target);

                let neighbor_node = Node::new(
                    neighbor_pos,
                    Some(tentative_g),
                    Some(h_value),
                    Some(current_node.clone()),
                );

                let f_score = tentative_g + h_value;
                open_queue.push((OrderedFloat(-f_score), neighbor_pos));
                open_map.insert(neighbor_pos, neighbor_node);
            }
        }
    }

    None // No path found
}

fn visualize_path(grid: &Array2<u32>, path: &[Position]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("result.png", (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let (rows, cols) = grid.dim();
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..cols, 0..rows)?;

    for row in 0..rows {
        for col in 0..cols {
            let color = match grid[[row, col]] {
                1 => BLACK, // obstacle
                _ => WHITE, // free space
            };
            chart.draw_series(std::iter::once(Rectangle::new(
                [(col, row), (col + 1, row + 1)],
                color.filled(),
            )))?;
        }
    }

    if !path.is_empty() {
        chart.draw_series(LineSeries::new(
            path.iter().map(|pos| (pos.y as usize, pos.x as usize)),
            &BLUE,
        ))?;

        let Position { x: sr, y: sc } = path[0];
        chart.draw_series(std::iter::once(Circle::new(
            (sc as usize, sr as usize),
            5,
            GREEN.filled(),
        )))?;

        let Position { x: gr, y: gc } = path[path.len() - 1];
        chart.draw_series(std::iter::once(Circle::new(
            (gc as usize, gr as usize),
            5,
            RED.filled(),
        )))?;
    }

    root.present()?;
    println!("Path visualization saved as 'path_result.png'");

    Ok(())
}

fn main() {
    let mut grid = Array2::<u32>::zeros((20, 20));

    for row in 5..15 {
        grid[[row, 10]] = 1;
    }
    for col in 5..15 {
        grid[[5, col]] = 1;
    }

    grid[[10, 10]] = 0;

    // Print grid for debugging
    println!("Grid:");
    for row in 0..grid.shape()[0] {
        for col in 0..grid.shape()[1] {
            print!("{} ", grid[[row, col]]);
        }
        println!();
    }

    let start_pos = Position::new(2, 2);
    let goal_pos = Position::new(18, 18);

    let path = find_path(&grid, start_pos, goal_pos);

    match path {
        Some(path) => {
            println!("Path found with {} steps!", path.len());
            println!("Path: {:?}", path);

            if let Err(e) = visualize_path(&grid, &path) {
                eprintln!("Failed to visualize path: {}", e);
            }
        }
        None => {
            println!("No path found!");
        }
    }
}
