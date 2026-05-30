//! Engine — orchestrates the Grand Pattern: graph construction, gossip rounds, GC cycles.

use crate::*;
use crate::room::Room;
use std::collections::HashMap;

/// The Grand Pattern engine: manages a cellular graph of rooms.
pub struct Engine<const D: usize> {
    pub graph: Graph,
    pub rooms: HashMap<RoomId, Room<D>>,
    pub global_tick: u64,
}

impl<const D: usize> Engine<D> {
    pub fn new() -> Self {
        Engine {
            graph: Graph::new(),
            rooms: HashMap::new(),
            global_tick: 0,
        }
    }

    /// Add a room with default JEPA learning rate.
    pub fn add_room(&mut self, id: RoomId) {
        self.add_room_with_lr(id, 0.1);
    }

    pub fn add_room_with_lr(&mut self, id: RoomId, lr: f64) {
        self.graph.add_room(id);
        self.rooms.insert(id, Room::new(id, lr));
    }

    /// Connect two rooms bidirectionally.
    pub fn connect(&mut self, a: RoomId, b: RoomId, weight: f64) {
        self.graph.add_edge(a, b, weight);
        self.graph.add_edge(b, a, weight);
    }

    /// Feed a perception into a specific room.
    pub fn perceive(&mut self, room_id: RoomId, v: Embed<D>) {
        if let Some(room) = self.rooms.get_mut(&room_id) {
            room.perceive(v);
        }
    }

    /// Run one gossip round: every room murmurs to its neighbors.
    pub fn gossip_round(&mut self) {
        self.global_tick += 1;
        // Collect murmurs first (immutable borrow phase).
        let murmurs: Vec<(RoomId, Murmur<D>)> = self.rooms.iter()
            .map(|(&id, room)| (id, room.murmur()))
            .collect();

        // Deliver murmurs (mutable borrow phase).
        for (source_id, murmur) in &murmurs {
            let neighbors = self.graph.neighbors(*source_id);
            for (neighbor_id, _weight) in neighbors {
                if let Some(room) = self.rooms.get_mut(&neighbor_id) {
                    room.hear(murmur);
                }
            }
        }
    }

    /// Run GC on all rooms.
    pub fn gc_all(&mut self, merge_threshold: f64, decay_sigma: f64, max_entries: usize) {
        for room in self.rooms.values_mut() {
            room.gc(merge_threshold, decay_sigma, max_entries);
        }
    }

    /// Compute total surprise across all rooms.
    pub fn total_surprise(&self) -> f64 {
        self.rooms.values().map(|r| r.total_surprise()).sum()
    }

    /// Check if all room ledgers are balanced.
    pub fn all_balanced(&self) -> bool {
        self.rooms.values().all(|r| r.is_balanced())
    }

    /// Run a full cycle: perceive, gossip, gc.
    pub fn cycle(&mut self, perceptions: &[(RoomId, Embed<D>)]) {
        for &(room_id, ref v) in perceptions {
            self.perceive(room_id, v.clone());
        }
        self.gossip_round();
        self.gc_all(0.1, 10.0, 64);
    }
}

impl<const D: usize> Default for Engine<D> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_add_rooms_and_connect() {
        let mut engine: Engine<4> = Engine::new();
        engine.add_room(1);
        engine.add_room(2);
        engine.add_room(3);
        engine.connect(1, 2, 0.8);
        engine.connect(2, 3, 0.5);
        assert_eq!(engine.graph.room_count(), 3);
        assert_eq!(engine.graph.edge_count(), 4); // bidirectional = 2 edges each
    }

    #[test]
    fn test_engine_perceive_and_balance() {
        let mut engine: Engine<4> = Engine::new();
        engine.add_room(1);
        engine.perceive(1, Embed([1.0, 0.5, 0.25, 0.125]));
        engine.perceive(1, Embed([0.9, 0.4, 0.2, 0.1]));
        assert!(engine.rooms[&1].is_balanced());
    }

    #[test]
    fn test_engine_gossip_propagation() {
        let mut engine: Engine<2> = Engine::new();
        engine.add_room(1);
        engine.add_room(2);
        engine.connect(1, 2, 1.0);

        // Room 1 perceives, room 2 starts empty
        engine.perceive(1, Embed([1.0, 0.0]));
        engine.perceive(1, Embed([0.0, 1.0]));
        assert_eq!(engine.rooms[&2].z_in.len(), 0);

        // After gossip, room 2 should have received murmurs
        engine.gossip_round();
        assert!(engine.rooms[&2].z_in.len() > 0, "Gossip should propagate info");
    }

    #[test]
    fn test_engine_full_cycle() {
        let mut engine: Engine<3> = Engine::new();
        engine.add_room(1);
        engine.add_room(2);
        engine.connect(1, 2, 0.9);

        engine.cycle(&[
            (1, Embed([1.0, 0.0, 0.0])),
            (2, Embed([0.0, 1.0, 0.0])),
        ]);
        assert!(engine.all_balanced());
        assert!(engine.total_surprise() > 0.0);
    }

    #[test]
    fn test_engine_total_surprise() {
        let mut engine: Engine<2> = Engine::new();
        engine.add_room(1);
        engine.add_room(2);
        engine.perceive(1, Embed([1.0, 0.0]));
        engine.perceive(2, Embed([0.0, 1.0]));
        assert!(engine.total_surprise() > 0.0);
    }
}
