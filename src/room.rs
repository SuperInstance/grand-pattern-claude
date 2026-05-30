//! Room — a single cell in the Grand Pattern graph.
//!
//! Each room owns its own perception DB (Z_in), prediction DB (Z_out),
//! JEPA predictive coder, double-entry ledger, and vibe tracker.

use crate::*;

/// A room in the cellular graph. Self-contained cognitive unit.
pub struct Room<const D: usize> {
    pub id: RoomId,
    pub z_in: VectorDb<D>,      // perception database
    pub z_out: VectorDb<D>,     // prediction database
    pub jepa: Jepa<D>,          // predictive coding map Z_in → Z_out
    pub ledger: Ledger,         // double-entry bookkeeping
    pub vibe: Vibe<D>,          // position/velocity/acceleration in embed space
    pub tick: u64,
    pub next_id: u64,           // auto-increment for vector entries
}

impl<const D: usize> Room<D> {
    pub fn new(id: RoomId, jepa_lr: f64) -> Self {
        Room {
            id,
            z_in: VectorDb::new(),
            z_out: VectorDb::new(),
            jepa: Jepa::new(jepa_lr),
            ledger: Ledger::new(),
            vibe: Vibe::new(Embed::zeros()),
            tick: 0,
            next_id: 0,
        }
    }

    /// Perceive a new embedding: store in Z_in, run JEPA to predict Z_out,
    /// record in ledger, update vibe.
    pub fn perceive(&mut self, v: Embed<D>) -> u64 {
        let eid = self.next_id;
        self.next_id += 1;
        self.tick += 1;

        // Store perception
        let perception_mag = v.l2_norm();
        self.z_in.insert(eid, v.clone());
        self.ledger.perceive(self.tick, perception_mag, &format!("perceive_{}", eid));

        // Predict via JEPA
        let predicted = self.jepa.predict(&v);
        let prediction_mag = predicted.l2_norm();
        self.z_out.insert(eid, predicted.clone());
        self.ledger.predict(self.tick, prediction_mag, &format!("predict_{}", eid));

        // Compute surprise and record it
        // Accounting identity: perceptions = predictions + surprise
        // → surprise = perceptions − predictions (the exact imbalance)
        let surprise = perception_mag - prediction_mag;
        self.ledger.record_surprise(self.tick, surprise, &format!("surprise_{}", eid));

        // Update vibe with centroid of Z_in
        self.vibe.push(self.z_in.centroid());

        eid
    }

    /// Teach the JEPA: given input and desired output, update weights.
    /// Returns prediction error.
    pub fn teach(&mut self, input: &Embed<D>, target: &Embed<D>) -> f64 {
        self.jepa.learn(input, target)
    }

    /// Total surprise accumulated (sum of all surprise entries).
    pub fn total_surprise(&self) -> f64 {
        self.ledger.entries.iter()
            .filter(|e| e.kind == EntryKind::Surprise)
            .map(|e| e.amount)
            .sum()
    }

    /// Check if ledger is balanced.
    pub fn is_balanced(&self) -> bool {
        self.ledger.is_balanced()
    }

    /// Current speed through embedding space.
    pub fn speed(&self) -> f64 {
        self.vibe.speed()
    }

    /// Compute a gossip murmur to send to neighbors.
    pub fn murmur(&self) -> Murmur<D> {
        Murmur {
            source: self.id,
            centroid: self.z_in.centroid(),
            surprise: self.total_surprise(),
            tick: self.tick,
        }
    }

    /// Receive a murmur from another room (integrates into Z_in as a soft perception).
    pub fn hear(&mut self, murmur: &Murmur<D>) {
        let eid = self.next_id;
        self.next_id += 1;
        self.tick += 1;
        // Integrate the neighbor's centroid as a perception (diminished by surprise)
        let weight = if murmur.surprise > 0.0 { 1.0 / (1.0 + murmur.surprise) } else { 1.0 };
        let soft = murmur.centroid.scale(weight);
        self.z_in.insert(eid, soft);
    }

    /// 3-phase garbage collection:
    ///   Phase 1 (Merge) — consolidate nearby entries in Z_in.
    ///   Phase 2 (Decay) — remove entries far from centroid.
    ///   Phase 3 (Prune) — cap DB size to `max_entries`.
    pub fn gc(&mut self, merge_threshold: f64, decay_sigma: f64, max_entries: usize) {
        // Phase 1: Merge — replace pairs closer than threshold with their mean.
        if self.z_in.len() >= 2 {
            let mut merged: Vec<(u64, Embed<D>)> = Vec::new();
            let entries = self.z_in.entries().to_vec();
            let mut consumed = vec![false; entries.len()];

            for i in 0..entries.len() {
                if consumed[i] { continue; }
                let mut acc = entries[i].1.clone();
                let mut count = 1u64;
                for j in (i+1)..entries.len() {
                    if consumed[j] { continue; }
                    if entries[i].1.distance(&entries[j].1) < merge_threshold {
                        acc = acc.add(&entries[j].1);
                        count += 1;
                        consumed[j] = true;
                    }
                }
                consumed[i] = true;
                let mean = acc.scale(1.0 / count as f64);
                merged.push((entries[i].0, mean));
            }
            self.z_in = VectorDb::new();
            for (id, v) in merged {
                self.z_in.insert(id, v);
            }
        }

        // Phase 2: Decay — remove entries far from centroid.
        let centroid = self.z_in.centroid();
        self.z_in.retain(|(_, v)| v.distance(&centroid) < decay_sigma);

        // Phase 3: Prune — keep only the most recent max_entries.
        if self.z_in.len() > max_entries {
            let entries = self.z_in.entries().to_vec();
            let keep = entries.into_iter().rev().take(max_entries).collect::<Vec<_>>();
            self.z_in = VectorDb::new();
            for (id, v) in keep.into_iter().rev() {
                self.z_in.insert(id, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_room_perceive_creates_balanced_ledger() {
        let mut room: Room<4> = Room::new(1, 0.1);
        room.perceive(Embed([1.0, 0.5, 0.25, 0.125]));
        assert!(room.is_balanced());
        assert_eq!(room.z_in.len(), 1);
        assert_eq!(room.z_out.len(), 1);
    }

    #[test]
    fn test_room_multiple_perceptions() {
        let mut room: Room<4> = Room::new(1, 0.1);
        room.perceive(Embed([1.0, 0.0, 0.0, 0.0]));
        room.perceive(Embed([0.0, 1.0, 0.0, 0.0]));
        room.perceive(Embed([0.0, 0.0, 1.0, 0.0]));
        assert_eq!(room.z_in.len(), 3);
        assert_eq!(room.z_out.len(), 3);
        assert!(room.total_surprise() > 0.0);
    }

    #[test]
    fn test_room_teach_reduces_error() {
        let mut room: Room<4> = Room::new(1, 0.1);
        let input = Embed([1.0, 0.0, 0.0, 0.0]);
        let target = Embed([0.5, 0.5, 0.0, 0.0]);
        let err1 = room.teach(&input, &target);
        for _ in 0..200 {
            room.teach(&input, &target);
        }
        let err_final = room.teach(&input, &target);
        assert!(err_final < err1, "Teaching should reduce error");
    }

    #[test]
    fn test_room_vibe_updates() {
        let mut room: Room<2> = Room::new(1, 0.1);
        room.perceive(Embed([1.0, 0.0]));
        room.perceive(Embed([2.0, 0.0]));
        assert!(room.speed() > 0.0);
    }

    #[test]
    fn test_room_gc_merges_nearby() {
        let mut room: Room<2> = Room::new(1, 0.1);
        room.perceive(Embed([1.0, 0.0]));
        room.perceive(Embed([1.01, 0.0])); // very close
        room.perceive(Embed([5.0, 5.0])); // far away
        assert_eq!(room.z_in.len(), 3);
        room.gc(0.1, 100.0, 100); // merge threshold 0.1
        assert!(room.z_in.len() < 3, "GC should merge nearby entries");
    }

    #[test]
    fn test_room_gc_prune() {
        let mut room: Room<2> = Room::new(1, 0.1);
        for i in 0..10 {
            room.perceive(Embed([i as f64, 0.0]));
        }
        assert_eq!(room.z_in.len(), 10);
        room.gc(0.01, 100.0, 3); // max 3 entries
        assert!(room.z_in.len() <= 3, "GC should prune to max_entries");
    }

    #[test]
    fn test_room_gossip_round_trip() {
        let mut room_a: Room<2> = Room::new(1, 0.1);
        let mut room_b: Room<2> = Room::new(2, 0.1);
        room_a.perceive(Embed([1.0, 0.0]));
        room_a.perceive(Embed([0.0, 1.0]));
        let murmur = room_a.murmur();
        assert_eq!(murmur.source, 1);
        room_b.hear(&murmur);
        assert_eq!(room_b.z_in.len(), 1);
    }
}
