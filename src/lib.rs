//! Grand Pattern — Core types, math primitives, and graph structure.
//!
//! Architecture: "Lensed Monolith" — all fundamental types live in one coherent
//! namespace, rooms compose them. No file-per-concept fragmentation.

#![allow(unused)]

pub mod room;
pub mod engine;

use std::collections::HashMap;

// ─── Const-generic embedding dimension ────────────────────────────────

/// A fixed-size vector using const generics. No heap allocation for the hot path.
#[derive(Clone, Debug, PartialEq)]
pub struct Embed<const D: usize>(pub [f64; D]);

impl<const D: usize> Embed<D> {
    pub const ZERO: Embed<D> = Embed([0.0; D]);

    #[inline]
    pub fn zeros() -> Self {
        Embed([0.0; D])
    }

    #[inline]
    pub fn from_fn(f: impl Fn(usize) -> f64) -> Self {
        let mut arr = [0.0; D];
        let mut i = 0;
        while i < D {
            arr[i] = f(i);
            i += 1;
        }
        Embed(arr)
    }

    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        let mut s = 0.0;
        for i in 0..D {
            s += self.0[i] * other.0[i];
        }
        s
    }

    #[inline]
    pub fn l2_norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn cosine(&self, other: &Self) -> f64 {
        let d = self.dot(other);
        let n = self.l2_norm() * other.l2_norm();
        if n == 0.0 { 0.0 } else { d / n }
    }

    /// Euclidean distance.
    #[inline]
    pub fn distance(&self, other: &Self) -> f64 {
        let mut s = 0.0;
        for i in 0..D {
            let d = self.0[i] - other.0[i];
            s += d * d;
        }
        s.sqrt()
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut r = [0.0; D];
        for i in 0..D {
            r[i] = self.0[i] + other.0[i];
        }
        Embed(r)
    }

    pub fn scale(&self, s: f64) -> Self {
        let mut r = [0.0; D];
        for i in 0..D {
            r[i] = self.0[i] * s;
        }
        Embed(r)
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut r = [0.0; D];
        for i in 0..D {
            r[i] = self.0[i] - other.0[i];
        }
        Embed(r)
    }
}

// ─── Vector DB (mini) ─────────────────────────────────────────────────

/// A lightweight vector database: entries tagged with an ID, linear scan
/// (good enough for cell-local N ≤ 1024).
#[derive(Clone, Debug)]
pub struct VectorDb<const D: usize> {
    entries: Vec<(u64, Embed<D>)>,
}

impl<const D: usize> VectorDb<D> {
    pub fn new() -> Self {
        VectorDb { entries: Vec::new() }
    }

    pub fn insert(&mut self, id: u64, v: Embed<D>) {
        self.entries.push((id, v));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// K-nearest by cosine similarity.
    pub fn knn(&self, query: &Embed<D>, k: usize) -> Vec<(u64, f64)> {
        let mut scored: Vec<_> = self.entries.iter()
            .map(|(id, v)| (*id, query.cosine(v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Sum of all vectors (for centroid).
    pub fn centroid(&self) -> Embed<D> {
        if self.entries.is_empty() {
            return Embed::zeros();
        }
        let mut sum = Embed::zeros();
        for (_, v) in &self.entries {
            sum = sum.add(v);
        }
        sum.scale(1.0 / self.entries.len() as f64)
    }

    pub fn entries(&self) -> &[(u64, Embed<D>)] {
        &self.entries
    }

    /// Retain only entries passing predicate.
    pub fn retain(&mut self, f: impl Fn(&(u64, Embed<D>)) -> bool) {
        self.entries.retain(f);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl<const D: usize> Default for VectorDb<D> {
    fn default() -> Self { Self::new() }
}

// ─── Ledger (double-entry bookkeeping) ────────────────────────────────

/// Every perception is a debit, every prediction a credit.
/// Surprise = perceptions − predictions (the imbalance).
#[derive(Clone, Debug)]
pub struct Ledger {
    pub debits: f64,   // perceptions (assets in)
    pub credits: f64,  // predictions (liabilities out)
    pub entries: Vec<LedgerEntry>,
}

#[derive(Clone, Debug)]
pub struct LedgerEntry {
    pub tick: u64,
    pub kind: EntryKind,
    pub amount: f64,
    pub description: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EntryKind {
    Perception,
    Prediction,
    Surprise,
}

impl Ledger {
    pub fn new() -> Self {
        Ledger {
            debits: 0.0,
            credits: 0.0,
            entries: Vec::new(),
        }
    }

    /// Record a perception (debit side).
    pub fn perceive(&mut self, tick: u64, amount: f64, desc: &str) {
        self.debits += amount;
        self.entries.push(LedgerEntry {
            tick,
            kind: EntryKind::Perception,
            amount,
            description: desc.to_string(),
        });
    }

    /// Record a prediction (credit side).
    pub fn predict(&mut self, tick: u64, amount: f64, desc: &str) {
        self.credits += amount;
        self.entries.push(LedgerEntry {
            tick,
            kind: EntryKind::Prediction,
            amount,
            description: desc.to_string(),
        });
    }

    /// Record surprise (the balancing entry).
    pub fn record_surprise(&mut self, tick: u64, amount: f64, desc: &str) {
        self.entries.push(LedgerEntry {
            tick,
            kind: EntryKind::Surprise,
            amount,
            description: desc.to_string(),
        });
    }

    /// Balance: surprise = perceptions − predictions. Must be zero when balanced.
    pub fn balance(&self) -> f64 {
        self.debits - self.credits
    }

    /// Double-entry invariant: debits == credits + recorded surprise total.
    pub fn is_balanced(&self) -> bool {
        let surprise_total: f64 = self.entries.iter()
            .filter(|e| e.kind == EntryKind::Surprise)
            .map(|e| e.amount)
            .sum();
        (self.debits - self.credits - surprise_total).abs() < 1e-10
    }
}

impl Default for Ledger {
    fn default() -> Self { Self::new() }
}

// ─── JEPA — Predictive Coding variant ─────────────────────────────────

/// Joint-Embedding Predictive Architecture using *predictive coding* rather
/// than Hebbian learning.  A simple linear map W: Z_in → Z_out trained by
/// residual error minimisation (predictive coding's core principle).
#[derive(Clone, Debug)]
pub struct Jepa<const D: usize> {
    /// Weight matrix stored as D rows of D elements (row-major).
    weights: Vec<[f64; D]>,
    lr: f64,
}

impl<const D: usize> Jepa<D> {
    pub fn new(lr: f64) -> Self {
        // Initialise with small random-ish values using a simple LCG.
        let mut weights = Vec::with_capacity(D);
        let mut seed: u64 = 0xC0FFEE;
        for _ in 0..D {
            let mut row = [0.0; D];
            for j in 0..D {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                row[j] = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
                row[j] *= 0.01; // small init
            }
            weights.push(row);
        }
        Jepa { weights, lr }
    }

    /// Forward: predict Z_out from Z_in.
    pub fn predict(&self, z_in: &Embed<D>) -> Embed<D> {
        let mut out = [0.0; D];
        for i in 0..D {
            let mut s = 0.0;
            for j in 0..D {
                s += self.weights[i][j] * z_in.0[j];
            }
            out[i] = s;
        }
        Embed(out)
    }

    /// Predictive coding update: minimise ‖target − W·input‖².
    /// ΔW = lr × error × inputᵀ  (outer product learning rule).
    /// Returns the prediction error magnitude.
    pub fn learn(&mut self, z_in: &Embed<D>, z_target: &Embed<D>) -> f64 {
        let predicted = self.predict(z_in);
        let mut err_sq = 0.0;
        for i in 0..D {
            let err = z_target.0[i] - predicted.0[i];
            err_sq += err * err;
            for j in 0..D {
                self.weights[i][j] += self.lr * err * z_in.0[j];
            }
        }
        err_sq.sqrt()
    }
}

// ─── Vibe (position, velocity, acceleration) ──────────────────────────

/// The "vibe" of a room: its trajectory through embedding space.
#[derive(Clone, Debug)]
pub struct Vibe<const D: usize> {
    pub position: Embed<D>,
    pub velocity: Embed<D>,
    pub acceleration: Embed<D>,
    pub history_len: usize,
}

impl<const D: usize> Vibe<D> {
    pub fn new(initial: Embed<D>) -> Self {
        Vibe {
            position: initial,
            velocity: Embed::zeros(),
            acceleration: Embed::zeros(),
            history_len: 0,
        }
    }

    /// Push a new position; updates velocity and acceleration via finite differences.
    pub fn push(&mut self, new_pos: Embed<D>) {
        let new_vel = new_pos.sub(&self.position);
        let new_acc = new_vel.sub(&self.velocity);
        self.acceleration = new_acc;
        self.velocity = new_vel;
        self.position = new_pos;
        self.history_len += 1;
    }

    /// Momentum magnitude (speed through embedding space).
    pub fn speed(&self) -> f64 {
        self.velocity.l2_norm()
    }

    /// Jerk (rate of change of acceleration).
    pub fn jerk(&self) -> f64 {
        self.acceleration.l2_norm()
    }
}

// ─── Graph / Room IDs ─────────────────────────────────────────────────

pub type RoomId = u64;

/// Edge between rooms with a weight (gossip strength).
#[derive(Clone, Debug)]
pub struct Edge {
    pub from: RoomId,
    pub to: RoomId,
    pub weight: f64,
}

/// The cellular graph: adjacency list + room metadata.
#[derive(Clone, Debug)]
pub struct Graph {
    pub rooms: Vec<RoomId>,
    pub edges: Vec<Edge>,
    pub adjacency: HashMap<RoomId, Vec<usize>>, // room → edge indices
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            rooms: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    pub fn add_room(&mut self, id: RoomId) {
        if !self.rooms.contains(&id) {
            self.rooms.push(id);
            self.adjacency.insert(id, Vec::new());
        }
    }

    pub fn add_edge(&mut self, from: RoomId, to: RoomId, weight: f64) {
        let idx = self.edges.len();
        self.edges.push(Edge { from, to, weight });
        self.adjacency.entry(from).or_default().push(idx);
    }

    pub fn neighbors(&self, id: RoomId) -> Vec<(RoomId, f64)> {
        match self.adjacency.get(&id) {
            Some(indices) => indices.iter()
                .map(|&i| (self.edges[i].to, self.edges[i].weight))
                .collect(),
            None => Vec::new(),
        }
    }

    pub fn room_count(&self) -> usize {
        self.rooms.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for Graph {
    fn default() -> Self { Self::new() }
}

// ─── Murmur (gossip message) ──────────────────────────────────────────

/// A gossip message passed between rooms.
#[derive(Clone, Debug)]
pub struct Murmur<const D: usize> {
    pub source: RoomId,
    pub centroid: Embed<D>,
    pub surprise: f64,
    pub tick: u64,
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_dot_product() {
        let a: Embed<3> = Embed([1.0, 2.0, 3.0]);
        let b: Embed<3> = Embed([4.0, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_embed_l2_norm() {
        let v: Embed<3> = Embed([3.0, 4.0, 0.0]);
        assert!((v.l2_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_embed_cosine_similarity() {
        let a: Embed<3> = Embed([1.0, 0.0, 0.0]);
        let b: Embed<3> = Embed([0.0, 1.0, 0.0]);
        assert!(a.cosine(&b).abs() < 1e-10);
        assert!((a.cosine(&a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_embed_distance() {
        let a: Embed<2> = Embed([0.0, 0.0]);
        let b: Embed<2> = Embed([3.0, 4.0]);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_db_insert_and_knn() {
        let mut db: VectorDb<3> = VectorDb::new();
        db.insert(1, Embed([1.0, 0.0, 0.0]));
        db.insert(2, Embed([0.0, 1.0, 0.0]));
        db.insert(3, Embed([0.9, 0.1, 0.0]));
        let results = db.knn(&Embed([1.0, 0.0, 0.0]), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // exact match first
        assert_eq!(results[1].0, 3); // near match second
    }

    #[test]
    fn test_vector_db_centroid() {
        let mut db: VectorDb<2> = VectorDb::new();
        db.insert(1, Embed([0.0, 0.0]));
        db.insert(2, Embed([2.0, 2.0]));
        let c = db.centroid();
        assert!((c.0[0] - 1.0).abs() < 1e-10);
        assert!((c.0[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_db_retain() {
        let mut db: VectorDb<2> = VectorDb::new();
        db.insert(1, Embed([1.0, 0.0]));
        db.insert(2, Embed([0.0, 1.0]));
        db.retain(|(id, _)| *id == 1);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_ledger_double_entry_balance() {
        let mut ledger = Ledger::new();
        ledger.perceive(1, 100.0, "sensory input");
        ledger.predict(1, 80.0, "expected");
        ledger.record_surprise(1, 20.0, "prediction error");
        assert!(ledger.is_balanced());
        assert!((ledger.balance() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_ledger_unbalanced() {
        let mut ledger = Ledger::new();
        ledger.perceive(1, 100.0, "input");
        ledger.predict(1, 80.0, "expected");
        // No surprise recorded → unbalanced
        assert!(!ledger.is_balanced());
    }

    #[test]
    fn test_jepa_predict_and_learn() {
        let mut jepa: Jepa<4> = Jepa::new(0.1);
        let input = Embed([1.0, 0.5, 0.25, 0.125]);
        let target = Embed([0.5, 0.25, 0.125, 0.0625]);
        let err1 = jepa.learn(&input, &target);
        // After many iterations, error should decrease
        let mut err_last = err1;
        for _ in 0..100 {
            err_last = jepa.learn(&input, &target);
        }
        assert!(err_last < err1, "JEPA should converge: {} < {}", err_last, err1);
    }

    #[test]
    fn test_vibe_kinematics() {
        let mut vibe: Vibe<2> = Vibe::new(Embed([0.0, 0.0]));
        vibe.push(Embed([1.0, 0.0]));
        assert!((vibe.speed() - 1.0).abs() < 1e-10);
        vibe.push(Embed([3.0, 0.0]));
        assert!((vibe.speed() - 2.0).abs() < 1e-10);
        assert!((vibe.acceleration.0[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_graph_construction() {
        let mut g = Graph::new();
        g.add_room(1);
        g.add_room(2);
        g.add_room(3);
        g.add_edge(1, 2, 0.8);
        g.add_edge(2, 3, 0.5);
        g.add_edge(1, 3, 0.3);
        assert_eq!(g.room_count(), 3);
        assert_eq!(g.edge_count(), 3);
        let n1 = g.neighbors(1);
        assert_eq!(n1.len(), 2);
    }

    #[test]
    fn test_graph_duplicate_room() {
        let mut g = Graph::new();
        g.add_room(42);
        g.add_room(42);
        assert_eq!(g.room_count(), 1);
    }

    #[test]
    fn test_embed_add_sub_scale() {
        let a: Embed<2> = Embed([1.0, 2.0]);
        let b: Embed<2> = Embed([3.0, 4.0]);
        let sum = a.add(&b);
        assert_eq!(sum.0, [4.0, 6.0]);
        let diff = b.sub(&a);
        assert_eq!(diff.0, [2.0, 2.0]);
        let scaled = a.scale(3.0);
        assert_eq!(scaled.0, [3.0, 6.0]);
    }
}
