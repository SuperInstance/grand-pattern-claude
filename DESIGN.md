# DESIGN.md — Grand Pattern (Claude Entry)

## Architectural Philosophy: The Lensed Monolith

Where Kimi's entry decomposes the system into **11 files** (one per concept), this entry takes the opposite approach: **4 source files**, each a *lens* that refracts the same core ideas from a different angle.

### File Structure

| File | Purpose | Lines of Concern |
|------|---------|------------------|
| `lib.rs` | Core types: Embed, VectorDb, Ledger, JEPA, Vibe, Graph, Murmur | ~300 |
| `room.rs` | Room cell composing all subsystems | ~180 |
| `engine.rs` | Orchestration: gossip rounds, GC cycles, full cycles | ~120 |
| `main.rs` | Entry point + demo | ~50 |

**Total: 4 files** vs Kimi's 11.

### Why Fewer Files?

1. **Cognitive coherence.** When you're reading the Grand Pattern, you're reading *one idea* expressed through layers. Splitting every struct into its own file fragments what should be a unified mental model. `lib.rs` is the entire ontology — read it top-to-bottom and you understand the system.

2. **Compile-time locality.** Rust's module system is great for *visibility control*, but when all types are tightly coupled (Embed feeds VectorDb feeds JEPA feeds Room), artificial barriers create import gymnastics without real encapsulation benefit. Our `lib.rs` is a single namespace with clear section headers.

3. **Const generics as architecture.** The `const D: usize` parameter threads through every type, giving us dimension-generic code without generics sprawl. One `Embed<D>` definition serves all dimensions. Kimi's approach would need a separate file just for the generic plumbing.

## Key Design Differences from Kimi

### 1. Predictive Coding vs Hebbian Learning

Kimi uses **Hebbian JEPA** ("fire together, wire together"). We use **Predictive Coding JEPA** — learning by *minimising prediction error*:

```
ΔW = lr × (target − W·input) × inputᵀ
```

This is fundamentally different:
- **Hebbian** strengthens correlations (what *co-occurs*)
- **Predictive coding** minimises residuals (what *surprises*)

Predictive coding is closer to how the brain actually works (Friston's Free Energy Principle). Surprise becomes a first-class quantity, not a side effect. The ledger's double-entry bookkeeping *is* the accounting of surprise.

### 2. Linear Scan Vector DB vs Arena/Ring Buffer

Kimi uses arena allocation and ring buffers for their VectorDb. We use a **simple `Vec<(u64, Embed<D>)>`** with linear scan KNN.

Why? Because each room's DB is cell-local (N ≤ 64 after GC). For N < 100, a linear scan with cache-friendly memory layout beats any fancy data structure. The GC's prune phase guarantees the bound. We chose *predictable simplicity* over *asymptotic complexity that never matters*.

### 3. Gossip as Murmur Struct, Not Protocol

Kimi's gossip is a separate module with its own lifecycle. Our gossip is a `Murmur<D>` struct — a simple data packet that rooms produce and consume. No protocol state machine, no channels. The `Engine` collects all murmurs in one pass, then delivers them in a second pass. Two-phase to satisfy the borrow checker, but also two-phase conceptually: *speak, then listen*.

### 4. 3-Phase GC as Room Method, Not Separate Module

Kimi separates GC into its own file. Our GC is a method on `Room` — because GC operates on room-local data (Z_in). The three phases are:

1. **Merge**: Replace nearby pairs (distance < threshold) with their mean. Reduces redundancy.
2. **Decay**: Remove entries far from centroid (> σ). The centroid IS the room's "self."
3. **Prune**: Hard cap on entry count. Keeps memory bounded.

All three phases are pure functions of room state. No external coordinator needed.

### 5. Vibe as Finite Differences, Not Accumulators

Kimi accumulates velocity/acceleration. We compute them via **finite differences**:

```
velocity[t] = position[t] − position[t−1]
acceleration[t] = velocity[t] − velocity[t−1]
```

This is cleaner, stateless beyond the last two positions, and directly maps to physical intuition. No accumulation drift.

## Test Coverage

**26 tests** across three test modules:

- `lib.rs`: 14 tests (Embed math, VectorDb CRUD, Ledger double-entry, JEPA convergence, Vibe kinematics, Graph construction)
- `room.rs`: 7 tests (perceive balance, multi-perception, teach convergence, vibe updates, GC merge, GC prune, gossip round-trip)
- `engine.rs`: 5 tests (graph wiring, perceive+balance, gossip propagation, full cycle, total surprise)

Every invariant is tested: double-entry balance, JEPA convergence, GC reduction, gossip propagation.

## Dependencies

**Zero.** Pure Rust, no `Cargo.toml` dependencies. Not even `rand` — we use an LCG for JEPA weight initialization.

## The Lensed Monolith Thesis

Complexity should live in *concepts*, not in *files*. The Grand Pattern is one concept: a cellular graph where rooms perceive, predict, and gossip. Splitting it into 11 files doesn't make it simpler — it makes you open 11 files to understand one idea.

Our 4 files are lenses:
- `lib.rs` = *What is it?* (ontology)
- `room.rs` = *How does a cell work?* (mechanism)
- `engine.rs` = *How do cells interact?* (dynamics)
- `main.rs` = *What does it look like?* (demonstration)

Each lens reveals a different aspect of the same whole. That's the Lensed Monolith.
