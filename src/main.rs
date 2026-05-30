//! Grand Pattern — main entry point and demo.

use grand_pattern::*;
use grand_pattern::engine::Engine;

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║        Grand Pattern — Claude Entry          ║");
    println!("║     Lensed Monolith Architecture (4 files)   ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();

    const DIM: usize = 8;
    let mut engine: Engine<DIM> = Engine::new();

    // Build a 5-room ring graph.
    for id in 1..=5 {
        engine.add_room_with_lr(id, 0.05);
    }
    for id in 1..5 {
        engine.connect(id, id + 1, 0.8);
    }
    engine.connect(5, 1, 0.8); // close the ring

    println!("Graph: {} rooms, {} edges", engine.graph.room_count(), engine.graph.edge_count());

    // Feed perceptions into each room.
    for cycle in 0..10 {
        for id in 1..=5 {
            let v = Embed::from_fn(|i| {
                let phase = (cycle as f64 + id as f64 * 0.5) * 0.3;
                (phase + i as f64 * 0.1).sin()
            });
            engine.perceive(id, v);
        }
        engine.gossip_round();
        if cycle % 3 == 2 {
            engine.gc_all(0.2, 8.0, 32);
        }
    }

    println!();
    println!("After 10 cycles:");
    for (&id, room) in &engine.rooms {
        println!(
            "  Room {}: z_in={}, z_out={}, surprise={:.4}, speed={:.4}, balanced={}",
            id, room.z_in.len(), room.z_out.len(),
            room.total_surprise(), room.speed(), room.is_balanced()
        );
    }
    println!();
    println!("Total surprise: {:.4}", engine.total_surprise());
    println!("All balanced: {}", engine.all_balanced());
    println!();
    println!("✓ Grand Pattern running.");
}
