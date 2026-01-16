# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust-to-Swift wrapper that exposes the `hnsw_rs` crate (Hierarchical Navigable Small World graphs for approximate nearest neighbor search) to iOS/macOS applications using Mozilla's UniFFI for FFI bindings generation.

## Build Commands

```bash
# Full build (generates bindings + builds for all iOS targets + creates XCFramework)
./build-ios.sh

# Run Rust tests
cargo test

# Run Swift tests (requires build-ios.sh to have been run first)
swift test

# Manual bindgen (usually run via build-ios.sh)
cargo run --bin uniffi-bindgen generate --library ./target/debug/libhnsw.dylib --language swift --out-dir ./out
```

## Architecture: Rust + Swift via UniFFI

### Layer Structure

```
┌─────────────────────────────────────────┐
│  Swift App Layer                        │
│  (Uses HnswIndex actor with distance    │
│   type selection)                       │
├─────────────────────────────────────────┤
│  Sources/HnswSwift/HnswSwift.swift      │
│  (Swift-native API with actors)         │
├─────────────────────────────────────────┤
│  Sources/HnswFFI/hnsw.swift             │
│  (UniFFI-generated bindings)            │
├─────────────────────────────────────────┤
│  XCFramework (libhnsw-rs.xcframework)   │
│  (Compiled Rust static libraries)       │
├─────────────────────────────────────────┤
│  src/lib.rs                             │
│  (Rust implementation with UniFFI attrs)│
└─────────────────────────────────────────┘
```

### Key Files

- **src/lib.rs**: Rust implementation with `#[uniffi::export]` annotations wrapping the `hnsw_rs` crate
- **src/uniffi-bindgen.rs**: Binary that invokes UniFFI's Swift code generator
- **Sources/HnswFFI/hnsw.swift**: Auto-generated Swift bindings (do not edit manually)
- **Sources/HnswSwift/HnswSwift.swift**: Hand-written Swift wrapper providing idiomatic API with actors
- **build-ios.sh**: Build script that orchestrates the entire build pipeline

### Distance Types

Due to UniFFI limitations with generic types, separate index types are provided for each distance metric:
- `HnswIndexL2` - Euclidean (L2) distance
- `HnswIndexCosine` - Cosine distance
- `HnswIndexDot` - Dot product distance
- `HnswIndexL1` - Manhattan (L1) distance

The Swift wrapper (`HnswIndex`) provides a unified interface that internally dispatches to the appropriate type.

## API Reference

### Rust API (via UniFFI)

- `HnswIndexL2::new(max_nb_connection, max_elements, max_layer, ef_construction, dimension)` - Create new L2 index
- `HnswIndexL2::load(directory, basename, dimension)` - Load saved L2 index from disk
- `insert(data, id)` - Insert a single vector
- `insert_batch(data, ids)` - Insert multiple vectors in parallel
- `search(query, k, ef_search)` - Search for k nearest neighbors
- `save(directory, basename)` - Save index to disk (creates .hnsw.graph and .hnsw.data files)
- `len()` / `is_empty()` - Query index size
- `set_searching_mode(enabled)` - Enable after parallel insertions

### Swift API

```swift
// Create a new index
let index = HnswIndex(
    maxConnections: 16,
    maxElements: 10000,
    maxLayers: 16,
    efConstruction: 200,
    dimension: 128,
    distanceType: .l2
)

// Or load from disk
let index = try HnswIndex.load(
    directory: "/path/to/dir",
    basename: "my_index",
    dimension: 128,
    distanceType: .l2
)

// Insert vectors
try await index.insert(vector: [1.0, 2.0, ...], id: 0)
try await index.insertBatch(vectors: [[1.0, ...], [2.0, ...]], ids: [0, 1])

// Search
let results = try await index.search(query: [1.0, 2.0, ...], k: 10)
for result in results {
    print("ID: \(result.id), Distance: \(result.distance)")
}

// Save to disk
try await index.save(directory: "/path/to/dir", basename: "my_index")
```

## iOS Build Considerations

- **Targets**: Build for `aarch64-apple-ios` (devices), `aarch64-apple-ios-sim` (Apple Silicon simulators), `aarch64-apple-darwin` (macOS)
- **Modulemap naming**: UniFFI generates `hnswFFI.modulemap` but Swift packages require `module.modulemap`

## Implementation Notes

### Memory Management for Loaded Indices

When loading an index from disk, the `HnswIo` loader must remain alive for the lifetime of the `Hnsw` index (the index holds references to data owned by the loader). This is a self-referential struct pattern that Rust's borrow checker cannot express directly.

**Solution**: We use `ManuallyDrop` and raw pointers with a custom `Drop` implementation:

```rust
struct HnswInnerL2 {
    hnsw: ManuallyDrop<Hnsw<'static, f32, DistL2>>,
    io_ptr: Option<NonNull<HnswIo>>,  // None for new indices, Some for loaded
}

impl Drop for HnswInnerL2 {
    fn drop(&mut self) {
        unsafe {
            // First drop the Hnsw (which borrows from HnswIo)
            ManuallyDrop::drop(&mut self.hnsw);
            // Then reclaim the HnswIo memory
            if let Some(ptr) = self.io_ptr.take() {
                drop(Box::from_raw(ptr.as_ptr()));
            }
        }
    }
}
```

This ensures:
1. `Hnsw` is dropped first (releasing its borrows)
2. `HnswIo` is reclaimed after (no memory leak)
3. New indices (`io_ptr: None`) work normally
4. Loaded indices properly clean up both structs

**No memory leaks** - you can safely create and destroy multiple indices in your app.

## Testing

Swift tests are in `Tests/HnswSwiftTests/`. The test suite demonstrates:
- Basic index operations (insert, search)
- Batch insertion with parallel processing
- Different distance metrics (L2, Cosine)
- Save and load functionality with proper memory cleanup
