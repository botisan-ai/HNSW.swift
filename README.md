# HNSW for iOS/macOS

A Swift wrapper for [hnsw_rs](https://crates.io/crates/hnsw_rs), a Hierarchical Navigable Small World graph implementation for approximate nearest neighbor search, written in Rust. Uses [UniFFI](https://github.com/mozilla/uniffi-rs) to generate Swift bindings.

## Features

- Fast approximate nearest neighbor (ANN) search using HNSW algorithm
- 4 distance metrics: L2 (Euclidean), Cosine, Dot Product, L1 (Manhattan)
- Create new indices or load existing ones from disk
- Thread-safe with Swift `actor` isolation
- Insert vectors individually or in batches
- Tombstone deletes with optional compaction
- Proper memory management with no leaks

## Installation

### Swift Package Manager

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/example/HNSW.swift.git", from: "0.1.0")
]
```

Or add it via Xcode: File → Add Package Dependencies → Enter the repository URL.

## Usage

### Creating an Index

```swift
import HnswSwift

// Create a new index
// Parameters: dimension, maxElements, maxLayers, maxConnections, efConstruction, distanceType
let index = HnswIndex(
    maxConnections: 16,    // Connections per node (M)
    maxElements: 10000,    // Maximum number of vectors
    maxLayers: 16,         // Max layers
    efConstruction: 200,   // Construction time/accuracy tradeoff (higher = more accurate)
    dimension: 128,        // Vector dimension
    distanceType: .cosine  // Distance metric
)
```

### Distance Metrics

```swift
// Available distance metrics
let l2Index = HnswIndex(maxConnections: 16, maxElements: 1000, maxLayers: 16, efConstruction: 200, dimension: 128, distanceType: .l2)       // Euclidean distance
let cosineIndex = HnswIndex(maxConnections: 16, maxElements: 1000, maxLayers: 16, efConstruction: 200, dimension: 128, distanceType: .cosine) // Cosine similarity
let dotIndex = HnswIndex(maxConnections: 16, maxElements: 1000, maxLayers: 16, efConstruction: 200, dimension: 128, distanceType: .dot)     // Dot product
let l1Index = HnswIndex(maxConnections: 16, maxElements: 1000, maxLayers: 16, efConstruction: 200, dimension: 128, distanceType: .l1)       // Manhattan distance
```

### Inserting Vectors

```swift
// Insert a single vector
let vector: [Float] = Array(repeating: 0.1, count: 128)
try await index.insert(vector: vector, id: 1)

// Insert multiple vectors in batch (more efficient)
let vectors: [[Float]] = (0..<100).map { _ in
    (0..<128).map { _ in Float.random(in: -1...1) }
}
let ids: [UInt64] = Array(0..<100)
try await index.insertBatch(vectors: vectors, ids: ids)
```

### Searching for Nearest Neighbors

```swift
// Search for k nearest neighbors
let queryVector: [Float] = Array(repeating: 0.5, count: 128)
let results = try await index.search(query: queryVector, k: 10, efSearch: 50)

// Results contain (id, distance) pairs
for result in results {
    print("ID: \(result.id), Distance: \(result.distance)")
}
```

### Deletion and Compaction

```swift
// Mark vectors as deleted (tombstones)
await index.delete(id: 42)
await index.delete(ids: [7, 9, 11])

// Rebuild the index without deleted IDs
// Uses the config stored at init time. If you loaded from disk, pass a config here.
try await index.compact()

// Or provide a config explicitly
let config = HnswIndexConfig(dimension: 128, distanceType: .cosine)
try await index.compact(config: config)
```

### Persistence

```swift
// Save index to disk
try await index.save(directory: "/path/to/index", basename: "my_index")

// Load index from disk
let loadedIndex = try HnswIndex.load(
    directory: "/path/to/index",
    basename: "my_index",
    dimension: 128,
    distanceType: .cosine
)

// Continue using the loaded index
let results = try await loadedIndex.search(query: queryVector, k: 5, efSearch: 50)
```

### Complete Example

```swift
import HnswSwift

// Create an index for 128-dimensional embeddings
let index = HnswIndex(
    maxConnections: 16,
    maxElements: 10000,
    maxLayers: 16,
    efConstruction: 200,
    dimension: 128,
    distanceType: .cosine
)

// Generate some sample vectors (in practice, these would be embeddings)
let vectors: [[Float]] = (0..<1000).map { _ in
    var v = (0..<128).map { _ in Float.random(in: -1...1) }
    // Normalize for cosine similarity
    let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
    return v.map { $0 / norm }
}

// Insert vectors
let ids = Array(UInt64(0)..<1000)
try await index.insertBatch(vectors: vectors, ids: ids)

// Search for similar vectors
let query = vectors[42]  // Find vectors similar to vector #42
let results = try await index.search(query: query, k: 5, efSearch: 50)

print("Nearest neighbors to vector #42:")
for result in results {
    print("  ID: \(result.id), Distance: \(result.distance)")
}

// Save for later use
try await index.save(directory: "./embeddings", basename: "my_embeddings")
```

## API Reference

### HnswIndex

| Method | Description |
|--------|-------------|
| `init(maxConnections:maxElements:maxLayers:efConstruction:dimension:distanceType:)` | Create a new empty index |
| `static load(directory:basename:dimension:distanceType:config:)` | Load an existing index from disk |
| `insert(vector:id:)` | Insert a single vector |
| `insertBatch(vectors:ids:)` | Insert multiple vectors (more efficient) |
| `delete(id:)` | Mark a vector as deleted |
| `delete(ids:)` | Mark multiple vectors as deleted |
| `compact(config:)` | Rebuild index without deleted IDs (in-place) |
| `search(query:k:efSearch:)` | Find k nearest neighbors |
| `save(directory:basename:)` | Save index to disk |
| `count()` | Count non-deleted vectors |
| `isEmpty()` | Check if index is empty (excluding tombstones) |
| `getDimension()` | Get vector dimension |
| `setSearchingMode(enabled:)` | Toggle searching mode |

### HnswDistanceType

| Metric | Description |
|--------|-------------|
| `.l2` | Euclidean (L2) distance |
| `.cosine` | Cosine distance (1 - cosine similarity) |
| `.dot` | Dot product distance |
| `.l1` | Manhattan (L1) distance |

### HnswSearchResult

| Property | Type | Description |
|----------|------|-------------|
| `id` | `UInt64` | The ID of the found vector |
| `distance` | `Float` | Distance from query vector |

## HNSW Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `dimension` | Vector dimensionality | Depends on your embeddings (e.g., 128, 384, 768, 1536) |
| `maxElements` | Maximum vectors the index can hold | Depends on your dataset |
| `maxConnections` | Connections per node (M) | 12-48 (higher = better recall, more memory) |
| `maxLayers` | Maximum number of layers | 8-32 (rarely adjusted) |
| `efConstruction` | Build-time quality parameter | 100-400 (higher = better quality, slower build) |
| `efSearch` | Search-time quality parameter | 50-500 (higher = better recall, slower search) |

## Development

### Prerequisites

- Rust toolchain with iOS targets
- Xcode with Swift 6.0+

### Building

```bash
# Install Rust targets (if not already done via rust-toolchain.toml)
rustup target add aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin

# Build everything (Rust lib + Swift bindings + XCFramework)
./build-ios.sh

# Run tests
swift test
```

### Project Structure

```
HNSW.swift/
├── src/
│   ├── lib.rs              # Rust FFI implementation
│   └── uniffi-bindgen.rs   # UniFFI code generator
├── Sources/
│   ├── HnswFFI/            # Auto-generated Swift bindings
│   └── HnswSwift/          # Hand-written Swift wrapper
├── Tests/
│   └── HnswSwiftTests/
├── build-ios.sh            # Build script
├── Cargo.toml              # Rust dependencies
└── Package.swift           # Swift package manifest
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [hnsw_rs](https://crates.io/crates/hnsw_rs) - The underlying Rust HNSW implementation
- [UniFFI](https://github.com/mozilla/uniffi-rs) - Mozilla's FFI bindings generator
- [HNSW paper](https://arxiv.org/abs/1603.09320) - "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Malkov & Yashunin
