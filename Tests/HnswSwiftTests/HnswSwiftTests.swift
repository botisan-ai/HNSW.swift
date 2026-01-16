import Foundation
import Testing
@testable import HnswSwift

@Test func testBasicIndexOperations() async throws {
    let index = HnswIndex(
        maxConnections: 16,
        maxElements: 1000,
        maxLayers: 16,
        efConstruction: 200,
        dimension: 4,
        distanceType: .l2
    )
    
    #expect(try await index.isEmpty() == true)
    #expect(try await index.count() == 0)
    #expect(await index.getDimension() == 4)
    
    try await index.insert(vector: [1.0, 0.0, 0.0, 0.0], id: 0)
    try await index.insert(vector: [0.0, 1.0, 0.0, 0.0], id: 1)
    try await index.insert(vector: [0.0, 0.0, 1.0, 0.0], id: 2)
    
    #expect(try await index.isEmpty() == false)
    #expect(try await index.count() == 3)
    
    let results = try await index.search(query: [1.0, 0.0, 0.0, 0.0], k: 2)
    #expect(results.count == 2)
    #expect(results[0].id == 0)
    #expect(results[0].distance == 0.0)
}

@Test func testBatchInsert() async throws {
    let index = HnswIndex(
        maxConnections: 16,
        maxElements: 1000,
        maxLayers: 16,
        efConstruction: 200,
        dimension: 3,
        distanceType: .l2
    )
    
    let vectors: [[Float]] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]
    ]
    let ids: [UInt64] = [0, 1, 2, 3, 4]
    
    try await index.insertBatch(vectors: vectors, ids: ids)
    try await index.setSearchingMode(enabled: true)
    
    #expect(try await index.count() == 5)
    
    let results = try await index.search(query: [1.0, 0.0, 0.0], k: 3)
    #expect(results.count == 3)
    #expect(results[0].id == 0)
}

@Test func testCosineDistance() async throws {
    let index = HnswIndex(
        maxConnections: 16,
        maxElements: 1000,
        maxLayers: 16,
        efConstruction: 200,
        dimension: 3,
        distanceType: .cosine
    )
    
    try await index.insert(vector: [1.0, 0.0, 0.0], id: 0)
    try await index.insert(vector: [0.707, 0.707, 0.0], id: 1)
    try await index.insert(vector: [0.0, 1.0, 0.0], id: 2)
    
    let results = try await index.search(query: [1.0, 0.0, 0.0], k: 3)
    #expect(results.count == 3)
    #expect(results[0].id == 0)
}

@Test func testSaveAndLoad() async throws {
    let tempDir = FileManager.default.temporaryDirectory
    let testDir = tempDir.appendingPathComponent("hnsw_test_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: testDir, withIntermediateDirectories: true)
    
    defer {
        try? FileManager.default.removeItem(at: testDir)
    }
    
    let index = HnswIndex(
        maxConnections: 16,
        maxElements: 1000,
        maxLayers: 16,
        efConstruction: 200,
        dimension: 4,
        distanceType: .l2
    )
    
    try await index.insert(vector: [1.0, 0.0, 0.0, 0.0], id: 100)
    try await index.insert(vector: [0.0, 1.0, 0.0, 0.0], id: 200)
    try await index.insert(vector: [0.0, 0.0, 1.0, 0.0], id: 300)
    
    try await index.save(directory: testDir.path, basename: "test_index")
    
    let graphFile = testDir.appendingPathComponent("test_index.hnsw.graph")
    let dataFile = testDir.appendingPathComponent("test_index.hnsw.data")
    
    #expect(FileManager.default.fileExists(atPath: graphFile.path))
    #expect(FileManager.default.fileExists(atPath: dataFile.path))
    
    let loadedIndex = try HnswIndex.load(
        directory: testDir.path,
        basename: "test_index",
        dimension: 4,
        distanceType: .l2
    )
    
    #expect(try await loadedIndex.count() == 3)
    
    let results = try await loadedIndex.search(query: [1.0, 0.0, 0.0, 0.0], k: 1)
    #expect(results.count == 1)
    #expect(results[0].id == 100)
    #expect(results[0].distance == 0.0)
}

@Test func testLoadCosineIndex() async throws {
    let tempDir = FileManager.default.temporaryDirectory
    let testDir = tempDir.appendingPathComponent("hnsw_cosine_test_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: testDir, withIntermediateDirectories: true)
    
    defer {
        try? FileManager.default.removeItem(at: testDir)
    }
    
    let index = HnswIndex(
        maxConnections: 16,
        maxElements: 1000,
        maxLayers: 16,
        efConstruction: 200,
        dimension: 3,
        distanceType: .cosine
    )
    
    try await index.insert(vector: [1.0, 0.0, 0.0], id: 10)
    try await index.insert(vector: [0.0, 1.0, 0.0], id: 20)
    try await index.insert(vector: [0.0, 0.0, 1.0], id: 30)
    
    try await index.save(directory: testDir.path, basename: "cosine_index")
    
    let loadedIndex = try HnswIndex.load(
        directory: testDir.path,
        basename: "cosine_index",
        dimension: 3,
        distanceType: .cosine
    )
    
    #expect(try await loadedIndex.count() == 3)
    
    let results = try await loadedIndex.search(query: [1.0, 0.0, 0.0], k: 2, efSearch: 100)
    #expect(results.count == 2)
    #expect(results[0].id == 10)
}
