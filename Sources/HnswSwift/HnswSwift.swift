import Foundation
import HnswFFI

public enum HnswSwiftError: Error {
    case dimensionMismatch(expected: UInt32, got: UInt32)
    case emptyIndex
    case saveFailed(String)
    case loadFailed(String)
    case compactMissingConfig
    case invalidInput(String)
}

public typealias HnswDistanceType = HnswFFI.DistanceType

public struct HnswSearchResult: Sendable {
    public let id: UInt64
    public let distance: Float
    
    init(from result: SearchResult) {
        self.id = result.id
        self.distance = result.distance
    }
}

public typealias HnswIndexConfig = HnswFFI.HnswIndexConfig

public extension HnswFFI.HnswIndexConfig {
    init(
        maxConnections: UInt32 = 16,
        maxElements: UInt64 = 10000,
        maxLayers: UInt32 = 16,
        efConstruction: UInt32 = 200,
        dimension: UInt32,
        distanceType: HnswFFI.DistanceType = .cosine
    ) {
        self.init(
            maxNbConnection: maxConnections,
            maxElements: maxElements,
            maxLayer: maxLayers,
            efConstruction: efConstruction,
            dimension: dimension,
            distance: distanceType
        )
    }
}

public actor HnswIndex {
    private var index: HnswFFI.HnswIndex
    private let distanceType: HnswDistanceType
    private var config: HnswIndexConfig?
    private var deletedIds: Set<UInt64>
    
    public init(
        maxConnections: UInt32 = 16,
        maxElements: UInt64 = 10000,
        maxLayers: UInt32 = 16,
        efConstruction: UInt32 = 200,
        dimension: UInt32,
        distanceType: HnswDistanceType = .cosine
    ) {
        let config = HnswIndexConfig(
            maxConnections: maxConnections,
            maxElements: maxElements,
            maxLayers: maxLayers,
            efConstruction: efConstruction,
            dimension: dimension,
            distanceType: distanceType
        )
        self.index = HnswFFI.HnswIndex(config: config)
        self.distanceType = distanceType
        self.config = config
        self.deletedIds = []
    }
    
    private init(
        index: HnswFFI.HnswIndex,
        distanceType: HnswDistanceType,
        deletedIds: Set<UInt64>,
        config: HnswIndexConfig?
    ) {
        self.index = index
        self.distanceType = distanceType
        self.deletedIds = deletedIds
        self.config = config
    }

    private static func tombstoneURL(directory: String, basename: String) -> URL {
        URL(fileURLWithPath: directory)
            .appendingPathComponent("\(basename).deleted")
    }

    private static func loadTombstones(directory: String, basename: String) -> Set<UInt64> {
        let url = tombstoneURL(directory: directory, basename: basename)
        guard let data = try? Data(contentsOf: url), !data.isEmpty else {
            return []
        }

        var ids: Set<UInt64> = []
        let chunkSize = MemoryLayout<UInt64>.size
        guard data.count % chunkSize == 0 else {
            return []
        }

        data.withUnsafeBytes { raw in
            for offset in stride(from: 0, to: data.count, by: chunkSize) {
                let value = raw.load(fromByteOffset: offset, as: UInt64.self)
                ids.insert(UInt64(littleEndian: value))
            }
        }

        return ids
    }

    private static func saveTombstones(_ ids: Set<UInt64>, directory: String, basename: String) throws {
        let url = tombstoneURL(directory: directory, basename: basename)
        var data = Data()
        data.reserveCapacity(ids.count * MemoryLayout<UInt64>.size)

        for id in ids.sorted() {
            var value = id.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }

        try data.write(to: url, options: .atomic)
    }
    
    public static func load(
        directory: String,
        basename: String,
        dimension: UInt32,
        distanceType: HnswDistanceType,
        config: HnswIndexConfig? = nil
    ) throws -> HnswIndex {
        if let config = config {
            guard config.dimension == dimension else {
                throw HnswSwiftError.dimensionMismatch(expected: dimension, got: config.dimension)
            }
            guard config.distance == distanceType else {
                throw HnswSwiftError.invalidInput("Config distance type does not match load distance type.")
            }
        }
        let loadConfig = config ?? HnswIndexConfig(dimension: dimension, distanceType: distanceType)
        let ffiIndex = try HnswFFI.HnswIndex.load(
            directory: directory,
            basename: basename,
            config: loadConfig
        )
        let deletedIds = loadTombstones(directory: directory, basename: basename)
        return HnswIndex(index: ffiIndex, distanceType: distanceType, deletedIds: deletedIds, config: config)
    }
    
    public func insert(vector: [Float], id: UInt64) throws {
        deletedIds.remove(id)
        try index.insert(data: vector, id: id)
    }
    
    public func insertBatch(vectors: [[Float]], ids: [UInt64]) throws {
        for id in ids {
            deletedIds.remove(id)
        }
        try index.insertBatch(data: vectors, ids: ids)
    }
    
    public func search(query: [Float], k: UInt32, efSearch: UInt32? = nil) throws -> [HnswSearchResult] {
        let ef = efSearch ?? max(k, 50)
        let extra = min(UInt32(deletedIds.count), k)
        let searchK = k + extra
        let results = try index.search(query: query, k: searchK, efSearch: ef)
        let filtered = results
            .filter { !deletedIds.contains($0.id) }
            .prefix(Int(k))
        return filtered.map { HnswSearchResult(from: $0) }
    }

    public func delete(id: UInt64) {
        deletedIds.insert(id)
    }

    public func delete(ids: [UInt64]) {
        for id in ids {
            deletedIds.insert(id)
        }
    }

    public func compact(config: HnswIndexConfig? = nil) throws {
        let resolvedConfig = config ?? self.config
        guard let resolvedConfig else {
            throw HnswSwiftError.compactMissingConfig
        }
        let currentDimension = getDimension()
        guard resolvedConfig.dimension == currentDimension else {
            throw HnswSwiftError.dimensionMismatch(expected: currentDimension, got: resolvedConfig.dimension)
        }
        guard resolvedConfig.distance == distanceType else {
            throw HnswSwiftError.invalidInput("Config distance type does not match index distance type.")
        }
        index = try index.compact(deletedIds: Array(deletedIds), config: resolvedConfig)
        deletedIds.removeAll()
        self.config = resolvedConfig
    }
    
    public func count() throws -> UInt64 {
        let total = try index.len()
        let deleted = UInt64(deletedIds.count)
        return total > deleted ? (total - deleted) : 0
    }
    
    public func isEmpty() throws -> Bool {
        return try count() == 0
    }
    
    public func getDimension() -> UInt32 {
        index.getDimension()
    }
    
    public func save(directory: String, basename: String) throws {
        try index.save(directory: directory, basename: basename)
        do {
            try Self.saveTombstones(deletedIds, directory: directory, basename: basename)
        } catch {
            throw HnswSwiftError.saveFailed(error.localizedDescription)
        }
    }
    
    public func setSearchingMode(enabled: Bool) throws {
        try index.setSearchingMode(enabled: enabled)
    }
}
