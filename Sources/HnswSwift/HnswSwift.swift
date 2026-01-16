import Foundation
import HnswFFI

public enum HnswSwiftError: Error {
    case dimensionMismatch(expected: UInt32, got: UInt32)
    case emptyIndex
    case saveFailed(String)
    case loadFailed(String)
}

public enum HnswDistanceType: Sendable {
    case l2
    case cosine
    case dot
    case l1
}

public struct HnswSearchResult: Sendable {
    public let id: UInt64
    public let distance: Float
    
    init(from result: SearchResult) {
        self.id = result.id
        self.distance = result.distance
    }
}

public actor HnswIndex {
    private enum IndexVariant {
        case l2(HnswIndexL2)
        case cosine(HnswIndexCosine)
        case dot(HnswIndexDot)
        case l1(HnswIndexL1)
    }
    
    private let index: IndexVariant
    private let distanceType: HnswDistanceType
    
    public init(
        maxConnections: UInt32 = 16,
        maxElements: UInt64 = 10000,
        maxLayers: UInt32 = 16,
        efConstruction: UInt32 = 200,
        dimension: UInt32,
        distanceType: HnswDistanceType = .l2
    ) {
        self.distanceType = distanceType
        switch distanceType {
        case .l2:
            self.index = .l2(HnswIndexL2(
                maxNbConnection: maxConnections,
                maxElements: maxElements,
                maxLayer: maxLayers,
                efConstruction: efConstruction,
                dimension: dimension
            ))
        case .cosine:
            self.index = .cosine(HnswIndexCosine(
                maxNbConnection: maxConnections,
                maxElements: maxElements,
                maxLayer: maxLayers,
                efConstruction: efConstruction,
                dimension: dimension
            ))
        case .dot:
            self.index = .dot(HnswIndexDot(
                maxNbConnection: maxConnections,
                maxElements: maxElements,
                maxLayer: maxLayers,
                efConstruction: efConstruction,
                dimension: dimension
            ))
        case .l1:
            self.index = .l1(HnswIndexL1(
                maxNbConnection: maxConnections,
                maxElements: maxElements,
                maxLayer: maxLayers,
                efConstruction: efConstruction,
                dimension: dimension
            ))
        }
    }
    
    private init(index: IndexVariant, distanceType: HnswDistanceType) {
        self.index = index
        self.distanceType = distanceType
    }
    
    public static func load(
        directory: String,
        basename: String,
        dimension: UInt32,
        distanceType: HnswDistanceType
    ) throws -> HnswIndex {
        let variant: IndexVariant
        switch distanceType {
        case .l2:
            variant = .l2(try HnswIndexL2.load(directory: directory, basename: basename, dimension: dimension))
        case .cosine:
            variant = .cosine(try HnswIndexCosine.load(directory: directory, basename: basename, dimension: dimension))
        case .dot:
            variant = .dot(try HnswIndexDot.load(directory: directory, basename: basename, dimension: dimension))
        case .l1:
            variant = .l1(try HnswIndexL1.load(directory: directory, basename: basename, dimension: dimension))
        }
        return HnswIndex(index: variant, distanceType: distanceType)
    }
    
    public func insert(vector: [Float], id: UInt64) throws {
        switch index {
        case .l2(let idx):
            try idx.insert(data: vector, id: id)
        case .cosine(let idx):
            try idx.insert(data: vector, id: id)
        case .dot(let idx):
            try idx.insert(data: vector, id: id)
        case .l1(let idx):
            try idx.insert(data: vector, id: id)
        }
    }
    
    public func insertBatch(vectors: [[Float]], ids: [UInt64]) throws {
        switch index {
        case .l2(let idx):
            try idx.insertBatch(data: vectors, ids: ids)
        case .cosine(let idx):
            try idx.insertBatch(data: vectors, ids: ids)
        case .dot(let idx):
            try idx.insertBatch(data: vectors, ids: ids)
        case .l1(let idx):
            try idx.insertBatch(data: vectors, ids: ids)
        }
    }
    
    public func search(query: [Float], k: UInt32, efSearch: UInt32? = nil) throws -> [HnswSearchResult] {
        let ef = efSearch ?? max(k, 50)
        let results: [SearchResult]
        switch index {
        case .l2(let idx):
            results = try idx.search(query: query, k: k, efSearch: ef)
        case .cosine(let idx):
            results = try idx.search(query: query, k: k, efSearch: ef)
        case .dot(let idx):
            results = try idx.search(query: query, k: k, efSearch: ef)
        case .l1(let idx):
            results = try idx.search(query: query, k: k, efSearch: ef)
        }
        return results.map { HnswSearchResult(from: $0) }
    }
    
    public func count() throws -> UInt64 {
        switch index {
        case .l2(let idx):
            return try idx.len()
        case .cosine(let idx):
            return try idx.len()
        case .dot(let idx):
            return try idx.len()
        case .l1(let idx):
            return try idx.len()
        }
    }
    
    public func isEmpty() throws -> Bool {
        switch index {
        case .l2(let idx):
            return try idx.isEmpty()
        case .cosine(let idx):
            return try idx.isEmpty()
        case .dot(let idx):
            return try idx.isEmpty()
        case .l1(let idx):
            return try idx.isEmpty()
        }
    }
    
    public func getDimension() -> UInt32 {
        switch index {
        case .l2(let idx):
            return idx.getDimension()
        case .cosine(let idx):
            return idx.getDimension()
        case .dot(let idx):
            return idx.getDimension()
        case .l1(let idx):
            return idx.getDimension()
        }
    }
    
    public func save(directory: String, basename: String) throws {
        switch index {
        case .l2(let idx):
            try idx.save(directory: directory, basename: basename)
        case .cosine(let idx):
            try idx.save(directory: directory, basename: basename)
        case .dot(let idx):
            try idx.save(directory: directory, basename: basename)
        case .l1(let idx):
            try idx.save(directory: directory, basename: basename)
        }
    }
    
    public func setSearchingMode(enabled: Bool) throws {
        switch index {
        case .l2(let idx):
            try idx.setSearchingMode(enabled: enabled)
        case .cosine(let idx):
            try idx.setSearchingMode(enabled: enabled)
        case .dot(let idx):
            try idx.setSearchingMode(enabled: enabled)
        case .l1(let idx):
            try idx.setSearchingMode(enabled: enabled)
        }
    }
}
