// swift-tools-version: 6.0

import PackageDescription

let useLocalFramework = false
let binaryTarget: Target

if useLocalFramework {
    binaryTarget = .binaryTarget(
        name: "HnswRS",
        path: "./build/libhnsw-rs.xcframework"
    )
} else {
    let releaseTag = "0.1.2"
    let releaseChecksum = "bd29625b6eb8af9ced60f6d301114bb0cca0f06414e973c938787a5a96b1cfac"
    binaryTarget = .binaryTarget(
        name: "HnswRS",
        url:
        "https://github.com/lhr0909/HNSW.swift/releases/download/\(releaseTag)/libhnsw-rs.xcframework.zip",
        checksum: releaseChecksum
    )
}

let package = Package(
    name: "HnswSwift",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15),
    ],
    products: [
        .library(
            name: "HnswSwift",
            targets: ["HnswSwift"]
        ),
    ],
    targets: [
        binaryTarget,
        .target(
            name: "HnswSwift",
            dependencies: ["HnswFFI"]
        ),
        .target(
            name: "HnswFFI",
            dependencies: ["HnswRS"]
        ),
        .testTarget(
            name: "HnswSwiftTests",
            dependencies: ["HnswSwift"]
        ),
    ]
)
