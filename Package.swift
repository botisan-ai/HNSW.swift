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
    let releaseTag = "0.2.0"
    let releaseChecksum = "9c9cef077cee73af1d084e629c56893a799ece28fc8a39b6adc95e6465f7c786"
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
