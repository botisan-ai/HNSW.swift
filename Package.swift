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
    let releaseTag = "0.2.4"
    let releaseChecksum = "a677278d3fcd5260da3acf8cff2c1a4c3c444a3a6b9ee70ff9c81b96a8073101"
    binaryTarget = .binaryTarget(
        name: "HnswRS",
        url:
        "https://github.com/botisan-ai/HNSW.swift/releases/download/\(releaseTag)/libhnsw-rs.xcframework.zip",
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
