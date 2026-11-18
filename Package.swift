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
    let releaseChecksum = "702826672c82c23736e47dfb6d9346d803f5ccf9493211eff977b53dd1636e4f"
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
