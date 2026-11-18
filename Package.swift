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
    let releaseChecksum = "133ee8bba414b6d2a42bb67d990050217eaf285aaeae3f9ebaeed21d5c270937"
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
