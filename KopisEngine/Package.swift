// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "KopisEngine",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "KopisEngine",
            targets: ["KopisEngine"]),
        .executable(
            name: "KopisEngineApp",
            targets: ["KopisEngineApp"]),
        .executable(
            name: "KopisEngineGUI",
            targets: ["KopisEngineGUI"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "KopisEngine",
            dependencies: []),
        .executableTarget(
            name: "KopisEngineApp",
            dependencies: ["KopisEngine"]),
        .executableTarget(
            name: "KopisEngineGUI",
            dependencies: ["KopisEngine"],
            resources: [
                .process("Shaders.metal")
            ],
            linkerSettings: [
                .linkedFramework("SwiftUI", .when(platforms: [.macOS])),
                .linkedFramework("AppKit", .when(platforms: [.macOS])),
                .linkedFramework("Metal", .when(platforms: [.macOS])),
                .linkedFramework("MetalKit", .when(platforms: [.macOS])),
                .linkedFramework("AVFoundation", .when(platforms: [.macOS]))
            ])
    ]
)
