#!/usr/bin/env swift
//
// KopisEnginePythonRunner
// Swift wrapper to execute the Python Kopis Engine
//

import Foundation

// Get the project root directory
// When run from Xcode, the executable is in DerivedData, so we need to find the source root
let fileManager = FileManager.default
let currentPath = fileManager.currentDirectoryPath

// Try to find the project root by looking for kopis_engine.py
func findProjectRoot() -> URL? {
    // Method 1: Use custom PROJECT_ROOT environment variable (set in Xcode build settings)
    if let projectRoot = ProcessInfo.processInfo.environment["PROJECT_ROOT"] {
        let projectRootURL = URL(fileURLWithPath: projectRoot)
        let pythonScript = projectRootURL.appendingPathComponent("kopis_engine.py")
        if fileManager.fileExists(atPath: pythonScript.path) {
            return projectRootURL
        }
    }
    
    // Method 2: Use Xcode environment variable SRCROOT if available
    if let srcRoot = ProcessInfo.processInfo.environment["SRCROOT"] {
        let srcRootURL = URL(fileURLWithPath: srcRoot)
        // SRCROOT points to KopisEnginePython directory, go up one level to project root
        let parentRoot = srcRootURL.deletingLastPathComponent()
        let parentPythonScript = parentRoot.appendingPathComponent("kopis_engine.py")
        if fileManager.fileExists(atPath: parentPythonScript.path) {
            return parentRoot
        }
        // Also check if kopis_engine.py is in SRCROOT itself (unlikely but possible)
        let pythonScript = srcRootURL.appendingPathComponent("kopis_engine.py")
        if fileManager.fileExists(atPath: pythonScript.path) {
            return srcRootURL
        }
    }
    
    // Method 3: Use the source file location (this Swift file) to find project root
    // The source file is at: KopisEnginePython/Sources/KopisEnginePythonRunner/main.swift
    if let sourceFile = ProcessInfo.processInfo.environment["SOURCE_ROOT"],
       fileManager.fileExists(atPath: sourceFile) {
        var sourcePath = URL(fileURLWithPath: sourceFile)
        // Navigate from source file to project root
        // Source: .../KopisEnginePython/Sources/KopisEnginePythonRunner/main.swift
        // Need: .../ (project root)
        for _ in 0..<4 {
            let pythonScript = sourcePath.appendingPathComponent("kopis_engine.py")
            if fileManager.fileExists(atPath: pythonScript.path) {
                return sourcePath
            }
            sourcePath = sourcePath.deletingLastPathComponent()
        }
    }
    
    // Method 4: Search from current working directory (DerivedData)
    var searchPath = URL(fileURLWithPath: currentPath)
    // Navigate up from DerivedData (can be many levels deep)
    for _ in 0..<15 {
        let pythonScript = searchPath.appendingPathComponent("kopis_engine.py")
        if fileManager.fileExists(atPath: pythonScript.path) {
            return searchPath
        }
        searchPath = searchPath.deletingLastPathComponent()
        // Stop if we've reached the filesystem root
        if searchPath.path == "/" {
            break
        }
    }
    
    // Method 5: Try relative to the executable location
    if let executablePath = Bundle.main.executablePath {
        var execPath = URL(fileURLWithPath: executablePath).deletingLastPathComponent()
        // Navigate up from build directory to project root
        // DerivedData path can be: .../DerivedData/ProjectName-hash/Build/Products/Debug
        for _ in 0..<15 {
            let pythonScript = execPath.appendingPathComponent("kopis_engine.py")
            if fileManager.fileExists(atPath: pythonScript.path) {
                return execPath
            }
            execPath = execPath.deletingLastPathComponent()
            // Stop if we've reached the filesystem root
            if execPath.path == "/" {
                break
            }
        }
    }
    
    // Method 6: Try common project locations relative to home directory
    if let homeDir = ProcessInfo.processInfo.environment["HOME"] {
        let projectsPath = URL(fileURLWithPath: homeDir).appendingPathComponent("Projects/Kopis-Engine")
        let pythonScript = projectsPath.appendingPathComponent("kopis_engine.py")
        if fileManager.fileExists(atPath: pythonScript.path) {
            return projectsPath
        }
    }
    
    return nil
}

guard let projectRoot = findProjectRoot() else {
    print("❌ Error: Could not find project root (looking for kopis_engine.py)")
    print("Current directory: \(currentPath)")
    print("\nEnvironment variables:")
    let env = ProcessInfo.processInfo.environment
    if let srcRoot = env["SRCROOT"] {
        print("  SRCROOT: \(srcRoot)")
    }
    if let projectRoot = env["PROJECT_ROOT"] {
        print("  PROJECT_ROOT: \(projectRoot)")
    }
    if let execPath = Bundle.main.executablePath {
        print("\nExecutable path: \(execPath)")
    }
    print("\nTried searching from:")
    print("  - Current directory: \(currentPath)")
    if let srcRoot = env["SRCROOT"] {
        print("  - SRCROOT: \(srcRoot)")
        print("  - SRCROOT parent: \(URL(fileURLWithPath: srcRoot).deletingLastPathComponent().path)")
    }
    exit(1)
}

// Debug output (can be removed later)
if ProcessInfo.processInfo.environment["DEBUG"] == "1" {
    print("✓ Found project root: \(projectRoot.path)")
}

let pythonScript = projectRoot.appendingPathComponent("kopis_engine.py")
let requirementsFile = projectRoot.appendingPathComponent("requirements.txt")
let venvDir = projectRoot.appendingPathComponent("KopisEnginePython/venv")

// Check if Python 3 is available (try common locations)
func findPython3() -> String? {
    let possiblePaths = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3"
    ]
    
    for path in possiblePaths {
        if fileManager.fileExists(atPath: path) {
            return path
        }
    }
    
    // Try using which command
    let whichTask = Process()
    whichTask.executableURL = URL(fileURLWithPath: "/usr/bin/which")
    whichTask.arguments = ["python3"]
    whichTask.standardOutput = Pipe()
    try? whichTask.run()
    whichTask.waitUntilExit()
    
    if let output = (whichTask.standardOutput as? Pipe)?.fileHandleForReading.readDataToEndOfFile(),
       let path = String(data: output, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
       !path.isEmpty,
       fileManager.fileExists(atPath: path) {
        return path
    }
    
    return nil
}

guard let python3Path = findPython3() else {
    print("❌ Error: python3 not found")
    print("Please install Python 3:")
    print("  brew install python3")
    exit(1)
}

// Check if the Python script exists
guard FileManager.default.fileExists(atPath: pythonScript.path) else {
    print("❌ Error: Python script not found at: \(pythonScript.path)")
    exit(1)
}

print("==========================================")
print("Kopis Engine - Python Version")
print("==========================================")
print("")

// Check Python version
let pythonVersionTask = Process()
pythonVersionTask.executableURL = URL(fileURLWithPath: python3Path)
pythonVersionTask.arguments = ["--version"]
pythonVersionTask.standardOutput = Pipe()
try? pythonVersionTask.run()
pythonVersionTask.waitUntilExit()

if let output = (pythonVersionTask.standardOutput as? Pipe)?.fileHandleForReading.readDataToEndOfFile(),
   let version = String(data: output, encoding: .utf8) {
    print("✓ Found: \(version.trimmingCharacters(in: .whitespacesAndNewlines))")
}

print("")

// Setup virtual environment if needed
if !FileManager.default.fileExists(atPath: venvDir.path) {
    print("Creating virtual environment...")
    let venvTask = Process()
    venvTask.executableURL = URL(fileURLWithPath: python3Path)
    venvTask.arguments = ["-m", "venv", venvDir.path]
    venvTask.currentDirectoryURL = venvDir.deletingLastPathComponent()
    try? venvTask.run()
    venvTask.waitUntilExit()
}

// Install dependencies if requirements.txt exists
if FileManager.default.fileExists(atPath: requirementsFile.path) {
    print("Installing/updating dependencies...")
    let pipPath = venvDir.appendingPathComponent("bin/pip").path
    
    // Upgrade pip
    let upgradePipTask = Process()
    upgradePipTask.executableURL = URL(fileURLWithPath: pipPath)
    upgradePipTask.arguments = ["install", "--upgrade", "pip", "--quiet"]
    try? upgradePipTask.run()
    upgradePipTask.waitUntilExit()
    
    // Install requirements
    let installTask = Process()
    installTask.executableURL = URL(fileURLWithPath: pipPath)
    installTask.arguments = ["install", "-r", requirementsFile.path, "--quiet"]
    try? installTask.run()
    installTask.waitUntilExit()
    
    print("✓ Dependencies ready")
}

print("")
print("Starting Kopis Engine...")
print("==========================================")
print("")

// Run the Python script using the virtual environment's Python
let pythonPath = venvDir.appendingPathComponent("bin/python3").path
let pythonExecutable = FileManager.default.fileExists(atPath: pythonPath) ? pythonPath : python3Path

let task = Process()
task.executableURL = URL(fileURLWithPath: pythonExecutable)
task.arguments = [pythonScript.path]
task.currentDirectoryURL = projectRoot

// Forward standard input/output/error
task.standardInput = FileHandle.standardInput
task.standardOutput = FileHandle.standardOutput
task.standardError = FileHandle.standardError

do {
    try task.run()
    task.waitUntilExit()
    exit(task.terminationStatus)
} catch {
    print("❌ Error running Python script: \(error)")
    exit(1)
}
