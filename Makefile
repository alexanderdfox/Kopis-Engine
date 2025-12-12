.PHONY: build run clean test help swift-build swift-run python-run

# Default target
.DEFAULT_GOAL := help

# Swift project directory
SWIFT_DIR := KopisEngine
SWIFT_BUILD_DIR := $(SWIFT_DIR)/.build

# Python script
PYTHON_SCRIPT := kopis_engine.py

help: ## Show this help message
	@echo "Available targets:"
	@echo "  make build      - Build the Swift project"
	@echo "  make run        - Build and run the Swift project"
	@echo "  make gui        - Build and run the GUI version"
	@echo "  make xcode      - Open project in Xcode"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make test       - Run Swift tests"
	@echo "  make python     - Run the Python version"
	@echo "  make swift-build - Build Swift package only"
	@echo "  make swift-run  - Run Swift executable"
	@echo "  make help       - Show this help message"

build: swift-build ## Build the Swift project

swift-build: ## Build Swift package
	@echo "Building Swift package..."
	@which swift > /dev/null || (echo "Error: Swift is not installed. Install Xcode or Swift toolchain." && exit 1)
	cd $(SWIFT_DIR) && swift build
	@echo "✓ Build complete"

run: swift-run ## Build and run the Swift project

swift-run: swift-build ## Run Swift executable
	@echo "Running Kopis Engine (Swift)..."
	@cd $(SWIFT_DIR) && swift run KopisEngineApp || (echo "Error: Failed to run. Make sure the project builds successfully." && exit 1)

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	cd $(SWIFT_DIR) && swift package clean
	rm -rf $(SWIFT_BUILD_DIR)
	@echo "✓ Clean complete"

test: ## Run Swift tests
	@echo "Running Swift tests..."
	cd $(SWIFT_DIR) && swift test
	@echo "✓ Tests complete"

python: ## Run the Python version
	@echo "Running Kopis Engine (Python)..."
	python3 $(PYTHON_SCRIPT)

gui: ## Build and run the GUI version
	@echo "Building GUI version..."
	cd $(SWIFT_DIR) && swift build --product KopisEngineGUI
	@echo "Running GUI..."
	cd $(SWIFT_DIR) && swift run KopisEngineGUI

xcode: ## Open project in Xcode (uses Package.swift)
	@echo "Opening Kopis Engine in Xcode..."
	@echo ""
	@echo "Instructions:"
	@echo "1. Wait for Xcode to open"
	@echo "2. Select 'KopisEngineGUI' scheme from the top toolbar"
	@echo "3. Press ⌘R to run the GUI app"
	@echo ""
	@./scripts/open-xcode.sh

xcodeproj: ## Generate .xcodeproj file (requires xcodegen: brew install xcodegen)
	@echo "Generating .xcodeproj file..."
	@echo "Note: This requires xcodegen. Install with: brew install xcodegen"
	@cd KopisEngine && if command -v xcodegen &> /dev/null; then \
		xcodegen generate && echo "✓ Project generated: KopisEngine.xcodeproj"; \
	else \
		echo "⚠ xcodegen not found. Installing..."; \
		echo "Run: brew install xcodegen"; \
		echo "Then run: make xcodeproj"; \
		exit 1; \
	fi

fix-xcode: ## Fix Xcode compilation issues (clean, regenerate, test)
	@./scripts/fix-xcode.sh

xcode-open: ## Generate and open .xcodeproj file
	@if [ -f "KopisEngine/KopisEngine.xcodeproj" ]; then \
		echo "Opening KopisEngine.xcodeproj..."; \
		open KopisEngine/KopisEngine.xcodeproj; \
	elif command -v xcodegen &> /dev/null; then \
		echo "Generating .xcodeproj first..."; \
		$(MAKE) xcodeproj && open KopisEngine/KopisEngine.xcodeproj; \
	else \
		echo "⚠ .xcodeproj not found and xcodegen not installed."; \
		echo "Option 1: Install xcodegen and generate:"; \
		echo "  brew install xcodegen && make xcodeproj"; \
		echo ""; \
		echo "Option 2: Use Package.swift (no .xcodeproj needed):"; \
		echo "  make xcode"; \
	fi

# Development targets
dev: ## Run in development mode (with verbose output)
	cd $(SWIFT_DIR) && swift run KopisEngineApp --verbose

# Release build
release: ## Build release version
	@echo "Building release version..."
	cd $(SWIFT_DIR) && swift build -c release
	@echo "✓ Release build complete"

# Install (if needed for system-wide installation)
install: release ## Install to /usr/local/bin (requires sudo)
	@echo "Installing Kopis Engine..."
	sudo cp $(SWIFT_DIR)/.build/release/KopisEngineApp /usr/local/bin/kopis-engine
	@echo "✓ Installed to /usr/local/bin/kopis-engine"
