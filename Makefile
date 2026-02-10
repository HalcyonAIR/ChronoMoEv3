.PHONY: test test-chronomoe clean

# Run ChronoMoE integration tests
test: test-chronomoe

test-chronomoe:
	@echo "Running ChronoMoE integration validation..."
	python3 -m chronomoe_integration.tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
