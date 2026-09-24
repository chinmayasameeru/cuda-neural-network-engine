# CUDA Neural Network Engine — Build System

.PHONY: all build test benchmark clean

all: build

build:
	python setup.py build_ext --inplace

test:
	python tests/test_engine.py

benchmark:
	python benchmarks/benchmark.py

lint:
	black --check tests benchmarks || true
	flake8 tests benchmarks --max-line-length=100 || true

format:
	black tests benchmarks

clean:
	rm -rf build dist *.egg-info __pycache__ .pytest_cache
	find . -name "*.so" -delete
	find . -name "*.o" -delete
	find . -name "*.pyc" -delete
