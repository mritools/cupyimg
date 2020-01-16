.PHONY :  build develop test

all: develop

clean_pyc:
	-find . -name '*.py[co]' -exec rm {} \;

build: clean_pyc
	python setup.py build_ext --inplace

develop: build
	python -m pip install -e . -v  --no-build-isolation --no-use-pep517

test:
	py.test --pyargs cupyimg --cov=cupyimg --cov-report term-missing --cov-report html
