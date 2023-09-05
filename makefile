MAKEFLAGS += --no-print-directory

# make run: run the main program
# make mnist: run the mnist example
# make test: run the test cases
# make clean: clean the build directory

debug := off # make run debug=on to enable debug mode

all: cformer
cformer:
	@cmake -B build -DCF_DEBUG=$(debug) && cmake --build build -t cformer

mnist:
	@cmake -B build -DCF_MNIST=on && cmake --build build -t cformer_mnist
	@echo "\n==========program output==========\n"
	@build/cformer_mnist

chargen:
	@cmake -B build -DCF_CHARGEN=on && cmake --build build -t cformer_chargen
	@echo "\n==========program output==========\n"
	@build/cformer_chargen

run: cformer
	@echo "\n==========program output==========\n"
	@build/cformer

test:
	@cmake -B build -DCF_DEBUG=$(debug) -DCF_TEST=on && cmake --build build -t test_cformer -t test # -- ARGS="-V"

clean:
	@rm -rf build
	@rm -rf core.*
	@echo "clean done"
