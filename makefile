MAKEFLAGS += --no-print-directory

debug := off # make run debug=on to enable debug mode

all: cformer
cformer:
	@cmake -B build -DCF_DEBUG=$(debug) && cmake --build build -t cformer

run: cformer
	@echo "\n==========program output==========\n"
	@build/cformer

test:
	@cmake -B build -DCF_TEST=on && cmake --build build -t test_tensor -t test # -- ARGS="-V"

clean:
	@rm -rf build
	@echo "clean done"
