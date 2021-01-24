all:
	emcc --bind -o ./dist/rand/beta.js ./src/rand/beta.cpp
clean:
	rm -rf ./src