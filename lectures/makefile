.DEFAULT_GOAL := all

.PHONY: all
all:
	$(MAKE) -C introduction
	$(MAKE) -C least-squares
	$(MAKE) -C convexity
	$(MAKE) -C modeling

.PHONY: clean
clean:
	$(MAKE) -C introduction clean
	$(MAKE) -C least-squares clean
	$(MAKE) -C convexity clean
	$(MAKE) -C modeling clean

.PHONY: cleanall
cleanall:
	$(MAKE) -C introduction cleanall
	$(MAKE) -C least-squares cleanall
	$(MAKE) -C convexity cleanall
	$(MAKE) -C modeling clean

