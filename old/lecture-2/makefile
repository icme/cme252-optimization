### top level path and include
OPTHOME := $(abspath ..)
include $(OPTHOME)/default.mk

### directory targets
PDF := opt-lec-2.pdf
TEX := opt-lec-2.tex

### generate pdf file by default
.DEFAULT_GOAL := $(PDF)

### This pdf depends on some generated figures from lecture-1
### create make dependency for graph sequence
$(PDF): ../lecture-1/fig/graph-sequence-1.pdf

### split the graph-sequence.pdf
../lecture-1/fig/graph-sequence-1.pdf: ../lecture-1/fig/graph-sequence.pdf
	pdfseparate $< ../lecture-1/fig/graph-sequence-%d.pdf 

### clean up
.PHONY: clean
clean:
	$(RM) $(PDF) $(TEX)

.PHONY: cleanall
cleanall: clean
	$(RM) ../lecture-1/fig/graph-sequence-*.pdf
