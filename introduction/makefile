### top level path and include
OPTHOME := $(abspath ..)
include $(OPTHOME)/default.mk

### directory targets
PDF := introduction.pdf
TEX := introduction.tex

### generate pdf file by default
.DEFAULT_GOAL := $(PDF)

### create make dependency for graph sequence
$(PDF): fig/graph-sequence-1.pdf

### split the graph-sequence.pdf
fig/graph-sequence-1.pdf: fig/graph-sequence.pdf
	pdfseparate $< fig/graph-sequence-%d.pdf 

### clean up
.PHONY: clean
clean:
	$(RM) $(PDF) $(TEX)
	$(RM) cable.pdf cable.tex

.PHONY: cleanall
cleanall: clean
	$(RM) fig/graph-sequence-*.pdf
