### top level path and include
OPTHOME := $(abspath ..)
include $(OPTHOME)/default.mk

### directory targets
PDF := least-squares.pdf
TEX := least-squares.tex

### generate pdf file by default
.DEFAULT_GOAL := $(PDF)

### clean up
.PHONY: clean
clean:
	$(RM) $(PDF) $(TEX)

.PHONY: cleanall
cleanall: clean
	$(RM) *.aux *.log *.out *.vrb
