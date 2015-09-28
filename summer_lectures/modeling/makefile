### top level path and include
OPTHOME := $(abspath ..)
include $(OPTHOME)/default.mk

### directory targets
PDF := modeling.pdf
TEX := modeling.tex

### generate pdf file by default
.DEFAULT_GOAL := $(PDF)

### clean up
.PHONY: clean
clean:
	$(RM) $(PDF) $(TEX)

.PHONY: cleanall
cleanall: clean