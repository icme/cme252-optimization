### top level path and include
OPTHOME := $(abspath ..)
include $(OPTHOME)/default.mk

### directory targets
PDF := course-logistics.pdf
TEX := course-logistics.tex

### generate pdf file by default
.DEFAULT_GOAL := $(PDF)

### clean up
.PHONY: clean
clean:
	$(RM) $(PDF) $(TEX)

.PHONY: cleanall
cleanall: clean
