### make defaults
.SUFFIXES:

### pandoc settings
PANDOC := pandoc
PANDOC_FLAGS := -t beamer
PANDOC_FLAGS += -V classoption="aspectratio=169"
PANDOC_FLAGS += -H $(OPTHOME)/tex/formatting.tex
PANDOC_FLAGS += -H $(OPTHOME)/tex/talk_defs.tex
PANDOC_FLAGS += --slide-level 2
PANDOC_FLAGS += -s
PANDOC_DEPS := $(OPTHOME)/tex/formatting.tex $(OPTHOME)/tex/talk_defs.tex

### pattern rule to generate pdf from markdown
%.pdf: %.md $(PANDOC_DEPS)
	$(PANDOC) $(PANDOC_FLAGS) $< -o $@

### pattern rule to generate tex from markdown
%.tex: %.md $(PANDOC_DEPS)
	$(PANDOC) $(PANDOC_FLAGS) $< -o $@
