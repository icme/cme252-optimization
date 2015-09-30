all: hw0.pdf

### pattern rule to generate pdf from markdown
%.pdf: %.md
	pandoc $< -o $@

.PHONY: clean
clean:
	-rm *.pdf *.pyc