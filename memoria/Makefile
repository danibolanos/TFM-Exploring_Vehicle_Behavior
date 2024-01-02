# Name of the document
DOC=main
FILES=main.tex

# Tools for compiling
#RM=yes | rm -rf # it is usually well defined by make
LATEX=pdflatex
BIBER=biber
#READER=gedit
READER=emacs
#PDFREADER=acroread
#PDFREADER=evince
PDFREADER=mupdf
MAKE=make
CD=cd

# ------------------------------------------------------

# make: by default build the pdf
all: $(DOC).pdf

# make rebuild: remove just the pdf, and rebuild the project
rebuild: cleanPDF $(DOC).pdf

# make log: display the log of LATEX
log:
	$(READER) "$(DOC).log"

# make clean: remove auxiliary files
clean:
	$(RM) -R "$(DOC).toc" "$(DOC).nav" "$(DOC).out" "$(DOC).aux" "$(DOC).blg" "$(DOC).bbl" "$(DOC).bcf" "$(DOC).log" "$(DOC).lof" "$(DOC).lot" "$(DOC).run.xml" "$(DOC)-blx.bib" *~

# make clean: remove auxiliary files, the pdf, and temporal files
cleanAll: clean cleanPDF
	$(RM) -R *~

# make cleanPDF: remove just the pdf
cleanPDF:
	$(RM) "$(DOC).pdf"

# make read: display the final pdf
read: $(DOC).pdf
	$(PDFREADER) "$(DOC).pdf"

# make file.pdf: build the pdf
$(DOC).pdf: $(DOC).tex $(FILES)
	$(LATEX) "$(DOC).tex"
	$(BIBER) "$(DOC)"
	$(LATEX) "$(DOC).tex"
	$(LATEX) "$(DOC).tex"

