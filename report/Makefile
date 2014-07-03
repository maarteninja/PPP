default:
	pdflatex report.tex
	bibtex report.aux
	pdflatex report.tex
	pdflatex report.tex

all:
	default

clean:
	@rm -f *.pdf *.dvi *.aux *.log *.bbl *.blg
