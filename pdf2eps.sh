#!/bin/sh
# Convert PDF to encapsulated PostScript.
# usage:
# pdf2eps  <pdf file without ext>

pdf2ps $1.pdf 
ps2eps $1.ps

