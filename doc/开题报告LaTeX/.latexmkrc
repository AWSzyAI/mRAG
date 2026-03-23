#!/usr/bin/env perl
# Use xelatex for Chinese document (ctex package)
$pdf_mode = 5;  # 5 = use xelatex
$latex = 'xelatex -interaction=nonstopmode -synctex=1 %S';
$bibtex = 'bibtex %O %B';
$max_repeat = 5;
