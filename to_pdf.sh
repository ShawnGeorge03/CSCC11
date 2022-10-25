jupyter nbconvert --to latex $1.ipynb
sed -r -i 's/documentclass\[11pt\]\{article\}/documentclass[8pt]{extarticle}/' $1.tex
sed -r -i 's/geometry\{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}/geometry{verbose,tmargin=0.5in,bmargin=0.5in,lmargin=0.3in,rmargin=0.3in}/' $1.tex
pdflatex $1;