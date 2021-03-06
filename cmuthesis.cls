%> cmuthesis.cls: By Cristian Bravo-Lillo (cbravo@cmu.edu), based heavily on a previous one by dkoes (?), 5/10/2006.
\NeedsTeXFormat{LaTeX2e}
\DeclareOption{draft}{\def\mydraftmode{}\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClassWithOptions{book}

%> ---------------------------------------------------------------------------------------------------------------
%> Fonts.
\usepackage{times}
\usepackage[T1]{fontenc}
\usepackage{amssymb,textcomp}
\renewcommand{\familydefault}{\sfdefault}
%> ---------------------------------------------------------------------------------------------------------------
%> Other packages.
\usepackage{graphicx,color,multirow,colortbl,hhline,array,url,fancyhdr, setspace, enumerate, tikz, pgf-pie, algorithm, algorithmic, pgfplots, booktabs, siunitx, pgfplotstable, listings, rotating, hyperref}
\usepackage[normalem]{ulem}
\usepackage[margin=1.3in]{geometry}
\usepackage[font=small]{caption}
\usepackage{subcaption}
% \usepackage[markup=underlined]{changes}
\usepackage[final]{changes}

%> ---------------------------------------------------------------------------------------------------------------

%% dkoes - Here's a bunch of stuff taken from mjz-titlepage.sty that makes
%% a pretty CMU techreport acceptable title page
%% titlepage parameters
\def\title#1{\gdef\@title{#1}}
\def\author#1{\gdef\@author{#1}}
\def\affiliation#1{\gdef\@affiliation{#1}}
\def\Month#1{\gdef\@Month{#1}}
\def\Year#1{\gdef\@Year{#1}}
\def\permission#1{\gdef\@permission{#1}}
\def\disclaimer#1{\gdef\@disclaimer{#1}}
\def\keywords#1{\gdef\@keywords{#1}}
\def\bsdegree#1{\gdef\@bsdegree{B.S., #1}}
\def\msdegree#1{\gdef\@msdegree{M.S., #1}}

\def\maketitle{
%>----------------------------------------
%> Title page.
	\thispagestyle{empty}\vspace*{0.4in}
	\begin{center}\begin{large}
	\begin{spacing}{1.4}
	{\bf \LARGE \@title}\\[.5in]
	Submitted in partial fulfillment of the requirements for the degree of\\Doctor of Design\\ in \@ifundefined{@affiliation}{School of Architecture}{{\@affiliation}} \\[0.4in]
	{\bf \LARGE \@author}\\
	\vspace{0.2cm}
	\begin{spacing}{.7}{
		Carnegie Mellon University, Pittsburgh, PA\\}
	\vspace{0.3cm}
	\begin{small}
	\@ifundefined{@bsdegree}{\vspace*{0.8in}}{{\@bsdegree}\\}
	\@ifundefined{@msdegree}{\vspace*{0.8in}}{{\@msdegree}\\}
	\end{small}
	\end{spacing}
	\vfill
	\vspace{1cm}
	Doctoral Committee:\\
	{\normalsize \bf Dr. Erica Cochran Hameen Ph.D - \textit{Committee Chair}}\\ 
		\small{Assistant Professor Carnegie Mellon University, School or Architecture}\\
		\vspace{0.3cm}
	{\normalsize \bf Dr. Gwen DiPietro Ph.D}\\
		\small{Carnegie Mellon University, Civil \& Environmental Engineering} \\
		\vspace{0.3cm}
	{\normalsize \bf Dr. Arman Shehabi Ph.D}\\
		\small{Research Scientist \& Deputy Group Leader Sustainable Energy Systems Group
				Energy Analysis \\ \& Environmental Impacts Division Lawrence Berkeley National Laboratory} \\
				\vspace{0.3cm}
	{\normalsize \bf Dr. In\^es Azevedo Ph.D}\\
		\small{Associate Professor of Energy Resources Engineering, Stanford University \\ \&  Senior Fellow at the Woods Institute for the Environment }

	\vspace{0.8cm}
	\begin{LARGE}
	\@Month, \@Year
	\end{LARGE}
	\end{spacing}
	\end{large}\end{center}
	\clearpage
%>----------------------------------------
%> Copyright page.
	\thispagestyle{empty}
	\vspace*{5in}
	\vfill
	\noindent\copyright \@Year\ \@author.~~\@ifundefined{\usepackage{rotating}@permission}{\textit{All rights reserved.}}{{\@permission}}\\[2mm]
	\@ifundefined{@disclaimer}{The views and conclusions contained in this document are those of the author, and should not be interpreted as representing the official policies, either expressed or implied, of any sponsoring institution, current or past employers, the U.S. government, or any other entity.}{\@disclaimer}\\[2mm]
	\@ifundefined{@keywords}{}{\textbf{Keywords}: \@keywords\\}
	\clearpage
}

\newenvironment{dedication}{\cleardoublepage \thispagestyle{empty} \vspace*{\stretch{1}} \begin{center} \em}{\end{center} \vspace*{\stretch{3}} \clearpage}
\newenvironment{pseudochapter}[1]{
	\cleardoublepage
	\thispagestyle{empty}
	\vspace*{\stretch{1}}
	\begin{center}\large{\bf #1}\end{center}
	\begin{quotation}
}{
	\end{quotation}
	\vspace*{\stretch{3}}
	\clearpage
}
\newenvironment{abstract}{\begin{pseudochapter}{Abstract}}{\end{pseudochapter}}
\newenvironment{acknowledgments}{\begin{pseudochapter}{Acknowledgments}}{\end{pseudochapter}}


%> ---------------------------------------------------------------------------------------------------------------
%> Configuration: cbravo@cmu.edu, 20131202
\pagestyle{fancy}
\setlength{\pdfpagewidth}{8.5in}
\setlength{\pdfpageheight}{11in}

\renewcommand{\topfraction}{0.99} % be more aggressive about text around floats
\renewcommand{\floatpagefraction}{0.99}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\widowpenalty=1000
\clubpenalty=1000
\renewcommand{\chaptermark}[1]{ \markboth{#1}{} }
\renewcommand{\sectionmark}[1]{ \markright{#1}{} }
%> ---------------------------------------------------------------------------------------------------------------

%> ---------------------------------------------------------------------------------------------------------------
%> Definitions: cbravo@cmu.edu, 20131202
\newcommand{\smallurl}[1]{{\scriptsize\url{#1}}}
\newenvironment{question}{\vspace{0.05in}\begin{quote}\slshape}{\end{quote}\vspace{0.05in}}
\newcommand{\blindfootnote}[1]{{\let\thefootnote\relax\footnotetext{#1}}}

%> Comments
\newcommand{\ifnotadraft}[1]{
	\makeatletter
	\@ifundefined{mydraftmode}{#1}{}
	\makeatother
}
% \newcommand{\comment}[2][Comment]{\textcolor{red}{[#1: #2]}}
\newcommand{\colorcomment}[2]{\textcolor{#1}{[#2]}}
\newcommand{\edadd}[1]{\textcolor{red}{#1}}
\newcommand{\eddelete}[1]{\textcolor{red}{\sout{#1}}}
\newcommand{\edreplace}[2]{\textcolor{red}{\sout{#1}{#2}}}

\ifnotadraft{%
	\renewcommand{\comment}[2][Comment]{} %
	\renewcommand{\colorcomment}[2]{} %
	\renewcommand{\edadd}[1]{#1} %
	\renewcommand{\eddelete}[1]{} %
	\renewcommand{\edreplace}[2]{#2} %
}

%\setlength{\topsep}{1pt}
%\setlength{\itemsep}{1pt}
%\setlength{\parsep}{3pt}
\endinput
