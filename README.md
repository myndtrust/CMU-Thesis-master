# Eric Kumar Dissertation Paper.

## Quick Instructions

1. Download the zip file by clicking on the 'Download ZIP' button, to the right column. Uncompress the file.
1. Edit the files referenced below in "Important files" to suit your needs.
1. Open a terminal, go to the folder where you uncompressed the zip file, and type `make`. If everything goes fine, you should see soon the file 'thesis.pdf'. To clean up the folder, type `make clean`.

## Not-so-quick instructions

### Foreword

The first draft of this manuscript is being written through the 2020 COVID-19 stay in place orders set all over the world. However the discussed work spans over five years. During the five years, I have concluded that given the myriad of available type-setting tools, \LaTeX is the most expressive for technical writing. The template I use is a real gem. To any new starting their journey I suggest you adapt this work as a template for your as early as possible. Latex provides a level of organization that other type-setting programs just can't.


After some exploration with other forms of type-setting tools I chose to write this manuscript in Latex. My experience with Latex is that it has a steep learning curve, but for me it was the stepping stone into programming languages. The good thing with latex is that with it what you type is what you get. There are tons of resourcefully libraries that make programmatic publishable quality documents.

<!--Before going all in with Latex, I tried Markdown, Jupyter Notebooks, and of course Word. Markdown is cool for routine notes. Jupyter Notebook was not worth the effort. While Word was to con trained and is not conducive to programmatic writing.-->

<!--This is about my sixth document in Latex, first using this handy template from [@tamabravolillo](https://twitter.com/tamabravolillo).-->

<!--During 2013 I spent quite a while trying to understand the multiple requirements to put together my PhD thesis for Engineering and Public Policy (EPP) at Carnegie Mellon University (CMU).  Besides having many tables, lots of questions coming from surveys, graphics, statistical data, and a lot of code to put in, I also had to deal with a lot of editorial requirements (e.g., "*this is how* your cover page should look like").  Since there are no official templates for a phd thesis (which makes no sense to me since it's something we all PhD students have to go through), here's my attempt to save you the pain to do it.-->

<!--This template (actually, a set of files needed to give structure to the thesis) has a lof of filler texts, just to give you an example of how to structure your own thesis document. You can safely remove most of the texts.-->

<!--While there are surely many ways to do what I have done in this template, I strongly suggest that you create a set of rules for yourself and stick to them firmly (e.g., putting each chapter in a separate folder, naming all the images in a certain way, etc.) Writing your thesis can be very stressful, and many things can go wrong in stressful times (like deleting important files unrecoverably).-->

<!--The most important two pieces of advice that I can give for writing your thesis document are:-->

<!--1. **Back up everything twice out of your own computer**.-->
<!--1. **Be organized**.-->

### Important files

1. *thesis.tex*: This is the main file of this thesis. It contains references to all the chapters, appendices, and other needed commands.
1. *content/frontmatter.tex*: This is where most of the details are: the title of the author, author's name, author's degrees (goes on the cover page), the date of the authors graduation, copyright permission, keywords of the thesis, the abstract, the dedications, and the acknowledgements.
1. *content/macros.tex*: This is where you put all the special LaTeX packages that you need, and all those definitions that are repeated all over your work, but that nonetheless may change often.
1. *content/references.bib*: This is where your bibliographic references go (BibTeX format).

Except for these files, you should not need to change any other files.

### Structure of thesis

The starting file for this thesis is thesis.tex. This file includes references to the chapters and main files. Some lines in this file are not meant to be changed, others are.  The file is commented to indicate what you should change.

In this template, all the content belonging to a chapter has been put into a separate folder (a "chapter folder"). There is one folder (chp-main) that contains the "special" chapters: Introduction, Related Work and Conclusions, but if you don't like this you may tweak it as you please.

Each chapter folder has two subfolders: `images/` and `content/`. The first one is meant to contain all the images included in that chapter.  The second one is meant to contain latex snippets that will actually embed images or other graphical files into your thesis.

Each chapter folder contains one file with the body of the chapter; which in this package has the same name of the folder (e.g., chp-studyone contains the file chp-studyone.tex). This file starts with the usual chapter commands:

```tex
\chapter[Short chapter name]{Long chapter name}
\label{chp:shortlabelforchapter}
```

If the chapter is based on a paper you wrote, you should cite your own paper with the \blindfootnote{} command. For example:

```tex
\blindfootnote{This paper is based on Doe et al., 1999 \cite{DoeEtAl1999}.}
```

After that, you may put the content of your chapter, organized with the appropriate LaTeX commands: `\section`, `\subsection`, `\subsubsection`, etc. Don't forget to include a `\label{}` command with a short label if you want to cross-reference that section somewhere else in the document.

### How to include images or other graphical stuff

If you want to include an image, I strongly recommend doing the following:

1. Put the image into the images/ folder. For example, images/myimage.png

2. Create a file within the content/ folder (e.g., content/fig-myimage.tex) with commands to include and reference the image.  For example:
```tex
\begin{figure}\centering
\includegraphics[scale=0.8]{mychapter/images/myimage.png}
\caption[Short caption, to appear in the table of contents]{Long caption for the image, to appear beneath the image itself}
\label{LabelForTheImage}
\end{figure}
```

3. Include the previous file within the chapter:
```tex
\input{content/fig-myimage}
```

### How to make comments (actually, how to use editing commands)

There are a few commands to add editing capabilities to your thesis file. When you're editing your thesis, it's useful to have a mechanism to make comments, and introduce some stuff that may or may not end up in the thesis. Maybe an advisor wants to read the LaTeX file and make comments right there!

When writing and compiling this thesis, you can use two modes: *normal* and *draft*. The way to make it a draft is passing the option 'draft' to the documentclass command, like this:

```tex
\documentclass[11pt,draft]{cmuthesis}
```

To return to normal mode, simply take out the option 'draft'. All the following commands have different behavior depending on the mode you are:

`\comment{text}:`
- In draft mode: 'text' is displayed in red font, between square brackets, and preceded by the word 'Comment'.
- In normal mode: 'text' is not displayed.

`\edadd{text}: ('EDitor ADD')`
- In draft mode: 'text' is displayed in red font.
- In normal mode: 'text' is displayed in regular font.

`\eddelete{text}: ('EDitor DELETE')`
- In draft mode: 'text' is displayed in red font and crossed out.
- In normal mode: 'text' is not displayed

`\edreplace{text}{replacement}:`
- In draft mode: 'text' is displayed in red font and crossed out. 'replacement' is displayed in red font.
- In normal mode: only 'replacement' is displayed in regular font.


## For more information

### Official sources of information

My department (or at least my program and Chair) did not require any specific template. The resulting of a free form structure has been framework by Carnegie Mellon University's EPP school template with slight modifications for use on Overleaf.


### How do I get the latest version of this template?

1. Go to https://github.com/cristianbravolillo/EPP-CMU-Thesis/

2. Click on the button "Download ZIP", in the bottom of the column to the right.


### Dependencies

I use this template on Overleaf. The template and Overleaf are well integrated. The rigor of GitHub source file managment is a great feature.

### License

The document uses a non-official EPP-CMU Thesis Template by Cristian Bravo-Lillo which is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To see a human-readable version of the license, please go to http://creativecommons.org/licenses/by-nc-sa/4.0/.

### Author Contact, bugs and acknowledgments

<!--If you find a bug in the template (that is, something that is wrong based on the department or university guidelines), please leave me a note in GitHub:-->

1. Go to https://github.com/cristianbravolillo/EPP-CMU-Thesis/issues
1. Click on the button 'New issue' (green button to the right)
1. Give it a name and a longer description, and click on 'Submit new issue'.

<!--If you want to thank me, you may send me a tweet to [@tamabravolillo](https://twitter.com/tamabravolillo) or leave a message at [LinkedIn](https://www.linkedin.com/in/cristianbravolillo/).-->

Author: Eric M. Kumar &copy;
