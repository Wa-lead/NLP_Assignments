#!/bin/sh

echo "###############################################"
echo "First - Tokenize the texts"

cd Tools/

echo "perl tokenizer.pl -no-escape < ../English-Mix/UNCorpus.train > ../English-Mix/UNCorpus.train.tok"
perl tokenizer.pl -no-escape < ../English-Mix/UNCorpus.train > ../English-Mix/UNCorpus.train.tok

echo "perl tokenizer.pl -no-escape < ../English-Mix/UNCorpus.test > ../English-Mix/UNCorpus.test.tok"
perl tokenizer.pl -no-escape < ../English-Mix/UNCorpus.test > ../English-Mix/UNCorpus.test.tok

echo "perl tokenizer.pl -no-escape < ../English-Mix/Wiki.test > ../English-Mix/Wiki.test.tok"
perl tokenizer.pl -no-escape < ../English-Mix/Wiki.test > ../English-Mix/Wiki.test.tok
cd ..

echo "###############################################"
echo "Second - Create the LM"

echo "ngram-count -text English-Mix/UNCorpus.train.tok -order 3 -lm English-Mix/UNCorpus.train.tok.3.lm -addsmooth 0.01"
ngram-count -text English-Mix/UNCorpus.train.tok -order 3 -lm English-Mix/UNCorpus.train.tok.3.lm -addsmooth 0.01

echo "###############################################"
echo "Third - Compute Perplexity"

echo "ngram -lm English-Mix/UNCorpus.train.tok.3.lm -order 3 -ppl English-Mix/UNCorpus.test.tok"
ngram -lm English-Mix/UNCorpus.train.tok.3.lm -order 3 -ppl English-Mix/UNCorpus.test.tok

echo "ngram -lm English-Mix/UNCorpus.train.tok.3.lm -order 3 -ppl English-Mix/Wiki.test.tok"
ngram -lm English-Mix/UNCorpus.train.tok.3.lm -order 3 -ppl English-Mix/Wiki.test.tok

echo "###############################################"
echo "Fourth - For fun, generate some random UN resolution texts!"

echo "ngram -lm English-Mix/UNCorpus.train.tok.3.lm -gen 1"
ngram -lm English-Mix/UNCorpus.train.tok.3.lm -gen 1
