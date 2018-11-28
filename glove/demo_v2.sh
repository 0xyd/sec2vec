#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

#make




for i in "$@"
do
case $i in
-cf=*|--Corpus_File=*)
Corpus_File="${i#*=}"
shift # past argument=value
;;

-vf=*|--Vocab_File=*)
Vocab_File="${i#*=}"
shift # past argument=value
;;

-vmc=*|--Vocab_Min_Count=*)
Vocab_Min_Count="${i#*=}"
shift # past argument=value
;;

-vs=*|--Vector_Size=*)
Vector_Size="${i#*=}"
shift # past argument=value
;;

-w=*|--Window=*)
Window="${i#*=}"
shift # past argument=value
;;

-t=*|--Threads=*)
Threads="${i#*=}"
shift # past argument=value
;;

-irer=*|--iters=*)
iters="${i#*=}"
shift # past argument=value
;;

-xmax=*|--X_max=*)
X_max="${i#*=}"
shift # past argument=value
;;

-m=*|--Memory=*)
Memory="${i#*=}"
shift # past argument=value
;;

-sf=*|--Save_File=*)
Save_File="${i#*=}"
shift # past argument=value
;;

*)
# unknown option
;;
esac
done


# ./demo.sh --VOCAB_MIN_COUNT=123 --VECTOR_SIZE=456


echo $Corpus_File
echo $Vocab_File
echo $Vocab_Min_Count
echo $Vector_Size
echo $Window
echo $Threads
echo $iters
echo $X_max
echo $Memory

verbose=2
binary=2
cooccurrence_file=cooccurrence.bin
cooccurrence_shuf_file=cooccurrence.shuf.bin
builddir=build


echo "$ $builddir/vocab_count -min-count $Vocab_Min_Count -verbose $verbose < $Corpus_File > $Vocab_File"
$builddir/vocab_count -min-count $Vocab_Min_Count -verbose $verbose < $Corpus_File > $Vocab_File
#$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

echo "$ $builddir/cooccur -memory $Memory -vocab-file $Vocab_File -verbose $verbose -window-size $Window < $Corpus_File > $cooccurrence_file"
$builddir/cooccur -memory $Memory -vocab-file $Vocab_File -verbose $verbose -window-size $Window < $Corpus_File > $cooccurrence_file
echo "$ $builddir/shuffle -memory $Memory -verbose $verbose < $cooccurrence_file > $cooccurrence_shuf_file"
$builddir/shuffle -memory $Memory -verbose $verbose < $cooccurrence_file > $cooccurrence_shuf_file
echo "$ $builddir/glove -save-file $Save_File -threads $Threads -input-file $cooccurrence_shuf_file -x-max $X_max -iter $iters -vector-size $Vector_Size -binary $binary -vocab-file $Vocab_File -verbose $verbose"
$builddir/glove -save-file $Save_File -threads $Threads -input-file $cooccurrence_shuf_file -x-max $X_max -iter $iters -vector-size $Vector_Size -binary $binary -vocab-file $Vocab_File -verbose $verbose
echo "end to embedding..."






exit 0

