rem Create your own text.txt file with some text.


darknet.exe rnn train cfg/lstm.train.cfg -file text.txt


rem darknet.exe rnn train cfg/lstm.train.cfg backup/lstm.backup -file text.txt


pause

darknet.exe rnn generate cfg/lstm.train.cfg backup/lstm.backup -len 500 -seed apple

darknet.exe rnn generate cfg/lstm.train.cfg backup/lstm.backup -len 500 -seed apple > text_gen.txt

pause