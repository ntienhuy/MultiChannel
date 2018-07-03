import numpy as np
import re
import numpy

	
cvs = [1,2,3,4,5]
for cv in cvs:
	word2id= {}
	id2word={}
	index = 1
	maxlen = 0
	avglen = 0
	count100 = 0
	#read file
	train_pos_file = "data/VS/Data_not_token/Fold_"+ str(cv)+"/train_nhan_1.txt"
	train_neg_file = "data/VS/Data_not_token/Fold_"+ str(cv)+"/train_nhan_0.txt"
	train_neu_file = "data/VS/Data_not_token/Fold_"+ str(cv)+"/train_nhan_2.txt"

	test_pos_file = "data/VS/Data_not_token/Fold_"+ str(cv)+"/test_nhan_1.txt"
	test_neg_file = "data/VS/Data_not_token/Fold_"+ str(cv)+"/test_nhan_0.txt"
	test_neu_file = "data/VS/Data_not_token/Fold_"+ str(cv)+"/test_nhan_2.txt"



	open_files = [train_pos_file, train_neg_file, train_neu_file, test_pos_file, test_neg_file, test_neu_file]

	#save file
	train_pos_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/train_pos"
	train_neg_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/train_neg"
	train_neu_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/train_neu"

	test_pos_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/test_pos"
	test_neg_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/test_neg"
	test_neu_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/test_neu"

	save_files = [train_pos_save, train_neg_save, train_neu_save, test_pos_save, test_neg_save, test_neu_save]

	for open_file, save_file in zip(open_files,save_files):
		pos = []
		file = open(open_file, 'r')

		for aline in file.readlines():
		    aline = aline.replace('\n', "")
		    ids = np.array([], dtype='int32')
		    for word in aline.split(' '):
		        word = word.lower()
		        if word in word2id:
		            ids = np.append(ids, word2id[word])
		        else:
		            if word != '':
		                # print (word, "not in vocalbulary")
		                word2id[word] = index
		                id2word[index] = word
		                ids = np.append(ids, index)
		                index = index + 1
		    if len(ids) > 0:
		        pos.append(ids)

		file.close()
		print len(pos)
		np.save(save_file, pos)
		for li in pos:
		    if maxlen < len(li):
		        maxlen = len(li)
		    avglen += len(li)
		    if len(li) > 250:
		        count100+=1

	print len(word2id)
	print ("maxlen",maxlen)
	print ("maxlen250",count100)
