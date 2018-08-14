# NOTE 
# To run, type   python wtoc.py
# To quit, use   control c    

import jsonpickle

fd = open("words_to_contexts_jsonpickle.txt", "r")
serialstr = fd.read()
fd.close()
words_to_contexts = jsonpickle.decode(serialstr)
print("\ndecoded")
print("words_to_contexts['create'] =", list(words_to_contexts['create']))
print("For keyboard entry, don't use quotes around the word.")

while(1):
	word = input("\nEnter a word: ")		# with keyboard entry, don't use quotes around word
	print(list(words_to_contexts[word]))