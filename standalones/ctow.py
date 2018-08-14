# NOTE 
# To run, type   python ctow.py
# To quit, use   control c    

import jsonpickle

fd = open("contexts_to_words_jsonpickle.txt", "r")
serialstr = fd.read()
fd.close()
contexts_to_words = jsonpickle.decode(serialstr)
print("\ndecoded")
print("contexts_to_words[('timothy', '_')] =", list(contexts_to_words["('timothy', '_')"]))
print("Note that input from command line must be in this form: ('timothy', '_')")
print("Don't use quotes outside the parens.")

while(1):
	context = input("\nEnter a context: ")
	print(list(contexts_to_words[context]))